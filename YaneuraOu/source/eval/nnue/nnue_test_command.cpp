// NNUE評価関数に関するUSI拡張コマンド

#include "../../config.h"

#if defined(ENABLE_TEST_CMD) && defined(EVAL_NNUE)

#include "../../extra/all.h"
#include "../../evaluate.h"
#include "evaluate_nnue.h"
#include "nnue_test_command.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <set>
#include <string>
#include <string_view>

namespace Eval {

namespace NNUE {

namespace {

#if defined(ENABLE_STATIC_EVAL_BIN_TOOL)
struct StaticEvalBinRecord {
  PackedSfen sfen;
  s16 score;
  u16 move;
  u16 game_ply;
  s8 game_result;
  u8 padding;
};

static_assert(sizeof(StaticEvalBinRecord) == 40,
              "PackedSfenValue must be exactly 40 bytes");
static_assert(offsetof(StaticEvalBinRecord, score) == 32,
              "PackedSfenValue score must start at byte 32");
static_assert(sizeof(s16) == 2,
              "PackedSfenValue score must be signed 16-bit");

void WriteCsvField(std::ostream& output, std::string_view value) {
  output << '"';
  for (const char ch : value) {
    if (ch == '"') output << '"';
    output << ch;
  }
  output << '"';
}

// Decode each PackedSfenValue with the engine's existing decoder, perform a
// fresh NNUE evaluation without search, and overwrite only score bytes 32..33.
void MakeStaticEvalBin(std::istream& stream) {
  std::string input_name;
  std::string output_name;
  std::string csv_name;
  std::uint64_t start_record = 0;
  std::uint64_t max_records = 0;
  std::uint64_t csv_records = 0;
  stream >> std::quoted(input_name) >> std::quoted(output_name)
         >> start_record >> max_records >> std::quoted(csv_name) >> csv_records;
  if (input_name.empty() || output_name.empty()) {
    std::cout << "error: make_static_eval_bin requires input and output paths"
              << std::endl;
    return;
  }

  std::ifstream input(input_name, std::ios::binary);
  if (!input) {
    std::cout << "error: failed to open input: " << input_name << std::endl;
    return;
  }
  input.seekg(0, std::ios::end);
  const auto end_position = input.tellg();
  if (end_position < 0
      || static_cast<std::uint64_t>(end_position) % sizeof(StaticEvalBinRecord) != 0) {
    std::cout << "error: input size is not a multiple of 40 bytes" << std::endl;
    return;
  }
  const std::uint64_t total_records =
      static_cast<std::uint64_t>(end_position) / sizeof(StaticEvalBinRecord);
  if (start_record > total_records) {
    std::cout << "error: start record exceeds input record count" << std::endl;
    return;
  }
  const std::uint64_t available = total_records - start_record;
  const std::uint64_t requested =
      max_records == 0 ? available : std::min(max_records, available);
  input.seekg(static_cast<std::streamoff>(
                  start_record * sizeof(StaticEvalBinRecord)),
              std::ios::beg);

  // Never overwrite an existing result. Rename the temporary only after a
  // complete, error-free run.
  {
    std::ifstream existing(output_name, std::ios::binary);
    if (existing) {
      std::cout << "error: output already exists: " << output_name << std::endl;
      return;
    }
  }
  const std::string temporary_name = output_name + ".tmp";
  {
    std::ifstream existing(temporary_name, std::ios::binary);
    if (existing) {
      std::cout << "error: temporary output already exists: "
                << temporary_name << std::endl;
      return;
    }
  }
  std::ofstream output(temporary_name, std::ios::binary | std::ios::trunc);
  if (!output) {
    std::cout << "error: failed to create temporary output: "
              << temporary_name << std::endl;
    return;
  }

  const std::string csv_temporary_name =
      csv_name.empty() ? std::string() : csv_name + ".tmp";
  std::ofstream csv;
  if (!csv_name.empty()) {
    std::ifstream existing(csv_name);
    std::ifstream temporary_existing(csv_temporary_name);
    if (existing || temporary_existing) {
      std::cout << "error: validation CSV or temporary file already exists"
                << std::endl;
      output.close();
      std::remove(temporary_name.c_str());
      return;
    }
    csv.open(csv_temporary_name, std::ios::out | std::ios::trunc);
    if (!csv) {
      std::cout << "error: failed to create validation CSV" << std::endl;
      output.close();
      std::remove(temporary_name.c_str());
      return;
    }
    csv << "record_index,sfen,side_to_move,material_black,material_stm,"
           "original_depth9_score,static_eval_score,depth9_minus_static,"
           "game_ply,game_result,move\n";
  }

  std::uint64_t processed = 0;
  std::uint64_t decode_errors = 0;
  std::uint64_t illegal_positions = 0;
  std::uint64_t non_score_byte_mismatches = 0;
  bool failed = false;
  std::string failure;
  const auto started = std::chrono::steady_clock::now();
  auto last_report = started;
  std::array<char, sizeof(StaticEvalBinRecord)> original_bytes{};
  std::array<char, sizeof(StaticEvalBinRecord)> output_bytes{};

  while (processed < requested
         && input.read(original_bytes.data(), original_bytes.size())) {
    StaticEvalBinRecord record;
    std::memcpy(&record, original_bytes.data(), sizeof(record));

    Position position;
    StateInfo state;
    if (position.set_from_packed_sfen(
            record.sfen, &state, Threads.main(), false, record.game_ply).is_not_ok()) {
      ++decode_errors;
      failed = true;
      failure = "PackedSfen decode failed at record "
              + std::to_string(start_record + processed);
      break;
    }
    if (!position.pos_is_ok()) {
      ++illegal_positions;
      failed = true;
      failure = "illegal/inconsistent position at record "
              + std::to_string(start_record + processed);
      break;
    }

    // Eval::evaluate() performs no qsearch/alpha-beta/move generation. The
    // freshly decoded position has an invalid accumulator, so NNUE refreshes
    // it here. Its result is from the side-to-move perspective, matching the
    // teacher-generation PackedSfenValue convention.
    const s16 static_score = static_cast<s16>(Eval::evaluate(position));

    output_bytes = original_bytes;
    std::memcpy(output_bytes.data() + offsetof(StaticEvalBinRecord, score),
                &static_score, sizeof(static_score));
    for (std::size_t i = 0; i < output_bytes.size(); ++i)
      if ((i < offsetof(StaticEvalBinRecord, score)
           || i >= offsetof(StaticEvalBinRecord, score) + sizeof(static_score))
          && output_bytes[i] != original_bytes[i])
        ++non_score_byte_mismatches;
    output.write(output_bytes.data(), output_bytes.size());
    if (!output) {
      failed = true;
      failure = "failed while writing temporary output";
      break;
    }

    if (csv && processed < csv_records) {
      const int material_black = static_cast<int>(Eval::material(position));
      csv << (start_record + processed) << ',';
      WriteCsvField(csv, position.sfen());
      csv << ',' << static_cast<int>(position.side_to_move())
          << ',' << material_black
          << ',' << (position.side_to_move() == BLACK
                          ? material_black : -material_black)
          << ',' << record.score << ',' << static_score
          << ',' << (static_cast<int>(record.score) - static_cast<int>(static_score))
          << ',' << record.game_ply
          << ',' << static_cast<int>(record.game_result)
          << ',' << Move16(record.move).to_usi_string() << '\n';
    }
    ++processed;

    const auto now = std::chrono::steady_clock::now();
    if (now - last_report >= std::chrono::seconds(5)) {
      last_report = now;
      const double seconds = std::chrono::duration<double>(now - started).count();
      const double rate = seconds > 0.0 ? processed / seconds : 0.0;
      const double eta = rate > 0.0 ? (requested - processed) / rate : 0.0;
      std::cout << "static_eval_bin progress=" << processed << '/' << requested
                << " records_per_sec=" << std::fixed << std::setprecision(1)
                << rate << " eta_sec=" << std::setprecision(0) << eta
                << std::endl;
    }
  }

  if (!failed && processed != requested) {
    failed = true;
    failure = "input ended before requested record count";
  }
  output.flush();
  if (!failed && !output) {
    failed = true;
    failure = "failed while flushing temporary output";
  }
  if (csv) {
    csv.flush();
    if (!failed && !csv) {
      failed = true;
      failure = "failed while flushing validation CSV";
    }
  }
  output.close();
  if (csv) csv.close();

  if (failed || decode_errors != 0 || illegal_positions != 0
      || non_score_byte_mismatches != 0) {
    std::remove(temporary_name.c_str());
    if (!csv_temporary_name.empty()) std::remove(csv_temporary_name.c_str());
    std::cout << "static_eval_bin status=error processed=" << processed
              << " decode_errors=" << decode_errors
              << " illegal_positions=" << illegal_positions
              << " non_score_byte_mismatches=" << non_score_byte_mismatches
              << " message=" << failure << std::endl;
    return;
  }

  if (std::rename(temporary_name.c_str(), output_name.c_str()) != 0) {
    std::cout << "error: failed to rename completed output: "
              << temporary_name << std::endl;
    return;
  }
  if (!csv_temporary_name.empty()
      && std::rename(csv_temporary_name.c_str(), csv_name.c_str()) != 0) {
    std::cout << "error: output completed, validation CSV rename failed: "
              << csv_temporary_name << std::endl;
    return;
  }

  const double seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
  std::cout << "static_eval_bin status=ok input_records=" << total_records
            << " start_record=" << start_record
            << " processed=" << processed
            << " output_bytes=" << processed * sizeof(StaticEvalBinRecord)
            << " decode_errors=0 illegal_positions=0"
            << " non_score_byte_mismatches=0 records_per_sec="
            << std::fixed << std::setprecision(1)
            << (seconds > 0.0 ? processed / seconds : 0.0)
            << " elapsed_sec=" << std::setprecision(3) << seconds
            << std::endl;
}
#endif

// 主に差分計算に関するRawFeaturesのテスト
void TestFeatures(Position& pos) {
  const std::uint64_t num_games = 1000;
  StateInfo si;
  pos.set_hirate(&si,Threads.main());
  const int MAX_PLY = 256; // 256手までテスト

  StateInfo state[MAX_PLY]; // StateInfoを最大手数分だけ
  int ply; // 初期局面からの手数

  PRNG prng(20171128);

  std::uint64_t num_moves = 0;
  std::vector<std::uint64_t> num_updates(kRefreshTriggers.size() + 1);
  std::vector<std::uint64_t> num_resets(kRefreshTriggers.size());
  constexpr IndexType kUnknown = -1;
  std::vector<IndexType> trigger_map(RawFeatures::kDimensions, kUnknown);
  auto make_index_sets = [&](const Position& pos) {
    std::vector<std::vector<std::set<IndexType>>> index_sets(
        kRefreshTriggers.size(), std::vector<std::set<IndexType>>(2));
    for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
      Features::IndexList active_indices[2];
      RawFeatures::AppendActiveIndices(pos, kRefreshTriggers[i],
                                       active_indices);
      for (const auto perspective : COLOR) {
        for (const auto index : active_indices[perspective]) {
          ASSERT(index < RawFeatures::kDimensions);
          ASSERT(index_sets[i][perspective].count(index) == 0);
          ASSERT(trigger_map[index] == kUnknown || trigger_map[index] == i);
          index_sets[i][perspective].insert(index);
          trigger_map[index] = i;
        }
      }
    }
    return index_sets;
  };
  auto update_index_sets = [&](const Position& pos, auto* index_sets) {
    for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
      Features::IndexList removed_indices[2], added_indices[2];
      bool reset[2];
      RawFeatures::AppendChangedIndices(pos, kRefreshTriggers[i],
                                        removed_indices, added_indices, reset);
      for (const auto perspective : COLOR) {
        if (reset[perspective]) {
          (*index_sets)[i][perspective].clear();
          ++num_resets[i];
        } else {
          for (const auto index : removed_indices[perspective]) {
            ASSERT(index < RawFeatures::kDimensions);
            ASSERT((*index_sets)[i][perspective].count(index) == 1);
            ASSERT(trigger_map[index] == kUnknown || trigger_map[index] == i);
            (*index_sets)[i][perspective].erase(index);
            ++num_updates.back();
            ++num_updates[i];
            trigger_map[index] = i;
          }
        }
        for (const auto index : added_indices[perspective]) {
          ASSERT(index < RawFeatures::kDimensions);
          ASSERT((*index_sets)[i][perspective].count(index) == 0);
          ASSERT(trigger_map[index] == kUnknown || trigger_map[index] == i);
          (*index_sets)[i][perspective].insert(index);
          ++num_updates.back();
          ++num_updates[i];
          trigger_map[index] = i;
        }
      }
    }
  };

  std::cout << "feature set: " << RawFeatures::GetName()
            << "[" << RawFeatures::kDimensions << "]" << std::endl;
  std::cout << "start testing with random games";

  for (std::uint64_t i = 0; i < num_games; ++i) {
    auto index_sets = make_index_sets(pos);
    for (ply = 0; ply < MAX_PLY; ++ply) {
      MoveList<LEGAL_ALL> mg(pos); // 全合法手の生成

      // 合法な指し手がなかった == 詰み
      if (mg.size() == 0)
        break;

      // 生成された指し手のなかからランダムに選び、その指し手で局面を進める。
      Move m = mg.begin()[prng.rand(mg.size())];
      pos.do_move(m, state[ply]);

      ++num_moves;
      update_index_sets(pos, &index_sets);
      ASSERT(index_sets == make_index_sets(pos));
    }

    pos.set_hirate(&si,Threads.main());

    // 100回に1回ごとに'.'を出力(進んでいることがわかるように)
    if ((i % 100) == 0)
      std::cout << "." << std::flush;
  }
  std::cout << "passed." << std::endl;
  std::cout << num_games << " games, " << num_moves << " moves, "
            << num_updates.back() << " updates, "
            << (1.0 * num_updates.back() / num_moves)
            << " updates per move" << std::endl;
  std::size_t num_observed_indices = 0;
  for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
    const auto count = std::count(trigger_map.begin(), trigger_map.end(), i);
    num_observed_indices += count;
    std::cout << "TriggerEvent(" << static_cast<int>(kRefreshTriggers[i])
              << "): " << count << " features ("
              << (100.0 * count / RawFeatures::kDimensions) << "%), "
              << num_updates[i] << " updates ("
              << (1.0 * num_updates[i] / num_moves) << " per move), "
              << num_resets[i] << " resets ("
              << (100.0 * num_resets[i] / num_moves) << "%)"
              << std::endl;
  }
  std::cout << "observed " << num_observed_indices << " ("
            << (100.0 * num_observed_indices / RawFeatures::kDimensions)
            << "% of " << RawFeatures::kDimensions
            << ") features" << std::endl;
}

// 評価関数の構造を表す文字列を出力する
void PrintInfo(std::istream& stream) {
  std::cout << "network architecture: " << GetArchitectureString() << std::endl;

  while (true) {
    std::string file_name;
    stream >> file_name;
    if (file_name.empty()) break;

    std::uint32_t hash_value;
    std::string architecture;
    const bool success = [&]() {
      std::ifstream file_stream(file_name, std::ios::binary);
      if (!file_stream) return false;
      if (!ReadHeader(file_stream, &hash_value, &architecture)) return false;
      return true;
    }();

    std::cout << file_name << ": ";
    if (success) {
      if (hash_value == kHashValue) {
        std::cout << "matches with this binary";
        if (architecture != GetArchitectureString()) {
          std::cout << ", but architecture string differs: " << architecture;
        }
        std::cout << std::endl;
      } else {
        std::cout << architecture << std::endl;
      }
    } else {
      std::cout << "failed to read header" << std::endl;
    }
  }
}

}  // namespace

// NNUE評価関数に関するUSI拡張コマンド
void TestCommand(Position& pos, std::istream& stream) {
  std::string sub_command;
  stream >> sub_command;

  if (sub_command == "test_features") {
    TestFeatures(pos);
  } else if (sub_command == "info") {
    PrintInfo(stream);
  }
#if defined(ENABLE_STATIC_EVAL_BIN_TOOL)
  else if (sub_command == "make_static_eval_bin") {
    MakeStaticEvalBin(stream);
  }
#endif
  else {
    std::cout << "usage:" << std::endl;
    std::cout << " test nn test_features" << std::endl;
    std::cout << " test nn info [path/to/" << kFileName << "...]" << std::endl;
#if defined(ENABLE_STATIC_EVAL_BIN_TOOL)
    std::cout << " test nn make_static_eval_bin <input> <output> <start>"
                 " <max-or-0> <csv-or-empty> <csv-records>" << std::endl;
#endif
  }
}

}  // namespace NNUE

}  // namespace Eval

#endif  // defined(ENABLE_TEST_CMD) && defined(EVAL_NNUE)

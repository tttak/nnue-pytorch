#include <iostream>
#include <memory>
#include <string>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <iterator>
#include <future>
#include <mutex>
#include <thread>
#include <type_traits>
#include <deque>
#include <random>
#include <vector>

#include "YaneuraOu/source/config.h"
#include "YaneuraOu/source/usi.h"

#include "lib/nnue_training_data_formats.h"
#include "lib/nnue_training_data_stream.h"
#include "lib/rng.h"
#define NNUE_SIDE_INPUT_KING_SQUARE(pos, color) (pos).king_square(color)
#include "lib/nnue_side_input.h"
#include "lib/nnue_mobility_tactical.h"
#undef NNUE_SIDE_INPUT_KING_SQUARE

#if defined (__x86_64__)
#define EXPORT
#define CDECL
#else
#if defined (_MSC_VER)
#define EXPORT __declspec(dllexport)
#define CDECL __cdecl
#else
#define EXPORT
#define CDECL __attribute__ ((__cdecl__))
#endif
#endif

using namespace binpack;
//using namespace chess;

namespace {
std::atomic<std::uint64_t> g_training_data_seed{5489ULL};
}

namespace SimplePp3Wide {

constexpr int kStates = 4;
constexpr int kSameFilePairs = 9 * 36;
constexpr int kSquarePairs = kSameFilePairs + 8 * 81;
constexpr int kDimensions = kSquarePairs * kStates * kStates;

struct LocalPiece {
    int square;
    int state;
};

inline int mirror_square(const int sq) {
    return (8 - sq / 9) * 9 + sq % 9;
}

inline int compact_square_pair(int a, int b) {
    if (b < a)
        std::swap(a, b);
    if (a == b)
        return -1;
    const int fa = a / 9, ra = a % 9;
    const int fb = b / 9, rb = b % 9;
    if (fa == fb)
        return fa * 36 + rb * (rb - 1) / 2 + ra;
    if (fb == fa + 1)
        return kSameFilePairs + fa * 81 + ra * 9 + rb;
    return -1;
}

inline void append(const Position& pos, const Color perspective,
                   const int batch_index,
                   std::vector<std::int32_t>& indices,
                   std::vector<std::int32_t>& batch_indices) {
    int king = static_cast<int>(pos.king_square(perspective));
    if (perspective == WHITE)
        king = 80 - king;
    const bool mirror = king >= static_cast<int>(SQ_61);

    std::vector<LocalPiece> pieces;
    pieces.reserve(22);
    for (int c = 0; c < COLOR_NB; ++c) {
        const Color owner = static_cast<Color>(c);
        for (const PieceType pt : {PAWN, LANCE}) {
            Bitboard bb = pos.pieces(owner, pt);
            while (bb) {
                int sq = static_cast<int>(bb.pop());
                if (perspective == WHITE)
                    sq = 80 - sq;
                if (mirror)
                    sq = mirror_square(sq);
                const int relative_owner = c ^ static_cast<int>(perspective);
                const int piece_class = pt == LANCE ? 1 : 0;
                pieces.push_back({sq, relative_owner * 2 + piece_class});
            }
        }
    }
    std::sort(pieces.begin(), pieces.end(), [](const LocalPiece& lhs,
                                                const LocalPiece& rhs) {
        return lhs.square < rhs.square;
    });
    for (std::size_t i = 0; i < pieces.size(); ++i)
        for (std::size_t j = i + 1; j < pieces.size(); ++j) {
            const int pair = compact_square_pair(
                pieces[i].square, pieces[j].square);
            if (pair < 0)
                continue;
            indices.push_back(pair * 16 + pieces[i].state * 4
                              + pieces[j].state);
            batch_indices.push_back(batch_index);
        }
}

} // namespace SimplePp3Wide

namespace PairRelationSideInput {

constexpr int kPieceTypeCount = 14;
constexpr int kOwnerDirectionCount = 4;
constexpr int kRelationTypeCount =
    kPieceTypeCount * kPieceTypeCount * kOwnerDirectionCount;

// Compact v1 order is deliberately independent from YaneuraOu's PieceType
// numeric order (where Gold/Bishop/Rook/King have a different placement).
inline int compact_piece_type(const PieceType pt) {
    switch (pt) {
    case PAWN:       return 0;
    case LANCE:      return 1;
    case KNIGHT:     return 2;
    case SILVER:     return 3;
    case GOLD:       return 4;
    case BISHOP:     return 5;
    case ROOK:       return 6;
    case PRO_PAWN:   return 7;
    case PRO_LANCE:  return 8;
    case PRO_KNIGHT: return 9;
    case PRO_SILVER: return 10;
    case HORSE:      return 11;
    case DRAGON:     return 12;
    case KING:       return 13;
    default:         return -1;
    }
}

inline int owner_direction(const Color attacker, const Color target,
                           const Color us) {
    if (attacker == us)
        return target == us ? 0 : 1;
    return target == us ? 3 : 2;
}

inline int relation_index(const Piece attacker, const Piece target,
                          const Color us) {
    const int attacker_type = compact_piece_type(type_of(attacker));
    const int target_type = compact_piece_type(type_of(target));
    const int direction = owner_direction(
        color_of(attacker), color_of(target), us);
    if (attacker_type < 0 || target_type < 0)
        return -1;
    return attacker_type + kPieceTypeCount
        * (target_type + kPieceTypeCount * direction);
}

inline void append(const Position& pos, const int batch_index,
                   std::vector<std::int32_t>& indices,
                   std::vector<std::int32_t>& batch_indices) {
    const Color us = pos.side_to_move();
    const Bitboard occupied = pos.pieces();
    Bitboard attackers = occupied;
    while (attackers) {
        const Square from = attackers.pop();
        const Piece attacker = pos.piece_on(from);
        Bitboard targets = effects_from(attacker, from, occupied) & occupied;
        while (targets) {
            const Square to = targets.pop();
            const int index = relation_index(
                attacker, pos.piece_on(to), us);
            if (index >= 0) {
                indices.push_back(index);
                batch_indices.push_back(batch_index);
            }
        }
    }
}

}  // namespace PairRelationSideInput

static Square orient(Color color, Square sq)
{
    if (color == Color::BLACK)
    {
        return sq;
    }
    else
    {
        // IMPORTANT: for now we use rotate180 instead of rank flip
        //            for compatibility with the stockfish master branch.
        //            Note that this is inconsistent with nodchip/master.
        return Inv(sq);
    }
}

//static Square orient_flip(Color color, Square sq)
//{
//    if (color == Color::BLACK)
//    {
//        return sq;
//    }
//    else
//    {
//        return sq.flippedVertically();
//    }
//}


struct HalfKA_KSDG3 {
    static constexpr int INPUTS = 190998 + 12672;

    static constexpr int MAX_ACTIVE_FEATURES = 40 + 24;
    //static constexpr int MAX_ACTIVE_FEATURES = 39 + 24;


    // ----- KingSafety_DistinguishGolds

    // 壁のPiece値を定義
    static constexpr Piece PIECE_WALL = PIECE_NB;
    static constexpr Piece PIECE_WALL_NB = static_cast<Piece>(PIECE_WALL + 1);

    static Piece Inv(Piece pc) {
        if (pc == NO_PIECE) {
            return NO_PIECE;
        }
        else if (pc == PIECE_WALL) {
            return PIECE_WALL;
        }
        else {
            return make_piece(~color_of(pc), type_of(pc));
        }
    }

    static Effect24::Direct Inv(Effect24::Direct dir) {
        return Effect24::DIRECT_NB - static_cast<Effect24::Direct>(1) - dir;
    }

    static int MakeIndex(Color perspective, Effect24::Direct dir, Piece pc, int effect1, int effect2) {
        if (perspective == WHITE) {
            pc = Inv(pc);
            dir = Inv(dir);
        }

        return ((static_cast<int>(dir)
            * static_cast<int>(PIECE_WALL_NB) + static_cast<int>(pc))
            * 4 + effect1)
            * 4 + effect2;
    }

    static int GetEffectCount(const Position& pos, Square sq, Color perspective) {
        if (sq == SQ_NB) {
            return 0;
        }
        else {
            return std::min(int(pos.board_effect[perspective].effect(sq)), 3);
        }
    }

    // -----

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        int features_unordered[MAX_ACTIVE_FEATURES];
        int features_index = 0;
        auto& pos = *e.pos;

        // ----- KingSafety_DistinguishGolds

        // color側の玉のマス（先手目線）
        SquareWithWall sqww_king = to_sqww(pos.king_square(color));

        // 24近傍をループ
        for (int dir_int = 0; dir_int < int(Effect24::DIRECT_NB); ++dir_int) {
            const auto dir = static_cast<Effect24::Direct>(dir_int);
            SquareWithWall sqww = sqww_king + DirectToDeltaWW(dir);
            int index_caluculated = -1;

            // 盤内の場合
            if (is_ok(sqww)) {
                Square sq = sqww_to_sq(sqww);
                index_caluculated = MakeIndex(color, dir, pos.piece_on(sq)
                        , GetEffectCount(pos, sq,  color)
                        , GetEffectCount(pos, sq, ~color)
                    );
            }

            // 盤外の場合
            else {
                // KSDG3の場合、何もしない
                //index_caluculated = MakeIndex(color, dir, PIECE_WALL, 0, 0);
            }

            if(index_caluculated != -1) {
                features_unordered[features_index] = index_caluculated;
                features_index++;
            }
        }


        // ----- HalfKA
        Eval::BonaPiece* pieces = nullptr;
        if (color == Color::BLACK) {
            pieces = pos.eval_list()->piece_list_fb();
        }
        else {
            pieces = pos.eval_list()->piece_list_fw();
        }
        PieceNumber target = static_cast<PieceNumber>(PIECE_NUMBER_KING + color);
        auto sq_target_k = static_cast<Square>((pieces[target] - Eval::BonaPiece::f_king) % SQ_NB);

        for (PieceNumber i = PIECE_NUMBER_ZERO; i < PIECE_NUMBER_NB; ++i) {
            // 40→39
            if (i == target) {
                continue;
            }

            auto p = pieces[i];
            features_unordered[features_index] = 12672 + static_cast<int>(Eval::fe_end2) * static_cast<int>(sq_target_k) + p;
            features_index++;
        }


        // -----
        //std::sort(features_unordered, features_unordered + MAX_ACTIVE_FEATURES);
        for (int k = 0; k < features_index; ++k)
        {
            values[k] = 1.0f;
            features[k] = features_unordered[k];
        }

        return { features_index, INPUTS };
    }
};

// Experiment 69 variant D:
//   HalfKA_hm1 + merged gold/promoted-minor rows for both HalfKA and KSDG3.
//
// Keep this as an explicit remap of the proven production extractor. This
// guarantees that A (HalfKA_KSDG3) remains bit-for-bit unchanged while D can
// be selected solely by the Python --features name.
struct HalfKA_HM1_NoDG_KSDG3_NoDG {
    static constexpr int KSDG_INPUTS = 12672;
    static constexpr int OLD_BONA = 2358;
    static constexpr int NODG_BONA = 1710;
    static constexpr int INPUTS = KSDG_INPUTS + 45 * NODG_BONA;
    static constexpr int MAX_ACTIVE_FEATURES = HalfKA_KSDG3::MAX_ACTIVE_FEATURES;

    static int mirror_square(int sq) {
        return (8 - sq / 9) * 9 + sq % 9;
    }

    static int map_bona(int p, bool mirror) {
        if (mirror && p >= 90) {
            const int q = p - 90;
            p = 90 + (q / 81) * 81 + mirror_square(q % 81);
        }
        if (p >= 1548 && p < 2196)
            p = 738 + (p - 1548) % 162;
        if (p >= 2196)
            p -= 648;
        return p;
    }

    static int map_index(int index) {
        if (index < KSDG_INPUTS) {
            const int effect2 = index % 4;
            int x = index / 4;
            const int effect1 = x % 4;
            x /= 4;
            int pc = x % 33;
            const int direct = x / 33;
            if (pc >= 9 && pc <= 12)
                pc = 7;
            if (pc >= 25 && pc <= 28)
                pc = 23;
            return ((direct * 33 + pc) * 4 + effect1) * 4 + effect2;
        }

        const int h = index - KSDG_INPUTS;
        int king = h / OLD_BONA;
        int p = h % OLD_BONA;
        const bool mirror = king >= 45;
        if (mirror)
            king = mirror_square(king);
        p = map_bona(p, mirror);
        return KSDG_INPUTS + king * NODG_BONA + p;
    }

    static std::pair<int, int> fill_features_sparse(
        const TrainingDataEntry& e, int* features, float* values, Color color) {
        auto result = HalfKA_KSDG3::fill_features_sparse(
            e, features, values, color);

        // HalfKA_KSDG3 is the legacy Python training contract and excludes the
        // perspective-side king (39 HalfKA rows).  The production C++
        // HalfKA_hm1 extractor includes both kings (40 rows), so append that
        // one missing row in the old layout before applying the hm1/no-DG
        // remap.  Existing D checkpoints are neutral to this correction: the
        // corresponding Main and V rows were never trained and are zero.
        auto& pos = *e.pos;
        Eval::BonaPiece* pieces = color == Color::BLACK
            ? pos.eval_list()->piece_list_fb()
            : pos.eval_list()->piece_list_fw();
        const PieceNumber target =
            static_cast<PieceNumber>(PIECE_NUMBER_KING + color);
        const auto sq_target_k = static_cast<Square>(
            (pieces[target] - Eval::BonaPiece::f_king) % SQ_NB);
        features[result.first] =
            KSDG_INPUTS + OLD_BONA * static_cast<int>(sq_target_k)
            + static_cast<int>(pieces[target]);
        values[result.first] = 1.0f;
        ++result.first;

        for (int i = 0; i < result.first; ++i)
            features[i] = map_index(features[i]);
        return {result.first, INPUTS};
    }
};

// Experiment 85 canonical simple feature set.  This mirrors
// Features::HalfKA_hm2<Friend> while making the merged-gold/no-DG layout
// explicit and independent of the production complex extractor.
struct HalfKA_HM2_NoDG {
    static constexpr int BONA_PLANES = 1629;
    static constexpr int INPUTS = 45 * BONA_PLANES;
    static constexpr int MAX_ACTIVE_FEATURES = PIECE_NUMBER_NB;

    static int mirror_square(int sq) {
        return (8 - sq / 9) * 9 + sq % 9;
    }

    static int map_bona(int p, bool mirror) {
        if (mirror && p >= 90) {
            const int q = p - 90;
            p = 90 + (q / 81) * 81 + mirror_square(q % 81);
        }
        // Merge promoted pawn/lance/knight/silver into the gold planes.
        if (p >= 1548 && p < 2196)
            p = 738 + (p - 1548) % 162;
        // Remove the four distinguished-gold planes.
        if (p >= 2196)
            p -= 648;
        // hm2 shares the friend/enemy king plane.
        if (p >= BONA_PLANES)
            p -= 81;
        return p;
    }

    static std::pair<int, int> fill_features_sparse(
        const TrainingDataEntry& e, int* features, float* values, Color color) {
        const auto& pos = *e.pos;
        Eval::BonaPiece* pieces = color == BLACK
            ? pos.eval_list()->piece_list_fb()
            : pos.eval_list()->piece_list_fw();
        const PieceNumber target = static_cast<PieceNumber>(PIECE_NUMBER_KING + color);
        int king = static_cast<int>((pieces[target] - Eval::BonaPiece::f_king) % SQ_NB);
        const bool mirror = king >= static_cast<int>(SQ_61);
        if (mirror)
            king = mirror_square(king);

        int count = 0;
        for (PieceNumber i = PIECE_NUMBER_ZERO; i < PIECE_NUMBER_NB; ++i) {
            const int p = map_bona(static_cast<int>(pieces[i]), mirror);
            features[count] = king * BONA_PLANES + p;
            values[count] = 1.0f;
            ++count;
        }
        return {count, INPUTS};
    }
};

struct HalfKA_KSDG3_Factorized {
    // RelKA
    static constexpr int NUN_PIECE_KINDS = (Eval::fe_end2 - Eval::fe_hand_end) / 81; // 28
    static constexpr int REL_INPUTS = NUN_PIECE_KINDS * 17 * 17 + Eval::fe_hand_end; // 8182

    // HalfKA_KSDG3_Factorized = HalfKA_KSDG3 + A_GOLDS + HalfRelKAGOLDS + KSDGE00_GOLDS
    static constexpr int INPUTS = (190998 + 12672) + 2358 + REL_INPUTS + 12672;
    static constexpr int MAX_ACTIVE_FEATURES = (40 + 24) + 40 + 40 + 24;

    static int make_relka_index(Square sq_k, int p) {
        if (p < Eval::fe_hand_end) {
            return p;
        }
        constexpr int W = 9 * 2 - 1;
        constexpr int H = 9 * 2 - 1;
        const int piece_index = (p - Eval::fe_hand_end) / SQ_NB;
        const Square sq_p = static_cast<Square>((p - Eval::fe_hand_end) % SQ_NB);
        const int relative_file = file_of(sq_p) - file_of(sq_k) + (W / 2);
        const int relative_rank = rank_of(sq_p) - rank_of(sq_k) + (H / 2);
        return H * W * piece_index + H * relative_file + relative_rank + Eval::fe_hand_end;
    }

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        auto [start_j, offset] = HalfKA_KSDG3::fill_features_sparse(e, features, values, color);

        auto j = start_j;
        auto& pos = *e.pos;
        Eval::BonaPiece* pieces = nullptr;
        if (color == Color::BLACK) {
            pieces = pos.eval_list()->piece_list_fb();
        } else {
            pieces = pos.eval_list()->piece_list_fw();
        }
        PieceNumber target = static_cast<PieceNumber>(PIECE_NUMBER_KING + color);
        auto sq_target_k = static_cast<Square>((pieces[target] - Eval::BonaPiece::f_king) % SQ_NB);


        // ----- A_GOLDS
        // ・KAをAで次元下げ
        // ・「と金～成銀」を金と同一視する
        for (PieceNumber i = PIECE_NUMBER_ZERO; i < PIECE_NUMBER_NB; ++i) {
            // 40→39
            if (i == target) {
                continue;
            }

            auto p = pieces[i];

            if (Eval::fe_old_end <= p && p < Eval::fe_new_end) {
                p = static_cast<Eval::BonaPiece>((p - Eval::fe_old_end) % 162 + Eval::f_gold);
            }

            values[j] = 1.0f;
            features[j] = offset + p;
            ++j;
        }
        offset += 2358;


        // ----- HalfRelKAGOLDS
        // ・KAを「KとA（盤内の駒のみ）の相対位置」で次元下げ
        // ・Aで「と金～成銀」を金と同一視する
        for (PieceNumber i = PIECE_NUMBER_ZERO; i < PIECE_NUMBER_NB; ++i) {
            // 40→39
            if (i == target) {
                continue;
            }

            auto p = pieces[i];

            if (Eval::fe_old_end <= p && p < Eval::fe_new_end) {
                p = static_cast<Eval::BonaPiece>((p - Eval::fe_old_end) % 162 + Eval::f_gold);
            }

            values[j] = 1.0f;
            features[j] = offset + make_relka_index(sq_target_k, p);
            ++j;
        }
        offset += 8182;


        // ----- KSDGE00_GOLDS
        // ・KSDG3を「利き数の相違は無視」で次元下げ
        // ・「と金～成銀」を金と同一視する

        // color側の玉のマス（先手目線）
        SquareWithWall sqww_king = to_sqww(pos.king_square(color));

        // 24近傍をループ
        for (int dir_int = 0; dir_int < int(Effect24::DIRECT_NB); ++dir_int) {
            const auto dir = static_cast<Effect24::Direct>(dir_int);
            SquareWithWall sqww = sqww_king + DirectToDeltaWW(dir);
            int index_caluculated = -1;

            // 盤内の場合
            if (is_ok(sqww)) {
                Square sq = sqww_to_sq(sqww);

                Piece pc = pos.piece_on(sq);
                PieceType pt = type_of(pc);
                Color c = color_of(pc);

                if (pt == PRO_PAWN || pt == PRO_LANCE || pt == PRO_KNIGHT || pt == PRO_SILVER) {
                    pc = make_piece(c, GOLD);
                }

                index_caluculated = HalfKA_KSDG3::MakeIndex(color, dir, pc, 0, 0);
            }

            // 盤外の場合
            else {
                // KSDG3の場合、何もしない
                //index_caluculated = HalfKA_KSDG::MakeIndex(color, dir, HalfKA_KSDG::PIECE_WALL, 0, 0);
            }

            if(index_caluculated != -1) {
                values[j] = 1.0f;
                features[j] = offset + index_caluculated;
                ++j;
            }
        }
        offset += 12672;

        return { j, offset };
    }
};


template <typename T, typename... Ts>
struct FeatureSet
{
    static_assert(sizeof...(Ts) == 0, "Currently only one feature subset supported.");

    static constexpr int INPUTS = T::INPUTS;
    static constexpr int MAX_ACTIVE_FEATURES = T::MAX_ACTIVE_FEATURES;

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        return T::fill_features_sparse(e, features, values, color);
    }
};

struct SparseBatch
{
    static constexpr bool IS_BATCH = true;

    template <typename... Ts>
    SparseBatch(FeatureSet<Ts...>, const std::vector<TrainingDataEntry>& entries,
                const bool generate_pair_relations = false,
                const bool generate_mobility_tactical = false,
                const bool generate_simple_pp3wide = false)
    {
        num_inputs = FeatureSet<Ts...>::INPUTS;
        size = entries.size();
        is_white = new float[size];
        outcome = new float[size];
        score = new float[size];
        ranking_target = new float[size];
        white = new int[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES * 2];
        black = new int[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES * 2];
        white_values = new float[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        black_values = new float[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        layer_stack_indices = new int[size];
        material = new float[size];
        kif_group_id = new int[size];
        ply = new int[size];
        side_input_safe_escape = new std::uint16_t[size];
        side_input_mobility_tactical = (
            generate_mobility_tactical ? new float[size * 8] : nullptr);

        num_active_white_features = 0;
        num_active_black_features = 0;
        max_active_features = FeatureSet<Ts...>::MAX_ACTIVE_FEATURES;

        for (std::size_t i = 0; i < size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES; ++i)
            white[i] = -1;
        for (std::size_t i = 0; i < size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES; ++i)
            black[i] = -1;
        for (std::size_t i = 0; i < size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES; ++i)
            white_values[i] = 0.0f;
        for (std::size_t i = 0; i < size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES; ++i)
            black_values[i] = 0.0f;

        for (int i = 0; i < entries.size(); ++i)
        {
            source_sfens.emplace_back(entries[i].pos->sfen());
            mirror_applied.emplace_back(
                static_cast<std::uint8_t>(entries[i].mirror_applied));
            fill_entry(FeatureSet<Ts...>{}, i, entries[i],
                       generate_pair_relations, generate_mobility_tactical,
                       generate_simple_pp3wide);
        }
    }

    int num_inputs;
    int size;

    float* is_white;
    float* outcome;
    float* score;
    float* ranking_target;
    int num_active_white_features;
    int num_active_black_features;
    int max_active_features;
    int* white;
    int* black;
    float* white_values;
    float* black_values;
    int* layer_stack_indices;
    float* material;
    int* kif_group_id;
    int* ply;
    std::uint16_t* side_input_safe_escape;
    float* side_input_mobility_tactical;
    std::vector<std::int32_t> pair_relation_indices;
    std::vector<std::int32_t> pair_relation_batch_indices;
    std::vector<std::int32_t> pp3wide_white_indices;
    std::vector<std::int32_t> pp3wide_white_batch_indices;
    std::vector<std::int32_t> pp3wide_black_indices;
    std::vector<std::int32_t> pp3wide_black_batch_indices;
    // Diagnostic provenance for Python/C++ parity tests. This member is not
    // part of the stable ctypes prefix and is exposed only through an accessor.
    std::vector<std::string> source_sfens;
    std::vector<std::uint8_t> mirror_applied;

    ~SparseBatch()
    {
        delete[] is_white;
        delete[] outcome;
        delete[] score;
        delete[] ranking_target;
        delete[] white;
        delete[] black;
        delete[] white_values;
        delete[] black_values;
        delete[] layer_stack_indices;
        delete[] material;
        delete[] kif_group_id;
        delete[] ply;
        delete[] side_input_safe_escape;
        delete[] side_input_mobility_tactical;
    }

private:

    template <typename... Ts>
    void fill_entry(FeatureSet<Ts...>, int i, const TrainingDataEntry& e,
                    const bool generate_pair_relations,
                    const bool generate_mobility_tactical,
                    const bool generate_simple_pp3wide)
    {
        is_white[i] = static_cast<float>(e.pos->side_to_move() == Color::BLACK);
        outcome[i] = (e.result + 1.0f) / 2.0f;
        score[i] = e.score;
        ranking_target[i] = e.ranking_target;
        if constexpr ((std::is_same_v<Ts, HalfKA_HM2_NoDG> || ...)) {
            constexpr int friend_band[9] = {0,0,0,3,3,3,6,6,6};
            constexpr int enemy_band[9]  = {0,0,0,1,1,1,2,2,2};
            const Color stm = e.pos->side_to_move();
            const Square fk = e.pos->king_square(stm);
            const Square ek = e.pos->king_square(~stm);
            const int fr = stm == BLACK ? rank_of(fk) : rank_of(Inv(fk));
            const int er = stm == BLACK ? rank_of(Inv(ek)) : rank_of(ek);
            layer_stack_indices[i] = friend_band[fr] + enemy_band[er];
        } else {
            layer_stack_indices[i] = e.pos->stack_index();
        }
        material[i] = e.material;
        kif_group_id[i] = e.kif_group_id;
        ply[i] = e.ply;
        side_input_safe_escape[i] =
            NnueSideInput::safe_escape_mask16(*e.pos);
        if (generate_mobility_tactical) {
            const auto normalized = NnueMobilityTactical::normalize(
                NnueMobilityTactical::raw(*e.pos));
            std::copy(normalized.begin(), normalized.end(),
                      side_input_mobility_tactical + i * 8);
        }
        if (generate_pair_relations)
            PairRelationSideInput::append(
                *e.pos, i, pair_relation_indices,
                pair_relation_batch_indices);
        if (generate_simple_pp3wide) {
            SimplePp3Wide::append(
                *e.pos, BLACK, i, pp3wide_white_indices,
                pp3wide_white_batch_indices);
            SimplePp3Wide::append(
                *e.pos, WHITE, i, pp3wide_black_indices,
                pp3wide_black_batch_indices);
        }
        fill_features(FeatureSet<Ts...>{}, i, e);
    }

    template <typename... Ts>
    void fill_features(FeatureSet<Ts...>, int i, const TrainingDataEntry& e)
    {
        const int offset = i * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES;

        // Color::BLACKとColor::WHITEを逆にした
        num_active_white_features +=
            FeatureSet<Ts...>::fill_features_sparse(e, white + offset, white_values + offset, Color::BLACK)
            .first;
        num_active_black_features +=
            FeatureSet<Ts...>::fill_features_sparse(e, black + offset, black_values + offset, Color::WHITE)
            .first;
    }
};

struct AnyStream
{
    virtual ~AnyStream() = default;
};

template <typename StorageT>
struct Stream : AnyStream
{
    using StorageType = StorageT;

    Stream(int concurrency, const char* filename1, const char* filename2, const char* filename3, float train1_rate, float train2_rate, float skiprate, float mirror, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate, const char* ranking_target3_filename = nullptr) :
        m_stream(training_data::open_sfen_input_file_parallel(concurrency, filename1, filename2, filename3, train1_rate, train2_rate, skiprate, mirror, cyclic, skipPredicate, ranking_target3_filename ? ranking_target3_filename : "", g_training_data_seed.load(std::memory_order_relaxed)))
    {
    }

    virtual StorageT* next() = 0;

protected:
    std::unique_ptr<training_data::BasicSfenInputStream> m_stream;
};

template <typename StorageT>
struct AsyncStream : Stream<StorageT>
{
    using BaseType = Stream<StorageT>;

    AsyncStream(int concurrency, const char* filename1, const char* filename2, const char* filename3, float train1_rate, float train2_rate, float skiprate, float mirror, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
        BaseType(1, filename1, filename2, filename3, train1_rate, train2_rate, skiprate, mirror, cyclic, skipPredicate)
    {
    }

    ~AsyncStream()
    {
        if (m_next.valid())
        {
            delete m_next.get();
        }
    }

protected:
    std::future<StorageT*> m_next;
};

template <typename FeatureSetT, typename StorageT>
struct FeaturedBatchStream : Stream<StorageT>
{
    static_assert(StorageT::IS_BATCH);

    using FeatureSet = FeatureSetT;
    using BaseType = Stream<StorageT>;

    static constexpr int num_feature_threads_per_reading_thread = 2;

    FeaturedBatchStream(int concurrency, const char* filename1, const char* filename2, const char* filename3, float train1_rate, float train2_rate, float skiprate, float mirror, int batch_size, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate, const char* ranking_target3_filename = nullptr, const bool generate_pair_relations = false, const bool generate_mobility_tactical = false, const bool generate_simple_pp3wide = false) :
        BaseType(
            std::max(
                1,
                concurrency / num_feature_threads_per_reading_thread
            ),
            filename1,
            filename2,
            filename3,
            train1_rate,
            train2_rate,
            skiprate,
            mirror,
            cyclic,
            skipPredicate,
            ranking_target3_filename
        ),
        m_concurrency(concurrency),
        m_batch_size(batch_size),
        m_generate_pair_relations(generate_pair_relations),
        m_generate_mobility_tactical(generate_mobility_tactical),
        m_generate_simple_pp3wide(generate_simple_pp3wide)
    {
        m_stop_flag.store(false);

        auto worker = [this]()
        {
            std::vector<TrainingDataEntry> entries;
            entries.reserve(m_batch_size);

            while (!m_stop_flag.load())
            {
                entries.clear();

                {
                    std::unique_lock lock(m_stream_mutex);
                    BaseType::m_stream->fill(entries, m_batch_size);
                    if (entries.empty())
                    {
                        break;
                    }
                }

                auto batch = new StorageT(
                    FeatureSet{}, entries, m_generate_pair_relations,
                    m_generate_mobility_tactical,
                    m_generate_simple_pp3wide);

                {
                    std::unique_lock lock(m_batch_mutex);
                    m_batches_not_full.wait(lock, [this]() { return m_batches.size() < m_concurrency + 1 || m_stop_flag.load(); });

                    m_batches.emplace_back(batch);

                    lock.unlock();
                    m_batches_any.notify_one();
                }

            }
            m_num_workers.fetch_sub(1);
            m_batches_any.notify_one();
        };

        const int num_feature_threads = std::max(
            1,
            concurrency - std::max(1, concurrency / num_feature_threads_per_reading_thread)
        );

        for (int i = 0; i < num_feature_threads; ++i)
        {
            m_workers.emplace_back(worker);

            // This cannot be done in the thread worker. We need
            // to have a guarantee that this is incremented, but if
            // we did it in the worker there's no guarantee
            // that it executed.
            m_num_workers.fetch_add(1);
        }
    }

    StorageT* next() override
    {
        std::unique_lock lock(m_batch_mutex);
        m_batches_any.wait(lock, [this]() { return !m_batches.empty() || m_num_workers.load() == 0; });

        if (!m_batches.empty())
        {
            auto batch = m_batches.front();
            m_batches.pop_front();

            lock.unlock();
            m_batches_not_full.notify_one();

            return batch;
        }
        return nullptr;
    }

    ~FeaturedBatchStream()
    {
        m_stop_flag.store(true);
        m_batches_not_full.notify_all();

        for (auto& worker : m_workers)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }

        for (auto& batch : m_batches)
        {
            delete batch;
        }
    }

private:
    int m_batch_size;
    int m_concurrency;
    bool m_generate_pair_relations;
    bool m_generate_mobility_tactical;
    bool m_generate_simple_pp3wide;
    std::deque<StorageT*> m_batches;
    std::mutex m_batch_mutex;
    std::mutex m_stream_mutex;
    std::condition_variable m_batches_not_full;
    std::condition_variable m_batches_any;
    std::atomic_bool m_stop_flag;
    std::atomic_int m_num_workers;

    std::vector<std::thread> m_workers;
};

static bool initialized = false;

static void EnsureInitialize()
{
    if (initialized) {
        return;
    }
    initialized = true;

    USI::init(Options);
    Bitboards::init();
    Position::init();
    Search::init();

    Threads.set(1);

    Eval::init();

    is_ready();
}

extern "C" {

    EXPORT void CDECL set_training_data_seed(const std::uint64_t seed)
    {
        // Set before creating the streams. Stream instances copy this value
        // into independent shuffle and mirror RNGs.
        g_training_data_seed.store(seed, std::memory_order_relaxed);
    }

    EXPORT std::uint64_t CDECL get_training_data_seed()
    {
        return g_training_data_seed.load(std::memory_order_relaxed);
    }

    // Diagnostic/reference ABI for Pair Relation Side Input v1.  This path
    // intentionally does not build NNUE sparse FT features, so compact
    // handcrafted SFENs containing only the pieces under test are valid.
    EXPORT std::size_t CDECL get_pair_relation_indices_from_sfen(
        const char* sfen, std::int32_t* output, std::size_t capacity)
    {
        EnsureInitialize();
        Position pos;
        StateInfo state;
        pos.set(sfen, &state, Threads.main());
        std::vector<std::int32_t> indices;
        std::vector<std::int32_t> batch_indices;
        PairRelationSideInput::append(pos, 0, indices, batch_indices);
        const std::size_t copied = std::min(indices.size(), capacity);
        std::copy_n(indices.begin(), copied, output);
        return indices.size();
    }

    EXPORT SparseBatch* get_sparse_batch_from_fens(
        const char* feature_set_c,
        int num_fens,
        const char* const* fens,
        int* scores,
        int* plies,
        int* results
    )
    {
        EnsureInitialize();

        std::vector<TrainingDataEntry> entries;
        entries.reserve(num_fens);
        for (int i = 0; i < num_fens; ++i)
        {
            auto& e = entries.emplace_back();
            e.pos->set(fens[i], &e.stateInfo, Threads.main());
            //movegen::forEachLegalMove(e.pos, [&](Move m){e.move = m;});
            e.move = MOVE_NONE;
            e.score = scores[i];
            e.ply = plies[i];
            e.result = results[i];
        }

        std::string_view feature_set(feature_set_c);

        if (feature_set == "HalfKA_KSDG3")
        {
            return new SparseBatch(FeatureSet<HalfKA_KSDG3>{}, entries);
        }
        else if (feature_set == "HalfKA_KSDG3^")
        {
            return new SparseBatch(FeatureSet<HalfKA_KSDG3_Factorized>{}, entries);
        }
        else if (feature_set == "HalfKA_HM1_NoDG_KSDG3_NoDG")
        {
            return new SparseBatch(
                FeatureSet<HalfKA_HM1_NoDG_KSDG3_NoDG>{}, entries);
        }
        else if (feature_set == "HalfKA_HM2_NoDG")
            return new SparseBatch(FeatureSet<HalfKA_HM2_NoDG>{}, entries);

        fprintf(stderr, "Unknown feature_set %s\n", feature_set_c);
        return nullptr;
    }

    EXPORT SparseBatch* get_sparse_batch_from_fens_pair_relation(
        const char* feature_set_c, int num_fens, const char* const* fens,
        int* scores, int* plies, int* results)
    {
        EnsureInitialize();
        std::vector<TrainingDataEntry> entries;
        entries.reserve(num_fens);
        for (int i = 0; i < num_fens; ++i) {
            auto& e = entries.emplace_back();
            e.pos->set(fens[i], &e.stateInfo, Threads.main());
            e.move = MOVE_NONE;
            e.score = scores[i];
            e.ply = plies[i];
            e.result = results[i];
        }
        std::string_view feature_set(feature_set_c);
        if (feature_set == "HalfKA_KSDG3")
            return new SparseBatch(FeatureSet<HalfKA_KSDG3>{}, entries, true);
        if (feature_set == "HalfKA_KSDG3^")
            return new SparseBatch(
                FeatureSet<HalfKA_KSDG3_Factorized>{}, entries, true);
        if (feature_set == "HalfKA_HM1_NoDG_KSDG3_NoDG")
            return new SparseBatch(
                FeatureSet<HalfKA_HM1_NoDG_KSDG3_NoDG>{}, entries, true);
        if (feature_set == "HalfKA_HM2_NoDG")
            return new SparseBatch(FeatureSet<HalfKA_HM2_NoDG>{}, entries, true);
        fprintf(stderr, "Unknown feature_set %s\n", feature_set_c);
        return nullptr;
    }

    EXPORT SparseBatch* get_sparse_batch_from_fens_pp3wide(
        const char* feature_set_c, int num_fens, const char* const* fens,
        int* scores, int* plies, int* results)
    {
        EnsureInitialize();
        std::vector<TrainingDataEntry> entries;
        entries.reserve(num_fens);
        for (int i = 0; i < num_fens; ++i) {
            auto& e = entries.emplace_back();
            e.pos->set(fens[i], &e.stateInfo, Threads.main());
            e.move = MOVE_NONE;
            e.score = scores[i];
            e.ply = plies[i];
            e.result = results[i];
        }
        if (std::string_view(feature_set_c) == "HalfKA_HM2_NoDG")
            return new SparseBatch(
                FeatureSet<HalfKA_HM2_NoDG>{}, entries, false, false, true);
        fprintf(stderr, "PP3Wide requires HalfKA_HM2_NoDG\n");
        return nullptr;
    }

    EXPORT Stream<SparseBatch>* CDECL create_sparse_batch_stream(const char* feature_set_c, int concurrency, const char* filename1, const char* filename2, const char* filename3, float train1_rate, float train2_rate, float skiprate, float mirror, int batch_size, int cyclic, int filtered, int random_fen_skipping)
    {
        EnsureInitialize();

        std::function<bool(const TrainingDataEntry&)> skipPredicate = nullptr;
        if (filtered || random_fen_skipping)
        {
            skipPredicate = [
                random_fen_skipping,
                    prob = double(random_fen_skipping) / (random_fen_skipping + 1),
                    filtered
            ](const TrainingDataEntry& e){

                    auto do_skip = [&]() {
                        std::bernoulli_distribution distrib(prob);
                        auto& prng = rng::get_thread_local_rng();
                        return distrib(prng);
                    };

                    auto do_filter = [&]() {
                        return (e.isCapturingMove() || e.isInCheck());
                    };

                    static thread_local std::mt19937 gen(std::random_device{}());
                    return (random_fen_skipping && do_skip()) || (filtered && do_filter());
                };
        }

        std::string_view feature_set(feature_set_c);

        if (feature_set == "HalfKA_KSDG3")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKA_KSDG3>, SparseBatch>(concurrency, filename1, filename2, filename3, train1_rate, train2_rate, skiprate, mirror, batch_size, cyclic, skipPredicate);
        }
        else if (feature_set == "HalfKA_KSDG3^")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKA_KSDG3_Factorized>, SparseBatch>(concurrency, filename1, filename2, filename3, train1_rate, train2_rate, skiprate, mirror, batch_size, cyclic, skipPredicate);
        }
        else if (feature_set == "HalfKA_HM1_NoDG_KSDG3_NoDG")
        {
            return new FeaturedBatchStream<
                FeatureSet<HalfKA_HM1_NoDG_KSDG3_NoDG>, SparseBatch>(
                    concurrency, filename1, filename2, filename3, train1_rate,
                    train2_rate, skiprate, mirror, batch_size, cyclic,
                    skipPredicate);
        }
        else if (feature_set == "HalfKA_HM2_NoDG")
            return new FeaturedBatchStream<FeatureSet<HalfKA_HM2_NoDG>, SparseBatch>(
                concurrency, filename1, filename2, filename3, train1_rate,
                train2_rate, skiprate, mirror, batch_size, cyclic, skipPredicate);

        fprintf(stderr, "Unknown feature_set %s\n", feature_set_c);
        return nullptr;
    }

    EXPORT Stream<SparseBatch>* CDECL create_sparse_batch_stream_with_ranking_target3(
        const char* feature_set_c, int concurrency, const char* filename1,
        const char* filename2, const char* filename3,
        const char* ranking_target3_filename, float train1_rate,
        float train2_rate, float skiprate, float mirror, int batch_size,
        int cyclic, int filtered, int random_fen_skipping)
    {
        EnsureInitialize();
        if (filtered || random_fen_skipping) {
            fprintf(stderr,
                "ranking-target3 stream requires filtering and random skipping disabled\n");
            return nullptr;
        }

        std::string_view feature_set(feature_set_c);
        if (feature_set == "HalfKA_KSDG3")
            return new FeaturedBatchStream<FeatureSet<HalfKA_KSDG3>, SparseBatch>(
                concurrency, filename1, filename2, filename3, train1_rate,
                train2_rate, skiprate, mirror, batch_size, cyclic, nullptr,
                ranking_target3_filename);
        if (feature_set == "HalfKA_KSDG3^")
            return new FeaturedBatchStream<FeatureSet<HalfKA_KSDG3_Factorized>, SparseBatch>(
                concurrency, filename1, filename2, filename3, train1_rate,
                train2_rate, skiprate, mirror, batch_size, cyclic, nullptr,
                ranking_target3_filename);
        if (feature_set == "HalfKA_HM1_NoDG_KSDG3_NoDG")
            return new FeaturedBatchStream<
                FeatureSet<HalfKA_HM1_NoDG_KSDG3_NoDG>, SparseBatch>(
                concurrency, filename1, filename2, filename3, train1_rate,
                train2_rate, skiprate, mirror, batch_size, cyclic, nullptr,
                ranking_target3_filename);
        if (feature_set == "HalfKA_HM2_NoDG")
            return new FeaturedBatchStream<FeatureSet<HalfKA_HM2_NoDG>, SparseBatch>(
                concurrency, filename1, filename2, filename3, train1_rate,
                train2_rate, skiprate, mirror, batch_size, cyclic, nullptr,
                ranking_target3_filename);

        fprintf(stderr, "Unknown feature_set %s\n", feature_set_c);
        return nullptr;
    }

    // Pair-relation v1 stream ABI.  Separate entry points keep every existing
    // caller and the default OFF path byte-for-byte compatible with the old
    // argument list and avoid relation generation when the branch is disabled.
    EXPORT Stream<SparseBatch>* CDECL create_sparse_batch_stream_pair_relation(
        const char* feature_set_c, int concurrency, const char* filename1,
        const char* filename2, const char* filename3, float train1_rate,
        float train2_rate, float skiprate, float mirror, int batch_size,
        int cyclic, int filtered, int random_fen_skipping)
    {
        EnsureInitialize();
        std::function<bool(const TrainingDataEntry&)> skipPredicate = nullptr;
        if (filtered || random_fen_skipping) {
            skipPredicate = [
                random_fen_skipping,
                prob = double(random_fen_skipping) / (random_fen_skipping + 1),
                filtered
            ](const TrainingDataEntry& e) {
                auto do_skip = [&]() {
                    std::bernoulli_distribution distrib(prob);
                    auto& prng = rng::get_thread_local_rng();
                    return distrib(prng);
                };
                auto do_filter = [&]() {
                    return e.isCapturingMove() || e.isInCheck();
                };
                return (random_fen_skipping && do_skip())
                    || (filtered && do_filter());
            };
        }

        std::string_view feature_set(feature_set_c);
        if (feature_set == "HalfKA_KSDG3")
            return new FeaturedBatchStream<FeatureSet<HalfKA_KSDG3>, SparseBatch>(
                concurrency, filename1, filename2, filename3, train1_rate,
                train2_rate, skiprate, mirror, batch_size, cyclic,
                skipPredicate, nullptr, true);
        if (feature_set == "HalfKA_KSDG3^")
            return new FeaturedBatchStream<FeatureSet<HalfKA_KSDG3_Factorized>, SparseBatch>(
                concurrency, filename1, filename2, filename3, train1_rate,
                train2_rate, skiprate, mirror, batch_size, cyclic,
                skipPredicate, nullptr, true);
        if (feature_set == "HalfKA_HM1_NoDG_KSDG3_NoDG")
            return new FeaturedBatchStream<
                FeatureSet<HalfKA_HM1_NoDG_KSDG3_NoDG>, SparseBatch>(
                    concurrency, filename1, filename2, filename3, train1_rate,
                    train2_rate, skiprate, mirror, batch_size, cyclic,
                    skipPredicate, nullptr, true);
        fprintf(stderr, "Unknown feature_set %s\n", feature_set_c);
        return nullptr;
    }

    EXPORT Stream<SparseBatch>* CDECL
    create_sparse_batch_stream_with_ranking_target3_pair_relation(
        const char* feature_set_c, int concurrency, const char* filename1,
        const char* filename2, const char* filename3,
        const char* ranking_target3_filename, float train1_rate,
        float train2_rate, float skiprate, float mirror, int batch_size,
        int cyclic, int filtered, int random_fen_skipping)
    {
        EnsureInitialize();
        if (filtered || random_fen_skipping) {
            fprintf(stderr,
                "ranking-target3 stream requires filtering and random skipping disabled\n");
            return nullptr;
        }
        std::string_view feature_set(feature_set_c);
        if (feature_set == "HalfKA_KSDG3")
            return new FeaturedBatchStream<FeatureSet<HalfKA_KSDG3>, SparseBatch>(
                concurrency, filename1, filename2, filename3, train1_rate,
                train2_rate, skiprate, mirror, batch_size, cyclic, nullptr,
                ranking_target3_filename, true);
        if (feature_set == "HalfKA_KSDG3^")
            return new FeaturedBatchStream<FeatureSet<HalfKA_KSDG3_Factorized>, SparseBatch>(
                concurrency, filename1, filename2, filename3, train1_rate,
                train2_rate, skiprate, mirror, batch_size, cyclic, nullptr,
                ranking_target3_filename, true);
        if (feature_set == "HalfKA_HM1_NoDG_KSDG3_NoDG")
            return new FeaturedBatchStream<
                FeatureSet<HalfKA_HM1_NoDG_KSDG3_NoDG>, SparseBatch>(
                    concurrency, filename1, filename2, filename3, train1_rate,
                    train2_rate, skiprate, mirror, batch_size, cyclic, nullptr,
                    ranking_target3_filename, true);
        fprintf(stderr, "Unknown feature_set %s\n", feature_set_c);
        return nullptr;
    }

    // Experiment 84 formal branch. Separate factories preserve the default
    // stream ABI and avoid mobility/tactical extraction when side input is OFF.
    EXPORT Stream<SparseBatch>* CDECL
    create_sparse_batch_stream_mobility_tactical(
        const char* feature_set_c, int concurrency, const char* filename1,
        const char* filename2, const char* filename3, float train1_rate,
        float train2_rate, float skiprate, float mirror, int batch_size,
        int cyclic, int filtered, int random_fen_skipping)
    {
        EnsureInitialize();
        std::function<bool(const TrainingDataEntry&)> skipPredicate = nullptr;
        if (filtered || random_fen_skipping) {
            skipPredicate = [
                random_fen_skipping,
                prob = double(random_fen_skipping) / (random_fen_skipping + 1),
                filtered](const TrainingDataEntry& e) {
                auto do_skip = [&]() {
                    std::bernoulli_distribution distrib(prob);
                    auto& prng = rng::get_thread_local_rng();
                    return distrib(prng);
                };
                return (random_fen_skipping && do_skip())
                    || (filtered && (e.isCapturingMove() || e.isInCheck()));
            };
        }
        std::string_view feature_set(feature_set_c);
        if (feature_set == "HalfKA_KSDG3")
            return new FeaturedBatchStream<FeatureSet<HalfKA_KSDG3>, SparseBatch>(
                concurrency, filename1, filename2, filename3, train1_rate,
                train2_rate, skiprate, mirror, batch_size, cyclic,
                skipPredicate, nullptr, false, true);
        if (feature_set == "HalfKA_KSDG3^")
            return new FeaturedBatchStream<FeatureSet<HalfKA_KSDG3_Factorized>, SparseBatch>(
                concurrency, filename1, filename2, filename3, train1_rate,
                train2_rate, skiprate, mirror, batch_size, cyclic,
                skipPredicate, nullptr, false, true);
        if (feature_set == "HalfKA_HM1_NoDG_KSDG3_NoDG")
            return new FeaturedBatchStream<
                FeatureSet<HalfKA_HM1_NoDG_KSDG3_NoDG>, SparseBatch>(
                    concurrency, filename1, filename2, filename3, train1_rate,
                    train2_rate, skiprate, mirror, batch_size, cyclic,
                    skipPredicate, nullptr, false, true);
        fprintf(stderr, "Unknown feature_set %s\n", feature_set_c);
        return nullptr;
    }

    EXPORT Stream<SparseBatch>* CDECL
    create_sparse_batch_stream_with_ranking_target3_mobility_tactical(
        const char* feature_set_c, int concurrency, const char* filename1,
        const char* filename2, const char* filename3,
        const char* ranking_target3_filename, float train1_rate,
        float train2_rate, float skiprate, float mirror, int batch_size,
        int cyclic, int filtered, int random_fen_skipping)
    {
        EnsureInitialize();
        if (filtered || random_fen_skipping) {
            fprintf(stderr,
                "ranking-target3 stream requires filtering and random skipping disabled\n");
            return nullptr;
        }
        std::string_view feature_set(feature_set_c);
        if (feature_set == "HalfKA_KSDG3")
            return new FeaturedBatchStream<FeatureSet<HalfKA_KSDG3>, SparseBatch>(
                concurrency, filename1, filename2, filename3, train1_rate,
                train2_rate, skiprate, mirror, batch_size, cyclic, nullptr,
                ranking_target3_filename, false, true);
        if (feature_set == "HalfKA_KSDG3^")
            return new FeaturedBatchStream<FeatureSet<HalfKA_KSDG3_Factorized>, SparseBatch>(
                concurrency, filename1, filename2, filename3, train1_rate,
                train2_rate, skiprate, mirror, batch_size, cyclic, nullptr,
                ranking_target3_filename, false, true);
        if (feature_set == "HalfKA_HM1_NoDG_KSDG3_NoDG")
            return new FeaturedBatchStream<
                FeatureSet<HalfKA_HM1_NoDG_KSDG3_NoDG>, SparseBatch>(
                    concurrency, filename1, filename2, filename3, train1_rate,
                    train2_rate, skiprate, mirror, batch_size, cyclic, nullptr,
                    ranking_target3_filename, false, true);
        fprintf(stderr, "Unknown feature_set %s\n", feature_set_c);
        return nullptr;
    }

    // Experiment 120.  Keep the historical SparseBatch prefix and ordinary
    // stream entry points untouched; PP rows are exposed by accessor vectors.
    EXPORT Stream<SparseBatch>* CDECL create_sparse_batch_stream_pp3wide(
        const char* feature_set_c, int concurrency, const char* filename1,
        const char* filename2, const char* filename3, float train1_rate,
        float train2_rate, float skiprate, float mirror, int batch_size,
        int cyclic, int filtered, int random_fen_skipping)
    {
        EnsureInitialize();
        std::function<bool(const TrainingDataEntry&)> skipPredicate = nullptr;
        if (filtered || random_fen_skipping) {
            skipPredicate = [
                random_fen_skipping,
                prob = double(random_fen_skipping) / (random_fen_skipping + 1),
                filtered](const TrainingDataEntry& e) {
                auto do_skip = [&]() {
                    std::bernoulli_distribution distrib(prob);
                    auto& prng = rng::get_thread_local_rng();
                    return distrib(prng);
                };
                return (random_fen_skipping && do_skip())
                    || (filtered && (e.isCapturingMove() || e.isInCheck()));
            };
        }
        if (std::string_view(feature_set_c) == "HalfKA_HM2_NoDG")
            return new FeaturedBatchStream<FeatureSet<HalfKA_HM2_NoDG>, SparseBatch>(
                concurrency, filename1, filename2, filename3, train1_rate,
                train2_rate, skiprate, mirror, batch_size, cyclic,
                skipPredicate, nullptr, false, false, true);
        fprintf(stderr, "PP3Wide requires HalfKA_HM2_NoDG\n");
        return nullptr;
    }

    EXPORT Stream<SparseBatch>* CDECL
    create_sparse_batch_stream_with_ranking_target3_pp3wide(
        const char* feature_set_c, int concurrency, const char* filename1,
        const char* filename2, const char* filename3,
        const char* ranking_target3_filename, float train1_rate,
        float train2_rate, float skiprate, float mirror, int batch_size,
        int cyclic, int filtered, int random_fen_skipping)
    {
        EnsureInitialize();
        if (filtered || random_fen_skipping) {
            fprintf(stderr,
                "ranking-target3 stream requires filtering and random skipping disabled\n");
            return nullptr;
        }
        if (std::string_view(feature_set_c) == "HalfKA_HM2_NoDG")
            return new FeaturedBatchStream<FeatureSet<HalfKA_HM2_NoDG>, SparseBatch>(
                concurrency, filename1, filename2, filename3, train1_rate,
                train2_rate, skiprate, mirror, batch_size, cyclic, nullptr,
                ranking_target3_filename, false, false, true);
        fprintf(stderr, "PP3Wide requires HalfKA_HM2_NoDG\n");
        return nullptr;
    }

    EXPORT void CDECL destroy_sparse_batch_stream(Stream<SparseBatch>* stream)
    {
        delete stream;
    }

    EXPORT SparseBatch* CDECL fetch_next_sparse_batch(Stream<SparseBatch>* stream)
    {
        return stream->next();
    }

    EXPORT void CDECL destroy_sparse_batch(SparseBatch* e)
    {
        delete e;
    }

    // Accessor keeps the historical SparseBatch C ABI unchanged. Python only
    // calls it when an optional side input is requested.
    EXPORT const std::uint16_t* CDECL get_sparse_batch_safe_escape(
        const SparseBatch* batch)
    {
        return batch ? batch->side_input_safe_escape : nullptr;
    }

    EXPORT const float* CDECL get_sparse_batch_mobility_tactical(
        const SparseBatch* batch)
    {
        return batch ? batch->side_input_mobility_tactical : nullptr;
    }

    EXPORT std::size_t CDECL get_sparse_batch_pair_relation_count(
        const SparseBatch* batch)
    {
        return batch ? batch->pair_relation_indices.size() : 0;
    }

    EXPORT const std::int32_t* CDECL get_sparse_batch_pair_relation_indices(
        const SparseBatch* batch)
    {
        return batch && !batch->pair_relation_indices.empty()
            ? batch->pair_relation_indices.data() : nullptr;
    }

    EXPORT const std::int32_t* CDECL
    get_sparse_batch_pair_relation_batch_indices(const SparseBatch* batch)
    {
        return batch && !batch->pair_relation_batch_indices.empty()
            ? batch->pair_relation_batch_indices.data() : nullptr;
    }

    EXPORT std::size_t CDECL get_sparse_batch_pp3wide_white_count(
        const SparseBatch* batch)
    {
        return batch ? batch->pp3wide_white_indices.size() : 0;
    }

    EXPORT const std::int32_t* CDECL get_sparse_batch_pp3wide_white_indices(
        const SparseBatch* batch)
    {
        return batch && !batch->pp3wide_white_indices.empty()
            ? batch->pp3wide_white_indices.data() : nullptr;
    }

    EXPORT const std::int32_t* CDECL
    get_sparse_batch_pp3wide_white_batch_indices(const SparseBatch* batch)
    {
        return batch && !batch->pp3wide_white_batch_indices.empty()
            ? batch->pp3wide_white_batch_indices.data() : nullptr;
    }

    EXPORT std::size_t CDECL get_sparse_batch_pp3wide_black_count(
        const SparseBatch* batch)
    {
        return batch ? batch->pp3wide_black_indices.size() : 0;
    }

    EXPORT const std::int32_t* CDECL get_sparse_batch_pp3wide_black_indices(
        const SparseBatch* batch)
    {
        return batch && !batch->pp3wide_black_indices.empty()
            ? batch->pp3wide_black_indices.data() : nullptr;
    }

    EXPORT const std::int32_t* CDECL
    get_sparse_batch_pp3wide_black_batch_indices(const SparseBatch* batch)
    {
        return batch && !batch->pp3wide_black_batch_indices.empty()
            ? batch->pp3wide_black_batch_indices.data() : nullptr;
    }

    EXPORT const char* CDECL get_sparse_batch_source_sfen(
        const SparseBatch* batch, const std::size_t index)
    {
        return batch && index < batch->source_sfens.size()
            ? batch->source_sfens[index].c_str() : nullptr;
    }

    EXPORT const std::uint8_t* CDECL get_sparse_batch_mirror_applied(
        const SparseBatch* batch)
    {
        return batch && !batch->mirror_applied.empty()
            ? batch->mirror_applied.data() : nullptr;
    }

}

/* benches */ //*
#include <chrono>

int main()
{
    auto stream = create_sparse_batch_stream("HalfKP^", 4, R"(C:\shogi\training_data\suisho5.shuffled.qsearch\shuffled.bin)", R"(C:\shogi\training_data\suisho5.shuffled.qsearch\shuffled.bin)", R"(C:\shogi\training_data\suisho5.shuffled.qsearch\shuffled.bin)", 0.33, 0.33, 1.5, 0.1, 8192, true, false, 0);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i)
    {
        if (i % 100 == 0) std::cout << i << '\n';
        destroy_sparse_batch(stream->next());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << (t1 - t0).count() / 1e9 << "s\n";
}
//*/

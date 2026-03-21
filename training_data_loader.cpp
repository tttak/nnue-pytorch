#include <iostream>
#include <memory>
#include <string>
#include <algorithm>
#include <iterator>
#include <future>
#include <mutex>
#include <thread>
#include <deque>
#include <random>

#include "YaneuraOu/source/config.h"
#include "YaneuraOu/source/usi.h"

#include "lib/nnue_training_data_formats.h"
#include "lib/nnue_training_data_stream.h"
#include "lib/rng.h"

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
        for (Effect24::Direct dir : Effect24::Direct()) {
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
        for (Effect24::Direct dir : Effect24::Direct()) {
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
    SparseBatch(FeatureSet<Ts...>, const std::vector<TrainingDataEntry>& entries)
    {
        num_inputs = FeatureSet<Ts...>::INPUTS;
        size = entries.size();
        is_white = new float[size];
        outcome = new float[size];
        score = new float[size];
        white = new int[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES * 2];
        black = new int[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES * 2];
        white_values = new float[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        black_values = new float[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        layer_stack_indices = new int[size];
        material = new float[size];

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
            fill_entry(FeatureSet<Ts...>{}, i, entries[i]);
        }
    }

    int num_inputs;
    int size;

    float* is_white;
    float* outcome;
    float* score;
    int num_active_white_features;
    int num_active_black_features;
    int max_active_features;
    int* white;
    int* black;
    float* white_values;
    float* black_values;
    int* layer_stack_indices;
    float* material;

    ~SparseBatch()
    {
        delete[] is_white;
        delete[] outcome;
        delete[] score;
        delete[] white;
        delete[] black;
        delete[] white_values;
        delete[] black_values;
        delete[] layer_stack_indices;
        delete[] material;
    }

private:

    template <typename... Ts>
    void fill_entry(FeatureSet<Ts...>, int i, const TrainingDataEntry& e)
    {
        is_white[i] = static_cast<float>(e.pos->side_to_move() == Color::BLACK);
        outcome[i] = (e.result + 1.0f) / 2.0f;
        score[i] = e.score;
        layer_stack_indices[i] = e.pos->stack_index();
        material[i] = e.material;
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

    Stream(int concurrency, const char* filename1, const char* filename2, const char* filename3, float train1_rate, float train2_rate, float skiprate, float mirror, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
        m_stream(training_data::open_sfen_input_file_parallel(concurrency, filename1, filename2, filename3, train1_rate, train2_rate, skiprate, mirror, cyclic, skipPredicate))
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

    FeaturedBatchStream(int concurrency, const char* filename1, const char* filename2, const char* filename3, float train1_rate, float train2_rate, float skiprate, float mirror, int batch_size, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
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
            skipPredicate
        ),
        m_concurrency(concurrency),
        m_batch_size(batch_size)
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

                auto batch = new StorageT(FeatureSet{}, entries);

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

        fprintf(stderr, "Unknown feature_set %s\n", feature_set_c);
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

        fprintf(stderr, "Unknown feature_set %s\n", feature_set_c);
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

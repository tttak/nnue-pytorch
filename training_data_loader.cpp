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

#include "lib/nnue_training_data_formats.h"
#include "lib/nnue_training_data_stream.h"
#include "lib/rng.h"

#include "lib/YaneuraOu/usi.h"

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

struct HalfKP {
    static constexpr int INPUTS = static_cast<int>(Eval::fe_end) * static_cast<int>(SQ_NB);

    static constexpr int MAX_ACTIVE_FEATURES = 38;

    static int fill_features_sparse(int i, const TrainingDataEntry& e, int* features, float* values, int& counter, Color color)
    {
        auto& pos = *e.pos;
        //std::cout << pos << std::endl;
        Eval::BonaPiece* pieces = (color == BLACK) ?
            pos.eval_list()->piece_list_fb() :
            pos.eval_list()->piece_list_fw();
        PieceNumber target = static_cast<PieceNumber>(PIECE_NUMBER_KING + color);
        Square sq_target_k = static_cast<Square>((pieces[target] - Eval::f_king) % SQ_NB);

        // We order the features so that the resulting sparse
        // tensor is coalesced.
        int features_unordered[MAX_ACTIVE_FEATURES];
        int j = 0;
        for (PieceNumber i = PIECE_NUMBER_ZERO; i < PIECE_NUMBER_KING; ++i) {
            features_unordered[j++] = static_cast<int>(Eval::fe_end) * static_cast<int>(sq_target_k) + pieces[j];
        }
        std::sort(features_unordered, features_unordered + j);
        for (int k = 0; k < j; ++k)
        {
            int idx = counter * 2;
            features[idx] = i;
            features[idx + 1] = features_unordered[k];
            values[counter] = 1.0f;
            counter += 1;
        }
        return SQ_NB * static_cast<int>(Eval::fe_end);
    }
};

struct HalfKPFactorized {
    // Factorized features
    static constexpr int INPUTS = HalfKP::INPUTS + SQ_NB + Eval::fe_end;

    static constexpr int MAX_K_FEATURES = 1;
    static constexpr int MAX_ACTIVE_FEATURES = HalfKP::MAX_ACTIVE_FEATURES + MAX_K_FEATURES + PIECE_NUMBER_KING;

    static void fill_features_sparse(int i, const TrainingDataEntry& e, int* features, float* values, int& counter, Color color)
    {
        auto counter_before = counter;
        int offset = HalfKP::fill_features_sparse(i, e, features, values, counter, color);
        auto& pos = e.pos;
        Eval::BonaPiece* pieces = (color == BLACK) ?
            pos->eval_list()->piece_list_fb() :
            pos->eval_list()->piece_list_fw();
        PieceNumber target = static_cast<PieceNumber>(PIECE_NUMBER_KING + color);
        Square sq_target_k = static_cast<Square>((pieces[target] - Eval::f_king) % SQ_NB);
        {
            auto num_added_features = counter - counter_before;
            // king square factor
            int idx = counter * 2;
            features[idx] = i;
            features[idx + 1] = offset + sq_target_k;
            values[counter] = static_cast<float>(num_added_features);
            counter += 1;
        }
        offset += SQ_NB;

        // We order the features so that the resulting sparse
        // tensor is coalesced. Note that we can just sort
        // the parts where values are all 1.0f and leave the
        // halfk feature where it was.
        int features_unordered[PIECE_NUMBER_KING];
        int j = 0;
        for (PieceNumber i = PIECE_NUMBER_ZERO; i < PIECE_NUMBER_KING; ++i) {
            features_unordered[j++] = offset + pieces[j];
        }
        std::sort(features_unordered, features_unordered + j);
        for (int k = 0; k < j; ++k)
        {
            int idx = counter * 2;
            features[idx] = i;
            features[idx + 1] = features_unordered[k];
            values[counter] = 1.0f;
            counter += 1;
        }
    }
};

template <typename T, typename... Ts>
struct FeatureSet
{
    static_assert(sizeof...(Ts) == 0, "Currently only one feature subset supported.");

    static constexpr int INPUTS = T::INPUTS;
    static constexpr int MAX_ACTIVE_FEATURES = T::MAX_ACTIVE_FEATURES;

    static void fill_features_sparse(int i, const TrainingDataEntry& e, int* features, float* values, int& counter, Color color)
    {
        T::fill_features_sparse(i, e, features, values, counter, color);
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
        is_black = new float[size];
        outcome = new float[size];
        score = new float[size];
        black = new int[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES * 2];
        white = new int[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES * 2];
        black_values = new float[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        white_values = new float[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];

        num_active_black_features = 0;
        num_active_white_features = 0;

        std::memset(black, 0, size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES * 2 * sizeof(int));
        std::memset(white, 0, size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES * 2 * sizeof(int));

        for(int i = 0; i < entries.size(); ++i)
        {
            fill_entry(FeatureSet<Ts...>{}, i, entries[i]);
        }
    }

    int num_inputs;
    int size;

    float* is_black;
    float* outcome;
    float* score;
    int num_active_black_features;
    int num_active_white_features;
    int* black;
    int* white;
    float* black_values;
    float* white_values;

    ~SparseBatch()
    {
        delete[] is_black;
        delete[] outcome;
        delete[] score;
        delete[] black;
        delete[] white;
        delete[] black_values;
        delete[] white_values;
    }

private:

    template <typename... Ts>
    void fill_entry(FeatureSet<Ts...>, int i, const TrainingDataEntry& e)
    {
        is_black[i] = static_cast<float>(e.pos->side_to_move() == BLACK);
        outcome[i] = (e.result + 1.0f) / 2.0f;
        score[i] = e.score;
        fill_features(FeatureSet<Ts...>{}, i, e);
    }

    template <typename... Ts>
    void fill_features(FeatureSet<Ts...>, int i, const TrainingDataEntry& e)
    {
        FeatureSet<Ts...>::fill_features_sparse(i, e, black, black_values, num_active_black_features, BLACK);
        FeatureSet<Ts...>::fill_features_sparse(i, e, white, white_values, num_active_white_features, WHITE);
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

    Stream(int concurrency, const char* filename, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
        m_stream(training_data::open_sfen_input_file_parallel(concurrency, filename, cyclic, skipPredicate))
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

    AsyncStream(int concurrency, const char* filename, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
        BaseType(1, filename, cyclic, skipPredicate)
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

    FeaturedBatchStream(int concurrency, const char* filename, int batch_size, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
        BaseType(
            std::max(
                1,
                concurrency / num_feature_threads_per_reading_thread
            ),
            filename,
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

            while(!m_stop_flag.load())
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

namespace {
    bool initialized = false;

    void EnsureInitializeYaneuraOu(int num_threads) {
        if (initialized) {
            return;
        }

        //CommandLine::init(argc, argv);
        USI::init(Options);
        Bitboards::init();
        Position::init();
        Search::init();

        // エンジンオプションの"Threads"があるとは限らないので…。
        //size_t thread_num = Options.count("Threads") ? (size_t)Options["Threads"] : 1;
        Threads.set(num_threads);

        //Search::clear();
        Eval::init();
        Eval::load_eval();

        TT.resize(1024);

        omp_set_num_threads(num_threads);

        initialized = true;
    }
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
        EnsureInitializeYaneuraOu(1);

        std::vector<TrainingDataEntry> entries;
        entries.reserve(num_fens);
        for (int i = 0; i < num_fens; ++i)
        {
            auto& e = entries.emplace_back();
            e.pos = std::make_shared<Position>();
            StateInfo state_info = {};
            e.pos->set(fens[i], &state_info, Threads[0]);
            ExtMove moves[1024];
            generateMoves<MOVE_GEN_TYPE::LEGAL>(*e.pos, moves);
            e.move = moves[0];
            e.score = scores[i];
            e.ply = plies[i];
            e.result = results[i];
        }

        std::string_view feature_set(feature_set_c);
        if (feature_set == "HalfKP")
        {
            return new SparseBatch(FeatureSet<HalfKP>{}, entries);
        }
        else if (feature_set == "HalfKP^")
        {
            return new SparseBatch(FeatureSet<HalfKPFactorized>{}, entries);
        }
        fprintf(stderr, "Unknown feature_set %s\n", feature_set_c);
        return nullptr;
    }

    EXPORT Stream<SparseBatch>* CDECL create_sparse_batch_stream(const char* feature_set_c, int concurrency, const char* filename, int batch_size, bool cyclic, bool filtered, int random_fen_skipping)
    {
        EnsureInitializeYaneuraOu(concurrency);

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
        if (feature_set == "HalfKP")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKP>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate);
        }
        else if (feature_set == "HalfKP^")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKPFactorized>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate);
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
    auto stream = create_sparse_batch_stream("HalfKP^", 4, "C:\\shogi\\kifu\\suisho-wcsoc2020.20200524.shuffled\\shuffled.000.bin", 8192, true, false, 0);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i)
    {
        if (i % 100 == 0) std::cout << i << '\n';
        destroy_sparse_batch(stream->next());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << (t1 - t0).count() / 1e9 << "s\n";

    delete stream;
    stream = nullptr;
}
//*/

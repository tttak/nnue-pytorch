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

struct HalfKP {
    static constexpr int NUM_SQ = 81;
    static constexpr int NUM_PLANES = 1548; // == fe_end
    static constexpr int INPUTS = NUM_PLANES * NUM_SQ;

    static constexpr int MAX_ACTIVE_FEATURES = 38;

    static int fill_features_sparse(int i, const TrainingDataEntry& e, int* features, float* values, int& counter, Color color)
    {
        auto& pos = *e.pos;
        Eval::BonaPiece* pieces = nullptr;
        if (color == Color::BLACK) {
            pieces = pos.eval_list()->piece_list_fb();
        }
        else {
            pieces = pos.eval_list()->piece_list_fw();
        }
        PieceNumber target = static_cast<PieceNumber>(PIECE_NUMBER_KING + color);
        auto sq_target_k = static_cast<Square>((pieces[target] - Eval::BonaPiece::f_king) % SQ_NB);

        // We order the features so that the resulting sparse
        // tensor is coalesced.
        int features_unordered[38];
        for (PieceNumber i = PIECE_NUMBER_ZERO; i < PIECE_NUMBER_KING; ++i) {
            auto p = pieces[i];
            features_unordered[i] = static_cast<int>(Eval::fe_end) * static_cast<int>(sq_target_k) + p;
        }
        std::sort(features_unordered, features_unordered + PIECE_NUMBER_KING);
        for (int k = 0; k < PIECE_NUMBER_KING; ++k)
        {
            int idx = counter * 2;
            features[idx] = i;
            features[idx + 1] = features_unordered[k];
            values[counter] = 1.0f;
            counter += 1;
        }
        return INPUTS;
    }
};

struct HalfKPFactorized {
    // Factorized features
    static constexpr int K_INPUTS = HalfKP::NUM_SQ;
    static constexpr int PIECE_INPUTS = HalfKP::NUM_PLANES;
    static constexpr int INPUTS = HalfKP::INPUTS + K_INPUTS + PIECE_INPUTS;

    static constexpr int MAX_K_FEATURES = 1;
    static constexpr int MAX_PIECE_FEATURES = 38;
    static constexpr int MAX_ACTIVE_FEATURES = HalfKP::MAX_ACTIVE_FEATURES + MAX_K_FEATURES + MAX_PIECE_FEATURES;

    static void fill_features_sparse(int i, const TrainingDataEntry& e, int* features, float* values, int& counter, Color color)
    {
        auto counter_before = counter;
        int offset = HalfKP::fill_features_sparse(i, e, features, values, counter, color);

        auto& pos = *e.pos;
        Eval::BonaPiece* pieces = nullptr;
        if (color == Color::BLACK) {
            pieces = pos.eval_list()->piece_list_fb();
        }
        else {
            pieces = pos.eval_list()->piece_list_fw();
        }

        {
            auto num_added_features = counter - counter_before;
            // king square factor
            PieceNumber target = static_cast<PieceNumber>(PIECE_NUMBER_KING + color);
            auto sq_target_k = static_cast<Square>((pieces[target] - Eval::BonaPiece::f_king) % SQ_NB);
            int idx = counter * 2;
            features[idx] = i;
            features[idx + 1] = offset + static_cast<int>(sq_target_k);
            values[counter] = static_cast<float>(num_added_features);
            counter += 1;
        }
        offset += K_INPUTS;

        // We order the features so that the resulting sparse
        // tensor is coalesced. Note that we can just sort
        // the parts where values are all 1.0f and leave the
        // halfk feature where it was.
        int features_unordered[38];
        for (PieceNumber i = PIECE_NUMBER_ZERO; i < PIECE_NUMBER_KING; ++i) {
            auto p = pieces[i];
            features_unordered[i] = offset + p;
        }
        std::sort(features_unordered, features_unordered + PIECE_NUMBER_KING);
        for (int k = 0; k < PIECE_NUMBER_KING; ++k)
        {
            int idx = counter * 2;
            features[idx] = i;
            features[idx + 1] = features_unordered[k];
            values[counter] = 1.0f;
            counter += 1;
        }
    }
};

// struct HalfKA {
//     static constexpr int NUM_SQ = 64;
//     static constexpr int NUM_PT = 12;
//     static constexpr int NUM_PLANES = (NUM_SQ * NUM_PT + 1);
//     static constexpr int INPUTS = NUM_PLANES * NUM_SQ;

//     static constexpr int MAX_ACTIVE_FEATURES = 32;

//     static int feature_index(Color color, Square ksq, Square sq, Piece p)
//     {
//         auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
//         return 1 + static_cast<int>(orient_flip(color, sq)) + p_idx * NUM_SQ + static_cast<int>(ksq) * NUM_PLANES;
//     }

//     static int fill_features_sparse(int i, const TrainingDataEntry& e, int* features, float* values, int& counter, Color color)
//     {
//         auto& pos = e.pos;
//         auto pieces = pos.piecesBB();
//         auto ksq = pos.kingSquare(color);

//         // We order the features so that the resulting sparse
//         // tensor is coalesced.
//         int features_unordered[32];
//         int j = 0;
//         for(Square sq : pieces)
//         {
//             auto p = pos.pieceAt(sq);
//             features_unordered[j++] = feature_index(color, orient_flip(color, ksq), sq, p);
//         }
//         std::sort(features_unordered, features_unordered + j);
//         for (int k = 0; k < j; ++k)
//         {
//             int idx = counter * 2;
//             features[idx] = i;
//             features[idx + 1] = features_unordered[k];
//             values[counter] = 1.0f;
//             counter += 1;
//         }
//         return INPUTS;
//     }
// };

// struct HalfKAFactorized {
//     // Factorized features
//     static constexpr int PIECE_INPUTS = HalfKA::NUM_SQ * HalfKA::NUM_PT;
//     static constexpr int INPUTS = HalfKA::INPUTS + PIECE_INPUTS;

//     static constexpr int MAX_PIECE_FEATURES = 32;
//     static constexpr int MAX_ACTIVE_FEATURES = HalfKA::MAX_ACTIVE_FEATURES + MAX_PIECE_FEATURES;

//     static void fill_features_sparse(int i, const TrainingDataEntry& e, int* features, float* values, int& counter, Color color)
//     {
//         auto counter_before = counter;
//         int offset = HalfKA::fill_features_sparse(i, e, features, values, counter, color);
//         auto& pos = e.pos;
//         auto pieces = pos.piecesBB();

//         // We order the features so that the resulting sparse
//         // tensor is coalesced. Note that we can just sort
//         // the parts where values are all 1.0f and leave the
//         // halfk feature where it was.
//         int features_unordered[32];
//         int j = 0;
//         for(Square sq : pieces)
//         {
//             auto p = pos.pieceAt(sq);
//             auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
//             features_unordered[j++] = offset + (p_idx * HalfKA::NUM_SQ) + static_cast<int>(orient_flip(color, sq));
//         }
//         std::sort(features_unordered, features_unordered + j);
//         for (int k = 0; k < j; ++k)
//         {
//             int idx = counter * 2;
//             features[idx] = i;
//             features[idx + 1] = features_unordered[k];
//             values[counter] = 1.0f;
//             counter += 1;
//         }
//     }
// };

struct HalfKPE9 {
    static constexpr int NUM_SQ = 81;
    static constexpr int NUM_PLANES = 1548; // == fe_end
    static constexpr int INPUTS = NUM_PLANES * NUM_SQ * 9;

    static constexpr int MAX_ACTIVE_FEATURES = 38;

    static Square GetSquareFromBonaPiece(Eval::BonaPiece p) {
        if (p < Eval::fe_hand_end) {
            return SQ_NB;
        }
        else {
            return static_cast<Square>((p - Eval::fe_hand_end) % SQ_NB);
        }
    }

    static int GetEffectCount(const Position& pos, Square sq_p, Color perspective_org, Color perspective) {
        if (sq_p == SQ_NB) {
            return 0;
        }
        else {
            if (perspective_org == WHITE) {
                sq_p = Inv(sq_p);
            }
            return std::min(int(pos.board_effect[perspective].effect(sq_p)), 2);
        }
    }

    static int fill_features_sparse(int i, const TrainingDataEntry& e, int* features, float* values, int& counter, Color color)
    {
        auto& pos = *e.pos;
        Eval::BonaPiece* pieces = nullptr;
        if (color == Color::BLACK) {
            pieces = pos.eval_list()->piece_list_fb();
        }
        else {
            pieces = pos.eval_list()->piece_list_fw();
        }
        PieceNumber target = static_cast<PieceNumber>(PIECE_NUMBER_KING + color);
        auto sq_target_k = static_cast<Square>((pieces[target] - Eval::BonaPiece::f_king) % SQ_NB);

#if 0
        sync_cout << "pos=" << sync_endl << pos << sync_endl;
        sync_cout << "color=" << color << sync_endl;
        sync_cout << "sq_target_k=" << sq_target_k << sync_endl;
#endif

        // We order the features so that the resulting sparse
        // tensor is coalesced.
        int features_unordered[38];
        for (PieceNumber i = PIECE_NUMBER_ZERO; i < PIECE_NUMBER_KING; ++i) {
            auto p = pieces[i];
            Square sq_p = GetSquareFromBonaPiece(p);
            int effect1 = GetEffectCount(pos, sq_p, color,  color);
            int effect2 = GetEffectCount(pos, sq_p, color, ~color);
            features_unordered[i] = static_cast<int>(Eval::fe_end) * static_cast<int>(sq_target_k) + p
                                  + static_cast<int>(Eval::fe_end) * static_cast<int>(SQ_NB) * (effect1 * 3 + effect2);

#if 0
            sync_cout << "p=" << p << ", sq_p=" << sq_p << ", effect1=" << effect1 << ", effect2=" << effect2
                      << ", features_unordered[" << (int)i << "]=" << features_unordered[i]
                      << sync_endl;
#endif

        }
        std::sort(features_unordered, features_unordered + PIECE_NUMBER_KING);
        for (int k = 0; k < PIECE_NUMBER_KING; ++k)
        {
            int idx = counter * 2;
            features[idx] = i;
            features[idx + 1] = features_unordered[k];
            values[counter] = 1.0f;
            counter += 1;
        }
        return INPUTS;
    }
};

struct HalfKP_KSDG {
    static constexpr int INPUTS = 177876 + 12672;
    static constexpr int MAX_ACTIVE_FEATURES = 38 + 24;

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

    static int fill_features_sparse(int i, const TrainingDataEntry& e, int* features, float* values, int& counter, Color color)
    {
        int features_unordered[MAX_ACTIVE_FEATURES];
        int features_index = 0;
        auto& pos = *e.pos;

        // ----- KingSafety_DistinguishGolds

        // color側の玉のマス（先手目線）
        SquareWithWall sqww_king = to_sqww(pos.king_square(color));

#if 0
        sync_cout << "pos=" << sync_endl << pos << sync_endl;
        sync_cout << "color=" << color << sync_endl;
        sync_cout << "sqww_king=" << sqww_king << sync_endl;
#endif

        // 24近傍をループ
        for (Effect24::Direct dir : Effect24::Direct()) {
            SquareWithWall sqww = sqww_king + DirectToDeltaWW(dir);
            int index_caluculated = 0;

            // 盤内の場合
            if (is_ok(sqww)) {
                Square sq = sqww_to_sq(sqww);
                index_caluculated = MakeIndex(color, dir, pos.piece_on(sq)
                        , GetEffectCount(pos, sq,  color)
                        , GetEffectCount(pos, sq, ~color)
                    );

#if 0
                sync_cout << (int)dir
                    << ":" << sq
                    << "," << pos.piece_on(sq)
                    << "," << GetEffectCount(pos, sq,  color)
                    << "," << GetEffectCount(pos, sq, ~color)
                    << sync_endl;
#endif

            }

            // 盤外の場合
            else {
                index_caluculated = MakeIndex(color, dir, PIECE_WALL, 0, 0);

#if 0
                sync_cout << (int)dir << ":bangai" << sync_endl;
#endif
            }

            features_unordered[features_index] = index_caluculated;
            features_index++;
        }

        // ----- HalfKP
        Eval::BonaPiece* pieces = nullptr;
        if (color == Color::BLACK) {
            pieces = pos.eval_list()->piece_list_fb();
        }
        else {
            pieces = pos.eval_list()->piece_list_fw();
        }
        PieceNumber target = static_cast<PieceNumber>(PIECE_NUMBER_KING + color);
        auto sq_target_k = static_cast<Square>((pieces[target] - Eval::BonaPiece::f_king) % SQ_NB);

        for (PieceNumber i = PIECE_NUMBER_ZERO; i < PIECE_NUMBER_KING; ++i) {
            auto p = pieces[i];
            features_unordered[features_index] = 12672 + static_cast<int>(Eval::fe_end) * static_cast<int>(sq_target_k) + p;
            features_index++;
        }

        // -----
        std::sort(features_unordered, features_unordered + MAX_ACTIVE_FEATURES);
        for (int k = 0; k < MAX_ACTIVE_FEATURES; ++k)
        {
            int idx = counter * 2;
            features[idx] = i;
            features[idx + 1] = features_unordered[k];
            values[counter] = 1.0f;
            counter += 1;
        }

        return INPUTS;
    }
};

struct HalfKP_KK {
    static constexpr int INPUTS = 125388 + 6561;
    static constexpr int MAX_ACTIVE_FEATURES = 38 + 1;

    static int MakeIndex(Color perspective, Square ksq1, Square ksq2) {
        if (perspective == WHITE) {
            ksq1 = Inv(ksq1);
            ksq2 = Inv(ksq2);
        }

        return static_cast<int>(ksq1) * static_cast<int>(SQ_NB) + ksq2;
    }

    static int fill_features_sparse(int i, const TrainingDataEntry& e, int* features, float* values, int& counter, Color color)
    {
        int features_unordered[MAX_ACTIVE_FEATURES];
        int features_index = 0;
        auto& pos = *e.pos;

        // ----- KK
        features_unordered[features_index] = MakeIndex(color, pos.king_square(color), pos.king_square(~color));
        features_index++;

        // ----- HalfKP
        Eval::BonaPiece* pieces = nullptr;
        if (color == Color::BLACK) {
            pieces = pos.eval_list()->piece_list_fb();
        }
        else {
            pieces = pos.eval_list()->piece_list_fw();
        }
        PieceNumber target = static_cast<PieceNumber>(PIECE_NUMBER_KING + color);
        auto sq_target_k = static_cast<Square>((pieces[target] - Eval::BonaPiece::f_king) % SQ_NB);

        for (PieceNumber i = PIECE_NUMBER_ZERO; i < PIECE_NUMBER_KING; ++i) {
            auto p = pieces[i];
            features_unordered[features_index] = 6561 + static_cast<int>(Eval::fe_end) * static_cast<int>(sq_target_k) + p;
            features_index++;
        }

        // -----
        std::sort(features_unordered, features_unordered + MAX_ACTIVE_FEATURES);
        for (int k = 0; k < MAX_ACTIVE_FEATURES; ++k)
        {
            int idx = counter * 2;
            features[idx] = i;
            features[idx + 1] = features_unordered[k];
            values[counter] = 1.0f;
            counter += 1;
        }

        return INPUTS;
    }
};

struct HalfKPE4 {
    static constexpr int NUM_SQ = 81;
    static constexpr int NUM_PLANES = 1548; // == fe_end
    static constexpr int INPUTS = NUM_PLANES * NUM_SQ * 4;

    static constexpr int MAX_ACTIVE_FEATURES = 38;

    static Square GetSquareFromBonaPiece(Eval::BonaPiece p) {
        if (p < Eval::fe_hand_end) {
            return SQ_NB;
        }
        else {
            return static_cast<Square>((p - Eval::fe_hand_end) % SQ_NB);
        }
    }

    static int GetEffectCount(const Position& pos, Square sq_p, Color perspective_org, Color perspective) {
        if (sq_p == SQ_NB) {
            return 0;
        }
        else {
            if (perspective_org == WHITE) {
                sq_p = Inv(sq_p);
            }
            return std::min(int(pos.board_effect[perspective].effect(sq_p)), 1);
        }
    }

    static int fill_features_sparse(int i, const TrainingDataEntry& e, int* features, float* values, int& counter, Color color)
    {
        auto& pos = *e.pos;
        Eval::BonaPiece* pieces = nullptr;
        if (color == Color::BLACK) {
            pieces = pos.eval_list()->piece_list_fb();
        }
        else {
            pieces = pos.eval_list()->piece_list_fw();
        }
        PieceNumber target = static_cast<PieceNumber>(PIECE_NUMBER_KING + color);
        auto sq_target_k = static_cast<Square>((pieces[target] - Eval::BonaPiece::f_king) % SQ_NB);

#if 0
        sync_cout << "pos=" << sync_endl << pos << sync_endl;
        sync_cout << "color=" << color << sync_endl;
        sync_cout << "sq_target_k=" << sq_target_k << sync_endl;
#endif

        // We order the features so that the resulting sparse
        // tensor is coalesced.
        int features_unordered[38];
        for (PieceNumber i = PIECE_NUMBER_ZERO; i < PIECE_NUMBER_KING; ++i) {
            auto p = pieces[i];
            Square sq_p = GetSquareFromBonaPiece(p);
            int effect1 = GetEffectCount(pos, sq_p, color,  color);
            int effect2 = GetEffectCount(pos, sq_p, color, ~color);
            features_unordered[i] = static_cast<int>(Eval::fe_end) * static_cast<int>(sq_target_k) + p
                                  + static_cast<int>(Eval::fe_end) * static_cast<int>(SQ_NB) * (effect1 * 2 + effect2);

#if 0
            sync_cout << "p=" << p << ", sq_p=" << sq_p << ", effect1=" << effect1 << ", effect2=" << effect2
                      << ", features_unordered[" << (int)i << "]=" << features_unordered[i]
                      << sync_endl;
#endif

        }
        std::sort(features_unordered, features_unordered + PIECE_NUMBER_KING);
        for (int k = 0; k < PIECE_NUMBER_KING; ++k)
        {
            int idx = counter * 2;
            features[idx] = i;
            features[idx + 1] = features_unordered[k];
            values[counter] = 1.0f;
            counter += 1;
        }
        return INPUTS;
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
        is_white = new float[size];
        outcome = new float[size];
        score = new float[size];
        white = new int[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES * 2];
        black = new int[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES * 2];
        white_values = new float[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        black_values = new float[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];

        num_active_white_features = 0;
        num_active_black_features = 0;

        std::memset(white, 0, size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES * 2 * sizeof(int));
        std::memset(black, 0, size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES * 2 * sizeof(int));

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
    int* white;
    int* black;
    float* white_values;
    float* black_values;

    ~SparseBatch()
    {
        delete[] is_white;
        delete[] outcome;
        delete[] score;
        delete[] white;
        delete[] black;
        delete[] white_values;
        delete[] black_values;
    }

private:

    template <typename... Ts>
    void fill_entry(FeatureSet<Ts...>, int i, const TrainingDataEntry& e)
    {
        is_white[i] = static_cast<float>(e.pos->side_to_move() == Color::BLACK);
        outcome[i] = (e.result + 1.0f) / 2.0f;
        score[i] = e.score;
        fill_features(FeatureSet<Ts...>{}, i, e);
    }

    template <typename... Ts>
    void fill_features(FeatureSet<Ts...>, int i, const TrainingDataEntry& e)
    {
        FeatureSet<Ts...>::fill_features_sparse(i, e, white, white_values, num_active_white_features, Color::BLACK);
        FeatureSet<Ts...>::fill_features_sparse(i, e, black, black_values, num_active_black_features, Color::WHITE);
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
    //Position::init();
    //Search::init();

    Threads.set(1);

    //Eval::init();

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
        if (feature_set == "HalfKP")
        {
            return new SparseBatch(FeatureSet<HalfKP>{}, entries);
        }
        else if (feature_set == "HalfKP^")
        {
            return new SparseBatch(FeatureSet<HalfKPFactorized>{}, entries);
        }
        // else if (feature_set == "HalfKA")
        // {
        //     return new SparseBatch(FeatureSet<HalfKA>{}, entries);
        // }
        // else if (feature_set == "HalfKA^")
        // {
        //     return new SparseBatch(FeatureSet<HalfKAFactorized>{}, entries);
        // }
        else if (feature_set == "HalfKPE9")
        {
            return new SparseBatch(FeatureSet<HalfKPE9>{}, entries);
        }
        else if (feature_set == "HalfKP_KSDG")
        {
            return new SparseBatch(FeatureSet<HalfKP_KSDG>{}, entries);
        }
        else if (feature_set == "HalfKP_KK")
        {
            return new SparseBatch(FeatureSet<HalfKP_KK>{}, entries);
        }
        else if (feature_set == "HalfKPE4")
        {
            return new SparseBatch(FeatureSet<HalfKPE4>{}, entries);
        }
        fprintf(stderr, "Unknown feature_set %s\n", feature_set_c);
        return nullptr;
    }

    EXPORT Stream<SparseBatch>* CDECL create_sparse_batch_stream(const char* feature_set_c, int concurrency, const char* filename, int batch_size, bool cyclic, bool filtered, int random_fen_skipping)
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
        if (feature_set == "HalfKP")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKP>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate);
        }
        else if (feature_set == "HalfKP^")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKPFactorized>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate);
        }
        // else if (feature_set == "HalfKA")
        // {
        //     return new FeaturedBatchStream<FeatureSet<HalfKA>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate);
        // }
        // else if (feature_set == "HalfKA^")
        // {
        //     return new FeaturedBatchStream<FeatureSet<HalfKAFactorized>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate);
        // }
        else if (feature_set == "HalfKPE9")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKPE9>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate);
        }
        else if (feature_set == "HalfKP_KSDG")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKP_KSDG>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate);
        }
        else if (feature_set == "HalfKP_KK")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKP_KK>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate);
        }
        else if (feature_set == "HalfKPE4")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKPE4>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate);
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
    auto stream = create_sparse_batch_stream("HalfKP^", 4, R"(C:\shogi\training_data\suisho5.shuffled.qsearch\shuffled.bin)", 8192, true, false, 0);
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

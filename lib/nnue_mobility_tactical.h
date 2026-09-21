#pragma once

// Loader-local definition of Experiment 84 mobility/tactical v1.  Keep this
// independent from the YaneuraOu copy and verify both with staged parity tests.

#include <array>
#include <algorithm>
#include <cstdint>

namespace NnueMobilityTactical {

constexpr std::size_t kDimensions = 8;
using RawFeatures = std::array<std::uint16_t, kDimensions>;
using NormalizedFeatures = std::array<float, kDimensions>;

inline std::uint8_t bucket(const PieceType pt) {
    switch (pt) {
    case PAWN: return 1;
    case LANCE: case KNIGHT: return 2;
    case SILVER: case GOLD: case PRO_PAWN: case PRO_LANCE:
    case PRO_KNIGHT: case PRO_SILVER: return 3;
    case BISHOP: return 4;
    case ROOK: return 5;
    case HORSE: case DRAGON: return 6;
    default: return 0;
    }
}

inline std::uint16_t rook_mobility(const Position& pos, const Color side) {
    const Bitboard occupied = pos.pieces();
    const Bitboard own = pos.pieces(side);
    Bitboard sliders = pos.pieces(side, ROOK_DRAGON);
    std::uint16_t result = 0;
    while (sliders) {
        const Square square = sliders.pop();
        result += static_cast<std::uint16_t>(
            (rookEffect(square, occupied) & ~own).pop_count());
    }
    return result;
}

inline std::uint16_t bishop_mobility(const Position& pos, const Color side) {
    const Bitboard occupied = pos.pieces();
    const Bitboard own = pos.pieces(side);
    Bitboard sliders = pos.pieces(side, BISHOP_HORSE);
    std::uint16_t result = 0;
    while (sliders) {
        const Square square = sliders.pop();
        result += static_cast<std::uint16_t>(
            (bishopEffect(square, occupied) & ~own).pop_count());
    }
    return result;
}

inline std::pair<std::uint16_t, std::uint8_t> captures(
    const Position& pos, const Color side) {
    std::uint16_t count = 0;
    std::uint8_t maximum = 0;
    Bitboard targets = pos.pieces(~side);
    while (targets) {
        const Square target = targets.pop();
        const PieceType pt = type_of(pos.piece_on(target));
        if (pt == KING || !pos.effected_to(side, target))
            continue;
        ++count;
        maximum = std::max(maximum, bucket(pt));
    }
    return {count, maximum};
}

inline RawFeatures raw(const Position& pos) {
    const Color us = pos.side_to_move();
    const Color them = ~us;
    const auto us_capture = captures(pos, us);
    const auto them_capture = captures(pos, them);
    return {{
        rook_mobility(pos, us), rook_mobility(pos, them),
        bishop_mobility(pos, us), bishop_mobility(pos, them),
        us_capture.first, them_capture.first,
        us_capture.second, them_capture.second,
    }};
}

inline NormalizedFeatures normalize(const RawFeatures& value) {
    constexpr std::array<float, kDimensions> scale =
        {{20.0f, 20.0f, 16.0f, 16.0f, 8.0f, 8.0f, 6.0f, 6.0f}};
    NormalizedFeatures result{};
    for (std::size_t i = 0; i < result.size(); ++i)
        result[i] = std::min(static_cast<float>(value[i]) / scale[i], 1.0f);
    return result;
}

}  // namespace NnueMobilityTactical

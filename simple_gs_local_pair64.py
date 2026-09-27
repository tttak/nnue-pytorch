"""Experiment 124 R5 silver/gold-like LocalPair64 feature contract."""

from __future__ import annotations

from dataclasses import dataclass


GS_LOCALPAIR64_TYPE = "gs_local_pair64"
GS_LOCALPAIR64_SCHEMA_VERSION = 1
GS_LOCALPAIR64_MAPPING_VERSION = \
    "r5_silver_goldlike_chebyshev2_hm2mirror_v1"
GS_LOCALPAIR64_WIDTH = 64
GS_LOCALPAIR64_TRANSFORMED = 64
GS_LOCALPAIR64_PROJECTION_OUTPUTS = 16
GS_LOCALPAIR64_INIT_TABLE_QUANTUM = 16
GS_LOCALPAIR64_QUANT_SCALE = 127.0
GS_LOCALPAIR64_CLASSES = 2
GS_LOCALPAIR64_STATES = 2 * GS_LOCALPAIR64_CLASSES
GS_LOCALPAIR64_MAX_ACTIVE = 256


def _build_square_pairs():
    pairs = []
    for a in range(81):
        fa, ra = divmod(a, 9)
        for b in range(a + 1, 81):
            fb, rb = divmod(b, 9)
            if abs(fa - fb) <= 2 and abs(ra - rb) <= 2:
                pairs.append((a, b))
    return tuple(pairs)


GS_LOCALPAIR64_SQUARE_PAIR_LIST = _build_square_pairs()
GS_LOCALPAIR64_SQUARE_PAIRS = len(GS_LOCALPAIR64_SQUARE_PAIR_LIST)
GS_LOCALPAIR64_PAIR_TO_INDEX = {
    pair: index for index, pair in enumerate(GS_LOCALPAIR64_SQUARE_PAIR_LIST)
}
GS_LOCALPAIR64_FEATURES = (
    GS_LOCALPAIR64_SQUARE_PAIRS
    * GS_LOCALPAIR64_STATES * GS_LOCALPAIR64_STATES)


def inv_square(square: int) -> int:
    return 80 - int(square)


def mirror_square(square: int) -> int:
    square = int(square)
    return (8 - square // 9) * 9 + square % 9


def orient_square(square: int, perspective: int, mirror: bool) -> int:
    square = inv_square(square) if int(perspective) else int(square)
    return mirror_square(square) if mirror else square


def compact_square_pair(square_a: int, square_b: int) -> int:
    a, b = sorted((int(square_a), int(square_b)))
    if a == b:
        return -1
    return GS_LOCALPAIR64_PAIR_TO_INDEX.get((a, b), -1)


def feature_index(square_a: int, state_a: int,
                  square_b: int, state_b: int) -> int:
    if square_b < square_a:
        square_a, square_b = square_b, square_a
        state_a, state_b = state_b, state_a
    pair = compact_square_pair(square_a, square_b)
    if pair < 0:
        return -1
    if not (0 <= state_a < GS_LOCALPAIR64_STATES
            and 0 <= state_b < GS_LOCALPAIR64_STATES):
        raise ValueError("GS LocalPair64 state is outside the R5 contract")
    return (pair * GS_LOCALPAIR64_STATES * GS_LOCALPAIR64_STATES
            + state_a * GS_LOCALPAIR64_STATES + state_b)


@dataclass(frozen=True)
class Piece:
    square: int
    owner: int
    piece_class: int  # SILVER=0, GOLD_LIKE=1


def enumerate_indices(pieces, perspective: int, friend_king_square: int):
    king = inv_square(friend_king_square) if perspective else friend_king_square
    mirror = king >= 45
    oriented = []
    for piece in pieces:
        square = orient_square(piece.square, perspective, mirror)
        state = ((piece.owner ^ perspective) * GS_LOCALPAIR64_CLASSES
                 + piece.piece_class)
        oriented.append((square, state))
    oriented.sort()
    out = []
    for i, (sa, sta) in enumerate(oriented):
        for sb, stb in oriented[i + 1:]:
            if sb // 9 - sa // 9 > 2:
                break
            index = feature_index(sa, sta, sb, stb)
            if index >= 0:
                out.append(index)
    return sorted(out)


assert GS_LOCALPAIR64_SQUARE_PAIRS == 720
assert GS_LOCALPAIR64_FEATURES == 11_520

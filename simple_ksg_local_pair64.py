"""Experiment 123 KSG LocalPair64 feature contract.

The geometry, owner normalization, horizontal mirror, physical unordered-pair
canonicalization and 64-wide branch contract are inherited from Experiment
122.  Only the represented piece classes differ: knight, silver and
gold-like.
"""

from __future__ import annotations

from dataclasses import dataclass


KSG_LOCALPAIR64_TYPE = "ksg_local_pair64"
KSG_LOCALPAIR64_SCHEMA_VERSION = 1
KSG_LOCALPAIR64_MAPPING_VERSION = \
    "r2_knight_silver_goldlike_chebyshev2_hm2mirror_v1"
KSG_LOCALPAIR64_WIDTH = 64
KSG_LOCALPAIR64_TRANSFORMED = 64
KSG_LOCALPAIR64_PROJECTION_OUTPUTS = 16
KSG_LOCALPAIR64_INIT_TABLE_QUANTUM = 16
KSG_LOCALPAIR64_QUANT_SCALE = 127.0

# R2: knight, silver, gold-like.  Gold-like includes GOLD, PRO_PAWN,
# PRO_LANCE, PRO_KNIGHT and PRO_SILVER in the loader/C++ extractor.
KSG_LOCALPAIR64_CLASSES = 3
KSG_LOCALPAIR64_STATES = 2 * KSG_LOCALPAIR64_CLASSES
KSG_LOCALPAIR64_MAX_ACTIVE = 256


def _build_square_pairs():
    pairs = []
    for a in range(81):
        fa, ra = divmod(a, 9)
        for b in range(a + 1, 81):
            fb, rb = divmod(b, 9)
            if abs(fa - fb) <= 2 and abs(ra - rb) <= 2:
                pairs.append((a, b))
    return tuple(pairs)


KSG_LOCALPAIR64_SQUARE_PAIR_LIST = _build_square_pairs()
KSG_LOCALPAIR64_SQUARE_PAIRS = len(KSG_LOCALPAIR64_SQUARE_PAIR_LIST)
KSG_LOCALPAIR64_PAIR_TO_INDEX = {
    pair: index for index, pair in enumerate(KSG_LOCALPAIR64_SQUARE_PAIR_LIST)
}
KSG_LOCALPAIR64_FEATURES = (
    KSG_LOCALPAIR64_SQUARE_PAIRS
    * KSG_LOCALPAIR64_STATES * KSG_LOCALPAIR64_STATES)


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
    return KSG_LOCALPAIR64_PAIR_TO_INDEX.get((a, b), -1)


def feature_index(square_a: int, state_a: int,
                  square_b: int, state_b: int) -> int:
    if square_b < square_a:
        square_a, square_b = square_b, square_a
        state_a, state_b = state_b, state_a
    pair = compact_square_pair(square_a, square_b)
    if pair < 0:
        return -1
    if not (0 <= state_a < KSG_LOCALPAIR64_STATES
            and 0 <= state_b < KSG_LOCALPAIR64_STATES):
        raise ValueError("KSG LocalPair64 state is outside the R2 contract")
    return (pair * KSG_LOCALPAIR64_STATES * KSG_LOCALPAIR64_STATES
            + state_a * KSG_LOCALPAIR64_STATES + state_b)


@dataclass(frozen=True)
class Piece:
    square: int
    owner: int       # absolute BLACK=0 / WHITE=1
    piece_class: int # KNIGHT=0, SILVER=1, GOLD_LIKE=2


def enumerate_indices(pieces, perspective: int, friend_king_square: int):
    king = inv_square(friend_king_square) \
        if perspective else friend_king_square
    mirror = king >= 45
    oriented = []
    for piece in pieces:
        square = orient_square(piece.square, perspective, mirror)
        state = ((piece.owner ^ perspective) * KSG_LOCALPAIR64_CLASSES
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


assert KSG_LOCALPAIR64_SQUARE_PAIRS == 720
assert KSG_LOCALPAIR64_FEATURES == 25_920

"""Experiment 122 LocalPair64 feature contract.

Squares use YaneuraOu's file-major numbering (``file * 9 + rank``).
Physical pairs are unordered; each state remains attached to its canonical
square endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass


LOCALPAIR64_TYPE = "nonpawn_local_pair64"
LOCALPAIR64_SCHEMA_VERSION = 1
LOCALPAIR64_MAPPING_VERSION = "l4_chebyshev2_goldlike_hm2mirror_v1"
LOCALPAIR64_WIDTH = 64
LOCALPAIR64_TRANSFORMED = 64
LOCALPAIR64_PROJECTION_OUTPUTS = 16
LOCALPAIR64_INIT_TABLE_QUANTUM = 16
LOCALPAIR64_QUANT_SCALE = 127.0

# L4: lance, knight, silver, gold-like, bishop, horse, rook, dragon.
LOCALPAIR64_CLASSES = 8
LOCALPAIR64_STATES = 2 * LOCALPAIR64_CLASSES
LOCALPAIR64_MAX_ACTIVE = 1024


def _build_square_pairs():
    pairs = []
    for a in range(81):
        fa, ra = divmod(a, 9)
        for b in range(a + 1, 81):
            fb, rb = divmod(b, 9)
            if abs(fa - fb) <= 2 and abs(ra - rb) <= 2:
                pairs.append((a, b))
    return tuple(pairs)


LOCALPAIR64_SQUARE_PAIR_LIST = _build_square_pairs()
LOCALPAIR64_SQUARE_PAIRS = len(LOCALPAIR64_SQUARE_PAIR_LIST)
LOCALPAIR64_PAIR_TO_INDEX = {
    pair: index for index, pair in enumerate(LOCALPAIR64_SQUARE_PAIR_LIST)
}
LOCALPAIR64_FEATURES = (
    LOCALPAIR64_SQUARE_PAIRS * LOCALPAIR64_STATES * LOCALPAIR64_STATES)


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
    return LOCALPAIR64_PAIR_TO_INDEX.get((a, b), -1)


def feature_index(square_a: int, state_a: int,
                  square_b: int, state_b: int) -> int:
    if square_b < square_a:
        square_a, square_b = square_b, square_a
        state_a, state_b = state_b, state_a
    pair = compact_square_pair(square_a, square_b)
    if pair < 0:
        return -1
    if not (0 <= state_a < LOCALPAIR64_STATES
            and 0 <= state_b < LOCALPAIR64_STATES):
        raise ValueError("LocalPair64 state is outside the L4 contract")
    return (pair * LOCALPAIR64_STATES * LOCALPAIR64_STATES
            + state_a * LOCALPAIR64_STATES + state_b)


@dataclass(frozen=True)
class Piece:
    square: int
    owner: int       # absolute BLACK=0 / WHITE=1
    piece_class: int # L4 class [0,8)


def enumerate_indices(pieces, perspective: int, friend_king_square: int):
    king = inv_square(friend_king_square) if perspective else friend_king_square
    mirror = king >= 45
    oriented = []
    for piece in pieces:
        square = orient_square(piece.square, perspective, mirror)
        state = (piece.owner ^ perspective) * LOCALPAIR64_CLASSES \
            + piece.piece_class
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


assert LOCALPAIR64_SQUARE_PAIRS == 720
assert LOCALPAIR64_FEATURES == 184_320

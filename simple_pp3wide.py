"""Shared constants and reference indexing for the Shogi PP_3Wide v1 feature.

The production extractor lives in C++, but this small pure-Python reference is
used by diagnostics and contract tests.  Squares use YaneuraOu's file-major
encoding: ``square = file * 9 + rank``.
"""

from __future__ import annotations

from dataclasses import dataclass


PP3WIDE_TYPE = "pawn_lance_pp3wide"
PP3WIDE_INIT_MODES = ("zero", "quantized_random")
PP3WIDE_STATES = 4  # relative owner (friend/enemy) x (pawn/lance)
PP3WIDE_SAME_FILE_PAIRS = 9 * 36
PP3WIDE_ADJACENT_FILE_PAIRS = 8 * 81
PP3WIDE_SQUARE_PAIRS = (
    PP3WIDE_SAME_FILE_PAIRS + PP3WIDE_ADJACENT_FILE_PAIRS)
PP3WIDE_FEATURES = PP3WIDE_SQUARE_PAIRS * PP3WIDE_STATES**2
PP3WIDE_MAX_ACTIVE = 256
PP3WIDE_QUANT_SCALE = 127.0
PP3WIDE_SCHEMA_VERSION = 1
PP3WIDE_MAPPING_VERSION = "pl_board_unordered_file1_hm2mirror_v1"


def inv_square(square: int) -> int:
    return 80 - int(square)


def mirror_square(square: int) -> int:
    square = int(square)
    return (8 - square // 9) * 9 + square % 9


def orient_square(square: int, perspective: int, mirror: bool) -> int:
    """Orient for fixed BLACK(0)/WHITE(1) perspective, then HM2 mirror."""
    square = inv_square(square) if int(perspective) else int(square)
    return mirror_square(square) if mirror else square


def compact_square_pair(square_a: int, square_b: int) -> int:
    """Return [0,972), or -1 when the two squares are outside 3Wide."""
    a, b = sorted((int(square_a), int(square_b)))
    if a == b:
        return -1
    fa, ra = divmod(a, 9)
    fb, rb = divmod(b, 9)
    delta = fb - fa
    if delta == 0:
        # a<b implies ra<rb.  Triangular rank-pair numbering.
        return fa * 36 + rb * (rb - 1) // 2 + ra
    if delta == 1:
        return PP3WIDE_SAME_FILE_PAIRS + fa * 81 + ra * 9 + rb
    return -1


def feature_index(square_a: int, state_a: int,
                  square_b: int, state_b: int) -> int:
    """Canonical unordered pair index with state kept attached to its square."""
    if square_b < square_a:
        square_a, square_b = square_b, square_a
        state_a, state_b = state_b, state_a
    pair = compact_square_pair(square_a, square_b)
    if pair < 0:
        return -1
    if not (0 <= state_a < 4 and 0 <= state_b < 4):
        raise ValueError("PP_3Wide state must be in [0,4)")
    return pair * 16 + state_a * 4 + state_b


@dataclass(frozen=True)
class Piece:
    square: int
    owner: int       # absolute BLACK=0 / WHITE=1
    piece_class: int # pawn=0 / lance=1


def enumerate_indices(pieces, perspective: int, friend_king_square: int):
    """Reference full enumeration for already filtered unpromoted pieces."""
    king = inv_square(friend_king_square) if perspective else friend_king_square
    mirror = king >= 45
    oriented = []
    for piece in pieces:
        square = orient_square(piece.square, perspective, mirror)
        relative_owner = piece.owner ^ perspective
        state = relative_owner * 2 + piece.piece_class
        oriented.append((square, state))
    oriented.sort()
    out = []
    for i, (sa, sta) in enumerate(oriented):
        for sb, stb in oriented[i + 1:]:
            index = feature_index(sa, sta, sb, stb)
            if index >= 0:
                out.append(index)
    return sorted(out)


assert PP3WIDE_FEATURES == 15_552

"""Experiment 69 feature layouts.

These blocks describe the physical row layouts used by converted checkpoints.
Experiment 69 used explicit old-index -> new-index tables for reproducibility;
the normal training-data loader now also extracts the D layout
``HalfKA_HM1_NoDG_KSDG3_NoDG`` natively.
"""

from collections import OrderedDict

from feature_block import FeatureBlock


KSDG3_DIM = 12_672
HALFKA_HM1_DG_DIM = 45 * 2_358
HALFKA_NODG_DIM = 81 * 1_710
HALFKA_HM1_NODG_DIM = 45 * 1_710


def _rotl1(value):
    value &= 0xFFFFFFFF
    return ((value << 1) | (value >> 31)) & 0xFFFFFFFF


# Friend-side hashes of the C++ raw feature classes.  The no-DG KSDG3 hash is
# intentionally distinct even though its physical 12,672-row table is kept for
# this experiment.
HALFKA_FRIEND_HASH = 0x5F134CB8
HALFKA_HM1_FRIEND_HASH = 0x7F134CB8
KSDG3_DG_FRIEND_HASH = 0x596B2374
KSDG3_NODG_FRIEND_HASH = 0x4A6B2374


class HalfKAHM1KSDG3(FeatureBlock):
    def __init__(self):
        super().__init__(
            "HalfKA_HM1_KSDG3",
            HALFKA_HM1_FRIEND_HASH ^ _rotl1(KSDG3_DG_FRIEND_HASH),
            OrderedDict([("HalfKA_HM1_KSDG3", KSDG3_DIM + HALFKA_HM1_DG_DIM)]),
        )


class HalfKANoDGKSDG3NoDG(FeatureBlock):
    def __init__(self):
        super().__init__(
            "HalfKA_NoDG_KSDG3_NoDG",
            HALFKA_FRIEND_HASH ^ _rotl1(KSDG3_NODG_FRIEND_HASH),
            OrderedDict([("HalfKA_NoDG_KSDG3_NoDG", KSDG3_DIM + HALFKA_NODG_DIM)]),
        )


class HalfKAHM1NoDGKSDG3NoDG(FeatureBlock):
    def __init__(self):
        super().__init__(
            "HalfKA_HM1_NoDG_KSDG3_NoDG",
            HALFKA_HM1_FRIEND_HASH ^ _rotl1(KSDG3_NODG_FRIEND_HASH),
            OrderedDict([("HalfKA_HM1_NoDG_KSDG3_NoDG", KSDG3_DIM + HALFKA_HM1_NODG_DIM)]),
        )


def get_feature_block_clss():
    return [HalfKAHM1KSDG3, HalfKANoDGKSDG3NoDG,
            HalfKAHM1NoDGKSDG3NoDG]

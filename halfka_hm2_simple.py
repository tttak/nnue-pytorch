"""Canonical YaneuraOu HalfKA_hm2 feature block (merged golds).

The C++ training-data loader owns the authoritative index generation.  This
small Python declaration supplies the dimension/name/hash used by checkpoints
and serializers; it deliberately has no DISTINGUISH_GOLDS switch.
"""

from collections import OrderedDict

from feature_block import FeatureBlock


HALFKA_HM2_NODG_PLANES = 1629
KING_BUCKETS = 45
NUM_FEATURES = KING_BUCKETS * HALFKA_HM2_NODG_PLANES


class Features(FeatureBlock):
    def __init__(self):
        super().__init__(
            "HalfKA_HM2_NoDG",
            0x7F234CB8,
            OrderedDict((("HalfKA_HM2_NoDG", NUM_FEATURES),)),
        )

    def get_active_features(self, board):
        raise RuntimeError(
            "HalfKA_HM2_NoDG uses the authoritative C++ loader; "
            "pure-Python feature extraction is intentionally unsupported"
        )


def get_feature_block_clss():
    return [Features]

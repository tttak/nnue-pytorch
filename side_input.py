"""Optional dense/context inputs that are independent of the sparse FT.

The schema is deliberately small.  New inputs add one preset here, one loader
extractor, and one C++ extractor; fusion stays shared.
"""

from enum import Enum


SIDE_INPUT_SCHEMA_VERSION = 1


class SideInputType(str, Enum):
    NONE = "none"
    SAFE_ESCAPE = "safe_escape"


class SideInputFusion(str, Enum):
    L2_RESIDUAL = "l2_residual"


def normalize_side_input(value):
    return SideInputType(value).value


def normalize_side_input_fusion(value):
    return SideInputFusion(value).value


def input_dimensions(side_input_type):
    value = normalize_side_input(side_input_type)
    return 0 if value == SideInputType.NONE.value else 16


def architecture_suffix(side_input_type, side_input_dim, fusion):
    value = normalize_side_input(side_input_type)
    if value == SideInputType.NONE.value:
        return ""
    normalize_side_input_fusion(fusion)
    if value == SideInputType.SAFE_ESCAPE.value:
        return f"-SideSafe{int(side_input_dim)}"
    raise ValueError(value)

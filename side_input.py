"""Optional dense/context inputs that are independent of the sparse FT.

The schema is deliberately small.  New inputs add one preset here, one loader
extractor, and one C++ extractor; fusion stays shared.
"""

from enum import Enum


SIDE_INPUT_SCHEMA_VERSION = 1
MOBILITY_TACTICAL_SCHEMA_VERSION = 2
MOBILITY_TACTICAL_V2_SCHEMA_VERSION = 2
MOBILITY_TACTICAL_V2_SCALE = 8.0


class SideInputType(str, Enum):
    NONE = "none"
    SAFE_ESCAPE = "safe_escape"
    MOBILITY_TACTICAL_V1 = "mobility_tactical_v1"
    MOBILITY_TACTICAL_V2 = "mobility_tactical_v2"


class SideInputFusion(str, Enum):
    L2_RESIDUAL = "l2_residual"


def normalize_side_input(value):
    return SideInputType(value).value


def normalize_side_input_fusion(value):
    return SideInputFusion(value).value


def input_dimensions(side_input_type):
    value = normalize_side_input(side_input_type)
    if value == SideInputType.NONE.value:
        return 0
    if value == SideInputType.SAFE_ESCAPE.value:
        return 16
    if value in (SideInputType.MOBILITY_TACTICAL_V1.value,
                 SideInputType.MOBILITY_TACTICAL_V2.value):
        return 8
    raise ValueError(value)


def schema_version(side_input_type):
    value = normalize_side_input(side_input_type)
    if value == SideInputType.MOBILITY_TACTICAL_V1.value:
        return MOBILITY_TACTICAL_SCHEMA_VERSION
    if value == SideInputType.MOBILITY_TACTICAL_V2.value:
        return MOBILITY_TACTICAL_V2_SCHEMA_VERSION
    return SIDE_INPUT_SCHEMA_VERSION


def fixed_scale(side_input_type):
    value = normalize_side_input(side_input_type)
    return (MOBILITY_TACTICAL_V2_SCALE
            if value == SideInputType.MOBILITY_TACTICAL_V2.value else 1.0)


def architecture_suffix(side_input_type, side_input_dim, fusion):
    value = normalize_side_input(side_input_type)
    if value == SideInputType.NONE.value:
        return ""
    normalize_side_input_fusion(fusion)
    if value == SideInputType.SAFE_ESCAPE.value:
        return f"-SideSafe{int(side_input_dim)}"
    if value == SideInputType.MOBILITY_TACTICAL_V1.value:
        return "-SideMobTac8x128-v1"
    if value == SideInputType.MOBILITY_TACTICAL_V2.value:
        return "-SideMobTac8x128-S8-v2"
    raise ValueError(value)

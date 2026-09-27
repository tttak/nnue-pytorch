"""Experiment 125 R5 relation contract with a 32-wide latent accumulator."""

from simple_gs_local_pair64 import (  # Relation/index contract is identical.
    GS_LOCALPAIR64_CLASSES as GS_LOCALPAIR32_CLASSES,
    GS_LOCALPAIR64_FEATURES as GS_LOCALPAIR32_FEATURES,
    GS_LOCALPAIR64_INIT_TABLE_QUANTUM as GS_LOCALPAIR32_INIT_TABLE_QUANTUM,
    enumerate_indices as enumerate_gs_local_pair32,
    feature_index as gs_local_pair32_index,
    orient_square as orient_gs_local_pair32_piece,
)

GS_LOCALPAIR32_TYPE = "gs_local_pair32"
GS_LOCALPAIR32_SCHEMA_VERSION = 1
GS_LOCALPAIR32_MAPPING_VERSION = "r5_silver_goldlike_chebyshev2_hm2mirror_v1"
GS_LOCALPAIR32_WIDTH = 32
GS_LOCALPAIR32_TRANSFORMED = 32
GS_LOCALPAIR32_PROJECTION_OUTPUTS = 16

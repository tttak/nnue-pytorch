import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

import features
import model as M


TRACE_MAGIC = "NNUE_TRACE_V1"
CPP_FT_SCALE = 127.0
CPP_FM_SCALE = 2032.0
FM_DIM = 32
KSDG3_END = 12672
HALFKA_END = 214210
FM_NAMES = ("ih", "ik", "sh", "sk")
CPP_FM_DIFF_SCALES = (42273, 42273, 13107, 13107)
CPP_FM_ABS_SCALES = (16909, 16909, 5243, 5243)
CPP_FM_SHIFTS = (37, 37, 22, 22)
PAIR_DIM = 640
PAIR_WEIGHT_SCALE = 16384
PAIR_OUTPUT_SHIFT = 21
ROUTER_INPUT_DIM = 384
ROUTER_OUTPUT_DIM = 12


class TraceEntry:
    def __init__(self, type_name, values):
        self.type_name = type_name
        self.values = values


def _parse_integer(value):
    if value.lower().startswith("0x"):
        return int(value, 16)
    return int(value, 10)


def read_trace(path):
    entries = {}
    with Path(path).open("r", encoding="utf-8") as trace_file:
        magic = trace_file.readline().rstrip("\r\n")
        if magic != TRACE_MAGIC:
            raise ValueError(
                f"Unsupported trace format: expected {TRACE_MAGIC!r}, got {magic!r}"
            )

        for line_number, line in enumerate(trace_file, start=2):
            line = line.rstrip("\r\n")
            if not line:
                continue

            fields = line.split("\t")
            if len(fields) < 4:
                raise ValueError(f"Invalid TSV row at line {line_number}")

            name, type_name, count_text = fields[:3]
            if name in entries:
                raise ValueError(f"Duplicate trace entry {name!r}")

            count = int(count_text)
            raw_values = fields[3:]
            if len(raw_values) != count:
                raise ValueError(
                    f"Trace entry {name!r} declares {count} values, "
                    f"but contains {len(raw_values)}"
                )

            if type_name == "str":
                values = raw_values
            else:
                values = [_parse_integer(value) for value in raw_values]
            entries[name] = TraceEntry(type_name, values)

    return entries


def get_trace_string(entries, name):
    if name not in entries:
        raise KeyError(f"Missing trace entry: {name}")
    entry = entries[name]
    if entry.type_name != "str" or len(entry.values) != 1:
        raise ValueError(f"Trace entry {name!r} is not a scalar string")
    return entry.values[0]


def get_trace_integers(entries, name, expected_count=None):
    if name not in entries:
        raise KeyError(f"Missing trace entry: {name}")
    entry = entries[name]
    if entry.type_name == "str":
        raise ValueError(f"Trace entry {name!r} is not numeric")
    if expected_count is not None and len(entry.values) != expected_count:
        raise ValueError(
            f"Trace entry {name!r} has {len(entry.values)} values; "
            f"expected {expected_count}"
        )
    return entry.values


def validate_perspectives(entries):
    us_perspective = get_trace_string(entries, "meta.us_perspective")
    them_perspective = get_trace_string(entries, "meta.them_perspective")
    if us_perspective not in ("BLACK", "WHITE"):
        raise ValueError(f"Invalid us perspective: {us_perspective!r}")
    expected_them = "WHITE" if us_perspective == "BLACK" else "BLACK"
    if them_perspective != expected_them:
        raise ValueError(
            f"Inconsistent perspectives: us={us_perspective}, "
            f"them={them_perspective}"
        )
    return us_perspective, them_perspective


def validate_trace_mapping(entries):
    expected = {
        "meta.pytorch.white_indices_perspective": "BLACK",
        "meta.pytorch.black_indices_perspective": "WHITE",
        "meta.pytorch.t_w_v_w_perspective": "BLACK",
        "meta.pytorch.t_b_v_b_perspective": "WHITE",
        "meta.raw_abs_source": "us_perspective_only",
        "pair.bucket_source": "stack_index_for_nnue_material_value",
        "pair.main.output_order": "us_then_them",
        "router.input_layout": (
            "abs_centered[0:128],diff[128:256],main_us[256:384]"
        ),
        "fm.path.scope": "direct_fc_gate_before_lca",
        "fm.path.diff_gate_target": "main_path",
        "lca.scope": "selected_bucket_before_cross",
        "deep.scope": "cross_through_fc2_preblend",
        "deep.phase.bucket_source": "material_pair_bucket",
        "final.scope": "fc2_through_evaluate",
        "final.score_perspective": "side_to_move",
        "final.side_adjustment": "none_already_side_to_move",
        "final.tempo": "none",
    }
    for name, expected_value in expected.items():
        actual_value = get_trace_string(entries, name)
        if actual_value != expected_value:
            raise ValueError(
                f"Trace mapping {name!r} is {actual_value!r}; "
                f"expected {expected_value!r}"
            )


def read_active_indices(entries, perspective):
    indices = get_trace_integers(entries, f"feature.{perspective}.indices")
    count = get_trace_integers(
        entries, f"feature.{perspective}.count", expected_count=1
    )[0]
    if len(indices) != count:
        raise ValueError(
            f"feature.{perspective}.count={count}, but indices has "
            f"{len(indices)} elements"
        )
    if indices != sorted(indices):
        raise ValueError(f"feature.{perspective}.indices is not sorted")
    return indices


def expand_feature_indices(real_indices, feature_set):
    expanded = []
    for index in real_indices:
        if index < 0 or index >= feature_set.num_real_features:
            raise ValueError(
                f"C++ real feature index {index} is outside [0, "
                f"{feature_set.num_real_features})"
            )
        expanded.extend(feature_set.get_feature_factors(index))
    return expanded


def make_sparse_inputs(entries, feature_set, device):
    cpp_black_indices = read_active_indices(entries, "BLACK")
    cpp_white_indices = read_active_indices(entries, "WHITE")

    # C++ BLACK maps to the historical PyTorch "white" input, and C++
    # WHITE maps to the historical PyTorch "black" input.
    pytorch_white_active = expand_feature_indices(cpp_black_indices, feature_set)
    pytorch_black_active = expand_feature_indices(cpp_white_indices, feature_set)

    max_active = max(len(pytorch_white_active), len(pytorch_black_active), 1)
    white_indices = torch.full(
        (1, max_active), -1, dtype=torch.int32, device=device
    )
    black_indices = torch.full(
        (1, max_active), -1, dtype=torch.int32, device=device
    )
    white_values = torch.zeros(
        (1, max_active), dtype=torch.float32, device=device
    )
    black_values = torch.zeros(
        (1, max_active), dtype=torch.float32, device=device
    )

    if pytorch_white_active:
        count = len(pytorch_white_active)
        white_indices[0, :count] = torch.tensor(
            pytorch_white_active, dtype=torch.int32, device=device
        )
        white_values[0, :count] = 1.0
    if pytorch_black_active:
        count = len(pytorch_black_active)
        black_indices[0, :count] = torch.tensor(
            pytorch_black_active, dtype=torch.int32, device=device
        )
        black_values[0, :count] = 1.0

    us_perspective, _ = validate_perspectives(entries)
    us = torch.tensor(
        [[1.0 if us_perspective == "BLACK" else 0.0]],
        dtype=torch.float32,
        device=device,
    )
    them = 1.0 - us

    return {
        "us": us,
        "them": them,
        "white_indices": white_indices.contiguous(),
        "white_values": white_values.contiguous(),
        "black_indices": black_indices.contiguous(),
        "black_values": black_values.contiguous(),
        "cpp_black_count": len(cpp_black_indices),
        "cpp_white_count": len(cpp_white_indices),
        "pytorch_white_count": len(pytorch_white_active),
        "pytorch_black_count": len(pytorch_black_active),
    }


def process_fm(v_features, indices):
    halfka_mask = (indices >= KSDG3_END) & (indices < HALFKA_END)
    ksdg_mask = (indices < KSDG3_END) | (indices >= HALFKA_END)

    halfka_v = v_features * halfka_mask.unsqueeze(-1)
    ksdg_v = v_features * ksdg_mask.unsqueeze(-1)

    halfka_sum_v = torch.sum(halfka_v, dim=1)
    ksdg_sum_v = torch.sum(ksdg_v, dim=1)
    halfka_sum_v2 = torch.sum(halfka_v**2, dim=1)
    ksdg_sum_v2 = torch.sum(ksdg_v**2, dim=1)

    ih = 0.5 * (halfka_sum_v**2 - halfka_sum_v2)
    ik = 0.5 * (ksdg_sum_v**2 - ksdg_sum_v2)

    return {
        "halfka.sum_v": halfka_sum_v.squeeze(0),
        "halfka.sum_v2": halfka_sum_v2.squeeze(0),
        "ksdg.sum_v": ksdg_sum_v.squeeze(0),
        "ksdg.sum_v2": ksdg_sum_v2.squeeze(0),
        "ih": ih.squeeze(0),
        "ik": ik.squeeze(0),
        "sh": halfka_sum_v.squeeze(0),
        "sk": ksdg_sum_v.squeeze(0),
    }


def quantize_feature_transformer_like_serialize(nnue):
    """Apply the exact FT quantization expressions used by NNUEWriter."""
    layer = nnue.input
    with torch.no_grad():
        bias_quantized = (
            layer.bias.data.mul(nnue.quantized_one).round().to(torch.int16)
        )

        all_weight = M.coalesce_ft_weights(nnue, layer.weight.data)
        weight_quantized = (
            all_weight.mul(nnue.quantized_one).round().to(torch.int16)
        )
        del all_weight

        all_v = M.coalesce_ft_weights(nnue, layer.v.data)
        v_custom_scale = nnue.quantized_one * 16.0
        v_quantized = all_v.mul(v_custom_scale).round().to(torch.int16)
        del all_v

    return bias_quantized, weight_quantized, v_quantized


def make_12_bucket_pair_softmax(pair_weight_logits):
    """Reproduce NNUEWriter's raw-logit interpolation before softmax."""
    bucket_indices = torch.arange(
        12,
        device=pair_weight_logits.device,
        dtype=pair_weight_logits.dtype,
    ).view(12, 1, 1)
    pf = bucket_indices / 11.0
    p3 = pf * 3.0
    w0 = torch.clamp(1.0 - p3, min=0.0)
    w1 = torch.clamp(1.0 - torch.abs(p3 - 1.0), min=0.0)
    w2 = torch.clamp(1.0 - torch.abs(p3 - 2.0), min=0.0)
    w3 = torch.clamp(p3 - 2.0, min=0.0)

    bucket_logits = (
        w0 * pair_weight_logits[0]
        + w1 * pair_weight_logits[1]
        + w2 * pair_weight_logits[2]
        + w3 * pair_weight_logits[3]
    )
    return torch.softmax(bucket_logits, dim=2)


def quantize_pair_weights_like_serialize(nnue):
    with torch.no_grad():
        bucket_softmax = make_12_bucket_pair_softmax(nnue.pair_weights.data)
        bucket_quantized = (
            bucket_softmax.mul(PAIR_WEIGHT_SCALE).round().to(torch.int16)
        )

        # This is the same per-bucket/per-channel correction as NNUEWriter.
        for bucket in range(12):
            diffs = PAIR_WEIGHT_SCALE - bucket_quantized[bucket].sum(
                dim=1, dtype=torch.int32
            )
            for index in range(PAIR_DIM):
                if diffs[index] != 0:
                    max_index = torch.argmax(bucket_quantized[bucket, index])
                    bucket_quantized[bucket, index, max_index] += diffs[index].item()

    return bucket_quantized


def quantize_hidden_fc_like_serialize(nnue, layer):
    """Apply NNUEWriter.write_fc_layer(..., is_output=False)."""
    return quantize_hidden_fc_parameters_like_serialize(
        nnue, layer.bias.data, layer.weight.data
    )


def quantize_hidden_fc_parameters_like_serialize(nnue, bias, weight):
    """Quantize already-coalesced hidden FC parameters like NNUEWriter."""
    return quantize_fc_parameters_like_serialize(
        nnue, bias, weight, is_output=False
    )


def quantize_fc_parameters_like_serialize(
    nnue, bias, weight, is_output=False
):
    """Apply NNUEWriter.write_fc_layer() quantization to FC parameters."""
    if is_output:
        weight_scale = (
            nnue.nnue2score * nnue.weight_scale_out / nnue.quantized_one
        )
        bias_scale = nnue.weight_scale_out * nnue.nnue2score
    else:
        weight_scale = nnue.weight_scale_hidden
        bias_scale = nnue.weight_scale_hidden * nnue.quantized_one
    max_weight = nnue.quantized_one / weight_scale
    with torch.no_grad():
        bias_quantized = bias.mul(bias_scale).round().to(torch.int32)
        weight_quantized = (
            weight.clamp(-max_weight, max_weight)
            .mul(weight_scale)
            .round()
            .to(torch.int8)
        )
    return bias_quantized, weight_quantized


def quantize_router_like_serialize(nnue):
    return quantize_hidden_fc_like_serialize(nnue, nnue.layer_stacks.router)


def wrap_to_signed_int16(values):
    return torch.remainder(values + 32768, 65536) - 32768


def wrap_to_signed_int32(values):
    return torch.remainder(values + 2147483648, 4294967296) - 2147483648


def float32_bits(values):
    values_f32 = values.detach().cpu().to(torch.float32).contiguous().reshape(-1)
    return torch.bitwise_and(
        values_f32.view(torch.int32).to(torch.int64), 0xFFFFFFFF
    )


def float32_from_bits(raw_bits):
    bits = torch.tensor(raw_bits, dtype=torch.int64).reshape(-1)
    signed = torch.where(bits >= 0x80000000, bits - 0x100000000, bits)
    return signed.to(torch.int32).view(torch.float32)


def sigmoid_float32(values):
    values = values.to(torch.float32)
    return 1.0 / (1.0 + torch.exp(-values))


def make_integer_fm_reference(v_quantized, real_indices):
    index_tensor = torch.tensor(real_indices, dtype=torch.long)
    selected_v = v_quantized.index_select(0, index_tensor).to(torch.int64)
    ksdg_mask = index_tensor < KSDG3_END
    halfka_mask = ~ksdg_mask

    def accumulate(mask):
        values = selected_v[mask]
        return values.sum(dim=0), (values * values).sum(dim=0)

    halfka_sum_v, halfka_sum_v2 = accumulate(halfka_mask)
    ksdg_sum_v, ksdg_sum_v2 = accumulate(ksdg_mask)

    # C++ signed integer division truncates toward zero.
    ih = torch.div(
        halfka_sum_v * halfka_sum_v - halfka_sum_v2,
        2,
        rounding_mode="trunc",
    )
    ik = torch.div(
        ksdg_sum_v * ksdg_sum_v - ksdg_sum_v2,
        2,
        rounding_mode="trunc",
    )
    return {
        "halfka.sum_v": halfka_sum_v,
        "halfka.sum_v2": halfka_sum_v2,
        "ksdg.sum_v": ksdg_sum_v,
        "ksdg.sum_v2": ksdg_sum_v2,
        "ih": ih,
        "ik": ik,
        "sh": halfka_sum_v,
        "sk": ksdg_sum_v,
    }


def scale_integer_fm_path(raw_channels, scales):
    scaled_channels = []
    for channel, (scale, shift) in enumerate(zip(scales, CPP_FM_SHIFTS)):
        product = raw_channels[channel] * scale
        # clang++/AVX2 uses an arithmetic right shift for these signed values.
        shifted = torch.floor_divide(product, 1 << shift)
        scaled_channels.append(torch.clamp(shifted + 63, 0, 127))
    return torch.stack(scaled_channels, dim=0).reshape(-1).to(torch.int64)


def make_integer_reference(entries, nnue):
    cpp_indices = {
        "BLACK": read_active_indices(entries, "BLACK"),
        "WHITE": read_active_indices(entries, "WHITE"),
    }
    bias_quantized, weight_quantized, v_quantized = (
        quantize_feature_transformer_like_serialize(nnue)
    )
    pair_weights_quantized = quantize_pair_weights_like_serialize(nnue)
    router_bias_quantized, router_weight_quantized = (
        quantize_router_like_serialize(nnue)
    )
    pair_bucket = get_trace_integers(
        entries, "pair.bucket_id", expected_count=1
    )[0]
    if pair_bucket < 0 or pair_bucket >= 12:
        raise ValueError(f"Invalid pair bucket: {pair_bucket}")
    selected_pair_weights = pair_weights_quantized[pair_bucket].to(torch.int64)

    reference = {}
    reference["pair.weight.mul"] = selected_pair_weights[:, 0]
    reference["pair.weight.diff"] = selected_pair_weights[:, 1]
    reference["pair.weight.sum"] = selected_pair_weights[:, 2]
    fm_by_perspective = {}
    for perspective in ("BLACK", "WHITE"):
        index_tensor = torch.tensor(cpp_indices[perspective], dtype=torch.long)
        ft_sum = (
            bias_quantized.to(torch.int64)
            + weight_quantized.index_select(0, index_tensor)
            .to(torch.int64)
            .sum(dim=0)
        )
        reference[f"ft.main.{perspective}"] = wrap_to_signed_int16(ft_sum)

        a_values = torch.clamp(reference[f"ft.main.{perspective}"][:PAIR_DIM], 0, 127)
        b_values = torch.clamp(reference[f"ft.main.{perspective}"][PAIR_DIM:], 0, 127)
        mul_term = a_values * b_values
        diff_sq_term = (a_values - b_values) * (a_values - b_values)
        sum_term = (a_values + b_values) * 64
        mixed_numerator = wrap_to_signed_int32(
            mul_term * selected_pair_weights[:, 0]
            + diff_sq_term * selected_pair_weights[:, 1]
            + sum_term * selected_pair_weights[:, 2]
        )
        shifted = torch.floor_divide(mixed_numerator, 1 << PAIR_OUTPUT_SHIFT)
        packed_int16 = torch.clamp(shifted, -32768, 32767)
        main_pair_output = torch.clamp(packed_int16, 0, 255).to(torch.int64)

        pair_prefix = f"pair.main.{perspective}"
        reference[pair_prefix + ".a"] = a_values
        reference[pair_prefix + ".b"] = b_values
        reference[pair_prefix + ".mul_term"] = mul_term
        reference[pair_prefix + ".diff_sq_term"] = diff_sq_term
        reference[pair_prefix + ".sum_term"] = sum_term
        reference[pair_prefix + ".mixed_numerator"] = mixed_numerator
        reference[pair_prefix + ".output"] = main_pair_output

        fm_values = make_integer_fm_reference(
            v_quantized, cpp_indices[perspective]
        )
        fm_by_perspective[perspective] = fm_values
        for name in ("halfka.sum_v", "halfka.sum_v2", "ksdg.sum_v", "ksdg.sum_v2"):
            reference[f"fm.accumulator.{perspective}.{name}"] = fm_values[name]
        for name in FM_NAMES:
            reference[f"fm.interaction.{perspective}.{name}"] = fm_values[name]

    us_perspective, them_perspective = validate_perspectives(entries)
    raw_diff_channels = []
    raw_abs_channels = []
    for name in FM_NAMES:
        us_values = fm_by_perspective[us_perspective][name]
        them_values = fm_by_perspective[them_perspective][name]
        raw_diff = us_values - them_values
        raw_abs = us_values
        reference[f"fm.raw_diff.{name}"] = raw_diff
        reference[f"fm.raw_abs.{name}"] = raw_abs
        raw_diff_channels.append(raw_diff)
        raw_abs_channels.append(raw_abs)

    raw_diff_channels = torch.stack(raw_diff_channels, dim=0)
    raw_abs_channels = torch.stack(raw_abs_channels, dim=0)
    reference["fm.scaled_diff"] = scale_integer_fm_path(
        raw_diff_channels, CPP_FM_DIFF_SCALES
    )
    reference["fm.scaled_abs"] = scale_integer_fm_path(
        raw_abs_channels, CPP_FM_ABS_SCALES
    )

    router_abs_input = torch.clamp(
        (reference["fm.scaled_abs"] - 64) * 2, 0, 127
    )
    router_diff_input = reference["fm.scaled_diff"]
    router_main_input = reference[
        f"pair.main.{us_perspective}.output"
    ][:FM_DIM * 4]
    router_input = torch.cat(
        (router_abs_input, router_diff_input, router_main_input), dim=0
    ).to(torch.int64)
    router_logits = wrap_to_signed_int32(
        router_bias_quantized.to(torch.int64)
        + torch.matmul(router_weight_quantized.to(torch.int64), router_input)
    )
    reference["router.input"] = router_input
    reference["router.logits"] = router_logits
    reference["router.selected_bucket"] = torch.argmax(
        router_logits[:ROUTER_OUTPUT_DIM]
    ).reshape(1)

    fm_bucket = int(reference["router.selected_bucket"].item())
    if fm_bucket < 0 or fm_bucket >= ROUTER_OUTPUT_DIM:
        raise ValueError(f"Invalid FM path bucket: {fm_bucket}")
    fm_start = fm_bucket * 64
    fm_end = fm_start + 64
    diff_bias, diff_weight = quantize_hidden_fc_like_serialize(
        nnue,
        nnue.layer_stacks.fm_diff,
    )
    abs_bias, abs_weight = quantize_hidden_fc_like_serialize(
        nnue,
        nnue.layer_stacks.fm_abs,
    )
    diff_fc_preact = wrap_to_signed_int32(
        diff_bias[fm_start:fm_end].to(torch.int64)
        + torch.matmul(
            diff_weight[fm_start:fm_end].to(torch.int64),
            reference["fm.scaled_diff"],
        )
    )
    abs_fc_preact = wrap_to_signed_int32(
        abs_bias[fm_start:fm_end].to(torch.int64)
        + torch.matmul(
            abs_weight[fm_start:fm_end].to(torch.int64),
            reference["fm.scaled_abs"],
        )
    )

    diff_gate = diff_fc_preact[:FM_DIM]
    diff_value = diff_fc_preact[FM_DIM:]
    abs_gate = abs_fc_preact[:FM_DIM]
    abs_value = abs_fc_preact[FM_DIM:]

    diff_value_f32 = diff_value.to(torch.float32)
    diff_sum_sq = torch.zeros((), dtype=torch.float32)
    for value in diff_value_f32:
        diff_sum_sq = diff_sum_sq + value * value
    diff_inv_rms = 1.0 / torch.sqrt(diff_sum_sq / 32.0 + 1e-8)
    diff_normalized = diff_value_f32 * diff_inv_rms
    diff_centered = torch.trunc(diff_normalized * 25.4).to(torch.int64)
    diff_output = torch.clamp(diff_centered + 64, 0, 127)

    main_gate_sigmoid = sigmoid_float32(
        (diff_gate - 2438).to(torch.float32) / 8128.0
    )
    main_gate_q64 = torch.trunc(main_gate_sigmoid * 64.0).to(torch.int64)

    abs_gate_sigmoid = sigmoid_float32(abs_gate.to(torch.float32) / 8128.0)
    abs_gated = torch.trunc(
        abs_value.to(torch.float32) * abs_gate_sigmoid
    ).to(torch.int64)
    abs_scaled_before_round = torch.clamp(
        abs_gated.to(torch.float32) / 8128.0 * 0.05 + 0.6,
        0.0,
        1.0,
    ) * 127.0
    # All values are non-negative here; floor(x + 0.5) matches std::round().
    abs_output = torch.floor(abs_scaled_before_round + 0.5).to(torch.int64)
    abs_squared_output = torch.div(
        abs_output * abs_output, 127, rounding_mode="trunc"
    )

    reference["fm.path.selected_bucket"] = torch.tensor([fm_bucket])
    reference["fm.path.diff.input"] = reference["fm.scaled_diff"]
    reference["fm.path.abs.input"] = reference["fm.scaled_abs"]
    reference["fm.path.diff.fc_preact"] = diff_fc_preact
    reference["fm.path.abs.fc_preact"] = abs_fc_preact
    reference["fm.path.diff.gate_preact"] = diff_gate
    reference["fm.path.diff.value_preact"] = diff_value
    reference["fm.path.diff.rms_sum_sq_f32_bits"] = float32_bits(diff_sum_sq)
    reference["fm.path.diff.inv_rms_f32_bits"] = float32_bits(diff_inv_rms)
    reference["fm.path.diff.normalized_f32_bits"] = float32_bits(
        diff_normalized
    )
    reference["fm.path.diff.normalized_scaled_centered"] = diff_centered
    reference["fm.path.diff.output_pre_lca"] = diff_output
    reference["fm.path.diff.main_gate_q64"] = main_gate_q64
    reference["fm.path.diff.main_gate_multiplier_q128"] = 64 + main_gate_q64
    reference["fm.path.abs.gate_preact"] = abs_gate
    reference["fm.path.abs.value_preact"] = abs_value
    reference["fm.path.abs.gate_sigmoid_f32_bits"] = float32_bits(
        abs_gate_sigmoid
    )
    reference["fm.path.abs.gated_value"] = abs_gated
    reference["fm.path.abs.scaled_before_round_f32_bits"] = float32_bits(
        abs_scaled_before_round
    )
    reference["fm.path.abs.output"] = abs_output
    reference["fm.path.abs.squared_output"] = abs_squared_output

    main_start = fm_bucket * FM_DIM
    main_end = main_start + FM_DIM
    main_weight = (
        nnue.layer_stacks.l1.weight.data[main_start:main_end]
        + nnue.layer_stacks.l1_fact.weight.data
    )
    main_bias = (
        nnue.layer_stacks.l1.bias.data[main_start:main_end]
        + nnue.layer_stacks.l1_fact.bias.data
    )
    main_bias_quantized, main_weight_quantized = (
        quantize_hidden_fc_parameters_like_serialize(
            nnue, main_bias, main_weight
        )
    )
    main_input = torch.cat(
        (
            reference[f"pair.main.{us_perspective}.output"],
            reference[f"pair.main.{them_perspective}.output"],
        ),
        dim=0,
    )
    main_fc_before_gate = wrap_to_signed_int32(
        main_bias_quantized.to(torch.int64)
        + torch.matmul(main_weight_quantized.to(torch.int64), main_input)
    )
    main_gate_product = wrap_to_signed_int32(
        main_fc_before_gate.to(torch.int64)
        * reference["fm.path.diff.main_gate_multiplier_q128"]
    )
    main_fc_after_gate = torch.div(
        main_gate_product, 128, rounding_mode="trunc"
    )
    main_fc_after_gate = main_fc_after_gate.clone()
    main_fc_after_gate[:31] = torch.clamp(
        main_fc_after_gate[:31], 0, 8128
    )
    lca_query_input = torch.clamp(
        torch.floor_divide(main_fc_after_gate[:31], 64), 0, 127
    )
    lca_fm_input = torch.cat((diff_output, abs_output), dim=0)

    q_bias, q_weight = quantize_hidden_fc_like_serialize(
        nnue, nnue.layer_stacks.q_proj
    )
    k_bias, k_weight = quantize_hidden_fc_like_serialize(
        nnue, nnue.layer_stacks.k_proj
    )
    v_bias, v_weight = quantize_hidden_fc_like_serialize(
        nnue, nnue.layer_stacks.v_proj
    )
    lca_query = wrap_to_signed_int32(
        q_bias.to(torch.int64)
        + torch.matmul(q_weight.to(torch.int64), lca_query_input)
    )
    lca_key = wrap_to_signed_int32(
        k_bias.to(torch.int64)
        + torch.matmul(k_weight.to(torch.int64), lca_fm_input)
    )
    lca_value = wrap_to_signed_int32(
        v_bias.to(torch.int64)
        + torch.matmul(v_weight.to(torch.int64), lca_fm_input)
    )

    lca_qk_width = int(lca_query.numel())
    lca_value_width = int(lca_value.numel())
    lca_dot_product = torch.zeros((), dtype=torch.float32)
    lca_query_f32 = lca_query.to(torch.float32)
    lca_key_f32 = lca_key.to(torch.float32)
    for index in range(lca_qk_width):
        lca_dot_product = lca_dot_product + (
            (lca_query_f32[index] / 8128.0)
            * (lca_key_f32[index] / 8128.0)
        )
    lca_temperature = torch.clamp(
        nnue.layer_stacks.lca_temp.detach().cpu().to(torch.float32),
        min=0.125,
    )
    lca_attention_logit = (
        lca_dot_product * torch.tensor(0.17677, dtype=torch.float32)
    ) / lca_temperature
    lca_attention_score = sigmoid_float32(lca_attention_logit)
    # Compact LCA keeps Q/K/V projections in ranking order.  Q/K are consumed
    # directly at their compact width, while V is scattered back to the
    # original 32 Diff channels.  An omitted V preactivation is exactly zero,
    # hence its mapped value is 0.5, matching the Python model and C++ path.
    lca_value_full = torch.zeros(FM_DIM, dtype=lca_value.dtype)
    lca_value_indices = tuple(getattr(nnue, "lca_value_indices", range(FM_DIM)))
    lca_value_full[list(lca_value_indices)] = lca_value
    lca_value_clamped = torch.clamp(
        lca_value_full.to(torch.float32) / 8128.0 * 0.4 + 0.5,
        0.0,
        1.0,
    )
    lca_current_diff = diff_output.to(torch.float32) / 127.0
    lca_output_f32 = (
        lca_current_diff * (1.0 - lca_attention_score)
        + lca_value_clamped * lca_attention_score
    )
    lca_correction = lca_output_f32 - lca_current_diff
    lca_output = torch.trunc(lca_output_f32 * 127.0).to(torch.int64)

    reference["lca.selected_bucket"] = torch.tensor([fm_bucket])
    reference["lca.main_fc_preact_before_gate"] = main_fc_before_gate
    reference["lca.main_fc_preact_after_gate"] = main_fc_after_gate
    reference["lca.query_input"] = lca_query_input
    reference["lca.fm_input"] = lca_fm_input
    reference["lca.query_preact"] = F.pad(
        lca_query, (0, FM_DIM - lca_qk_width))
    reference["lca.key_preact"] = F.pad(
        lca_key, (0, FM_DIM - lca_qk_width))
    reference["lca.value_preact"] = F.pad(
        lca_value, (0, FM_DIM - lca_value_width))
    reference["lca.temperature_f32_bits"] = float32_bits(lca_temperature)
    reference["lca.dot_product_f32_bits"] = float32_bits(lca_dot_product)
    reference["lca.attention_logit_f32_bits"] = float32_bits(
        lca_attention_logit
    )
    reference["lca.attention_score_f32_bits"] = float32_bits(
        lca_attention_score
    )
    reference["lca.value_clamped_f32_bits"] = float32_bits(
        lca_value_clamped
    )
    reference["lca.correction_f32_bits"] = float32_bits(lca_correction)
    reference["lca.output_post_lca_f32_bits"] = float32_bits(
        lca_output_f32
    )
    reference["lca.output_post_lca"] = lca_output

    phase_abs_input = torch.clamp(
        (reference["fm.scaled_abs"] - 64) * 2, 0, 127
    )
    phase_abs_input = phase_abs_input.clone()
    phase_abs_input[127] = (pair_bucket * 127) // 11
    phase_input = torch.cat(
        (
            phase_abs_input,
            reference["fm.scaled_diff"],
            main_input[:128],
        ),
        dim=0,
    )
    phase_bias, phase_weight = quantize_hidden_fc_like_serialize(
        nnue, nnue.layer_stacks.phase_proj
    )
    phase_preact_all = wrap_to_signed_int32(
        phase_bias.to(torch.int64)
        + torch.matmul(phase_weight.to(torch.int64), phase_input)
    )
    phase_preact = phase_preact_all[:6]
    phase_logit = phase_preact.to(torch.float32) / 8128.0 * 3.0 + 1.0
    phase_sigmoid = sigmoid_float32(phase_logit)
    phase_value = 0.1 + 0.9 * phase_sigmoid
    phase_multipliers = torch.tensor(
        [1.3, 1.5, 1.0, 0.7, 0.88, 1.5], dtype=torch.float32
    )
    channel_scales = (0.5 + 0.5 * phase_value) * phase_multipliers

    main_raw = torch.clamp(
        torch.floor_divide(main_fc_after_gate[:31], 64), 0, 127
    )
    main_squared = torch.clamp(
        torch.floor_divide(
            main_fc_after_gate[:31] * main_fc_after_gate[:31],
            1 << 19,
        ),
        0,
        127,
    )
    cross_main_squared = main_squared[:16]
    cross_diff = lca_output[:16]
    cross_main_raw = main_raw[:16]
    cross_abs = abs_output[:16]
    cross_product_diff = torch.div(
        cross_main_squared * cross_diff, 127, rounding_mode="trunc"
    )
    cross_product_abs = torch.div(
        cross_main_raw * cross_abs, 127, rounding_mode="trunc"
    )
    cross_input = torch.cat(
        (cross_product_diff, cross_product_abs), dim=0
    )

    cross_start = fm_bucket * FM_DIM
    cross_end = cross_start + FM_DIM
    cross_bias, cross_weight = quantize_hidden_fc_parameters_like_serialize(
        nnue,
        nnue.layer_stacks.cross_proj.bias.data[cross_start:cross_end],
        nnue.layer_stacks.cross_proj.weight.data[cross_start:cross_end],
    )
    cross_preact = wrap_to_signed_int32(
        cross_bias.to(torch.int64)
        + torch.matmul(cross_weight.to(torch.int64), cross_input)
    )
    cross_output = torch.clamp(
        torch.floor_divide(cross_preact, 64), 0, 127
    )

    def scale_channel(values, scale):
        return torch.clamp(
            torch.trunc(values.to(torch.float32) * scale).to(torch.int64),
            0,
            127,
        )

    fc1_parts = [
        scale_channel(main_squared, channel_scales[0]),
        scale_channel(main_raw, channel_scales[1]),
        scale_channel(lca_output, channel_scales[2]),
        scale_channel(abs_output, channel_scales[3]),
    ]
    if not nnue.remove_abs_sqr_l2:
        fc1_parts.append(scale_channel(abs_squared_output, channel_scales[4]))
    fc1_parts.extend((
        scale_channel(cross_output, channel_scales[5]),
        torch.zeros(2, dtype=torch.int64),
    ))
    fc1_input = torch.cat(tuple(fc1_parts), dim=0)

    fc1_start = fm_bucket * 96
    fc1_end = fc1_start + 96
    fc1_bias, fc1_weight = quantize_hidden_fc_parameters_like_serialize(
        nnue,
        nnue.layer_stacks.l2.bias.data[fc1_start:fc1_end],
        nnue.layer_stacks.l2.weight.data[
            fc1_start:fc1_end, :nnue.layer_stacks.l2_in_total
        ],
    )
    fc1_preact = wrap_to_signed_int32(
        fc1_bias.to(torch.int64)
        + torch.matmul(fc1_weight.to(torch.int64), fc1_input)
    )
    fc1_output = torch.clamp(
        torch.floor_divide(fc1_preact, 64), 0, 127
    )

    fc2_bias, fc2_weight = quantize_fc_parameters_like_serialize(
        nnue,
        nnue.layer_stacks.output.bias.data[fm_bucket:fm_bucket + 1],
        nnue.layer_stacks.output.weight.data[fm_bucket:fm_bucket + 1],
        is_output=True,
    )
    fc2_preact = wrap_to_signed_int32(
        fc2_bias.to(torch.int64)
        + torch.matmul(fc2_weight.to(torch.int64), fc1_output)
    )

    reference["deep.selected_bucket"] = torch.tensor([fm_bucket])
    reference["deep.phase.bucket_id"] = torch.tensor([pair_bucket])
    reference["deep.scale.activation"] = torch.tensor([127])
    reference["deep.scale.hidden_preact"] = torch.tensor([8128])
    reference["deep.scale.fc2_preact"] = torch.tensor([
        int(nnue.weight_scale_out * nnue.nnue2score)
    ])
    reference["deep.phase.input"] = phase_input
    reference["deep.phase.preact"] = phase_preact
    reference["deep.phase.logit_f32_bits"] = float32_bits(phase_logit)
    reference["deep.phase.sigmoid_f32_bits"] = float32_bits(phase_sigmoid)
    reference["deep.phase.value_f32_bits"] = float32_bits(phase_value)
    reference["deep.phase.channel_scale_f32_bits"] = float32_bits(
        channel_scales
    )
    reference["deep.main.raw"] = main_raw
    reference["deep.main.squared"] = main_squared
    reference["deep.cross.main_squared"] = cross_main_squared
    reference["deep.cross.diff"] = cross_diff
    reference["deep.cross.main_raw"] = cross_main_raw
    reference["deep.cross.abs"] = cross_abs
    reference["deep.cross.product_diff"] = cross_product_diff
    reference["deep.cross.product_abs"] = cross_product_abs
    reference["deep.cross.input"] = cross_input
    reference["deep.cross.preact"] = cross_preact
    reference["deep.cross.output"] = cross_output
    reference["deep.fc1.input"] = fc1_input
    reference["deep.fc1.preact"] = fc1_preact
    reference["deep.fc1.output"] = fc1_output
    reference["deep.fc2.preact"] = fc2_preact
    reference["deep.fc2.output_preblend"] = fc2_preact

    output_scale = int(nnue.weight_scale_out * nnue.nnue2score)
    hidden_scale = int(nnue.weight_scale_hidden * nnue.quantized_one)
    alpha_float = torch.sigmoid(torch.tensor(
        nnue.layer_stacks.blend.data[fm_bucket].item()
    ))
    alpha_q14 = int(alpha_float.item() * 16384)
    inv_alpha_q14 = 16384 - alpha_q14
    bypass_input = main_fc_after_gate[31:32]
    bypass_scaled_numerator = wrap_to_signed_int32(
        bypass_input.to(torch.int64) * output_scale
    )
    bypass_output = torch.div(
        bypass_scaled_numerator, hidden_scale, rounding_mode="trunc"
    )
    deep_term = fc2_preact.to(torch.int64) * alpha_q14
    bypass_term = bypass_output.to(torch.int64) * inv_alpha_q14
    blend_numerator = deep_term + bypass_term
    blend_output = torch.div(
        blend_numerator, 16384, rounding_mode="trunc"
    ).to(torch.int64)
    fv_scale = get_trace_integers(
        entries, "final.fv_scale", expected_count=1
    )[0]
    if fv_scale <= 0:
        raise ValueError(f"Invalid C++ FV_SCALE: {fv_scale}")
    value_max_eval = get_trace_integers(
        entries, "final.value_max_eval", expected_count=1
    )[0]
    eval_before_clamp = torch.div(
        blend_output, fv_scale, rounding_mode="trunc"
    )
    eval_after_clamp = torch.clamp(
        eval_before_clamp, -value_max_eval, value_max_eval
    )

    reference["final.selected_bucket"] = torch.tensor([fm_bucket])
    reference["final.material_bucket"] = torch.tensor([pair_bucket])
    reference["final.scale.deep_output"] = torch.tensor([output_scale])
    reference["final.scale.bypass_input"] = torch.tensor([hidden_scale])
    reference["final.scale.bypass_output"] = torch.tensor([output_scale])
    reference["final.scale.blend_alpha"] = torch.tensor([16384])
    reference["final.scale.blend_numerator"] = torch.tensor([
        output_scale * 16384
    ])
    reference["final.scale.network_output"] = torch.tensor([output_scale])
    reference["final.scale.eval_value"] = torch.tensor([1])
    reference["final.scale.pytorch_nnue2score"] = torch.tensor([
        int(nnue.nnue2score)
    ])
    reference["final.deep_output"] = fc2_preact
    reference["final.bypass.input"] = bypass_input
    reference["final.bypass.preact"] = bypass_input
    reference["final.bypass.scaled_numerator"] = bypass_scaled_numerator
    reference["final.bypass.output"] = bypass_output
    reference["final.blend.parameter_raw"] = torch.tensor([alpha_q14])
    reference["final.blend.alpha_q14"] = torch.tensor([alpha_q14])
    reference["final.blend.inv_alpha_q14"] = torch.tensor([inv_alpha_q14])
    reference["final.blend.deep_term"] = deep_term
    reference["final.blend.bypass_term"] = bypass_term
    reference["final.blend.numerator"] = blend_numerator
    reference["final.blend.output"] = blend_output
    reference["final.network_output"] = blend_output
    reference["final.fv_scale"] = torch.tensor([fv_scale])
    reference["final.eval_before_clamp"] = eval_before_clamp
    reference["final.value_max_eval"] = torch.tensor([value_max_eval])
    reference["final.eval_after_clamp"] = eval_after_clamp

    del bias_quantized
    del weight_quantized
    del v_quantized
    del pair_weights_quantized
    del router_bias_quantized
    del router_weight_quantized
    del diff_bias
    del diff_weight
    del abs_bias
    del abs_weight
    del main_bias_quantized
    del main_weight_quantized
    del q_bias
    del q_weight
    del k_bias
    del k_weight
    del v_bias
    del v_weight
    del phase_bias
    del phase_weight
    del cross_bias
    del cross_weight
    del fc1_bias
    del fc1_weight
    del fc2_bias
    del fc2_weight
    return reference


def report_integer_comparison(label, cpp_values, reference_values):
    cpp = torch.tensor(cpp_values, dtype=torch.int64).reshape(-1)
    reference = reference_values.detach().cpu().to(torch.int64).reshape(-1)
    if cpp.numel() != reference.numel():
        raise ValueError(
            f"{label}: C++ has {cpp.numel()} elements, "
            f"Python reference has {reference.numel()}"
        )

    integer_diff = reference - cpp
    absolute_diff = torch.abs(integer_diff)
    differing_indices = torch.nonzero(integer_diff, as_tuple=False).flatten()
    differing_count = int(differing_indices.numel())
    max_integer_diff = int(absolute_diff.max().item())

    print(f"  {label}")
    print(f"    differing elements      : {differing_count}")
    print(f"    max integer diff        : {max_integer_diff}")
    if differing_count == 0:
        print("    first differing index   : none")
        print("    C++ raw                 : n/a")
        print("    Python integer reference: n/a")
        return

    first_index = int(differing_indices[0].item())
    print(f"    first differing index   : {first_index}")
    if cpp.numel() == 4 * FM_DIM:
        print(
            f"    channel/index           : {FM_NAMES[first_index // FM_DIM]} / "
            f"{first_index % FM_DIM}"
        )
    print(f"    C++ raw                 : {int(cpp[first_index].item())}")
    print(
        "    Python integer reference: "
        f"{int(reference[first_index].item())}"
    )


def compare_integer_reference(entries, reference):
    print("[C++ integer reference]")
    comparison_names = [
        "ft.main.BLACK",
        "ft.main.WHITE",
        "pair.weight.mul",
        "pair.weight.diff",
        "pair.weight.sum",
    ]
    for perspective in ("BLACK", "WHITE"):
        comparison_names.extend(
            f"pair.main.{perspective}.{name}"
            for name in (
                "a",
                "b",
                "mul_term",
                "diff_sq_term",
                "sum_term",
                "mixed_numerator",
                "output",
            )
        )
    for perspective in ("BLACK", "WHITE"):
        comparison_names.extend(
            f"fm.accumulator.{perspective}.{name}"
            for name in (
                "halfka.sum_v",
                "halfka.sum_v2",
                "ksdg.sum_v",
                "ksdg.sum_v2",
            )
        )
        comparison_names.extend(
            f"fm.interaction.{perspective}.{name}" for name in FM_NAMES
        )
    comparison_names.extend(
        f"fm.raw_diff.{name}" for name in FM_NAMES
    )
    comparison_names.extend(
        f"fm.raw_abs.{name}" for name in FM_NAMES
    )
    comparison_names.extend(
        (
            "fm.scaled_diff",
            "fm.scaled_abs",
            "router.input",
            "router.logits",
            "router.selected_bucket",
            "fm.path.selected_bucket",
            "fm.path.diff.input",
            "fm.path.abs.input",
            "fm.path.diff.fc_preact",
            "fm.path.abs.fc_preact",
            "fm.path.diff.gate_preact",
            "fm.path.diff.value_preact",
            "fm.path.diff.rms_sum_sq_f32_bits",
            "fm.path.diff.inv_rms_f32_bits",
            "fm.path.diff.normalized_f32_bits",
            "fm.path.diff.normalized_scaled_centered",
            "fm.path.diff.output_pre_lca",
            "fm.path.diff.main_gate_q64",
            "fm.path.diff.main_gate_multiplier_q128",
            "fm.path.abs.gate_preact",
            "fm.path.abs.value_preact",
            "fm.path.abs.gate_sigmoid_f32_bits",
            "fm.path.abs.gated_value",
            "fm.path.abs.scaled_before_round_f32_bits",
            "fm.path.abs.output",
            "fm.path.abs.squared_output",
            "lca.selected_bucket",
            "lca.main_fc_preact_before_gate",
            "lca.main_fc_preact_after_gate",
            "lca.query_input",
            "lca.fm_input",
            "lca.query_preact",
            "lca.key_preact",
            "lca.value_preact",
            "lca.temperature_f32_bits",
            "lca.dot_product_f32_bits",
            "lca.attention_logit_f32_bits",
            "lca.attention_score_f32_bits",
            "lca.value_clamped_f32_bits",
            "lca.correction_f32_bits",
            "lca.output_post_lca_f32_bits",
            "lca.output_post_lca",
            "deep.selected_bucket",
            "deep.phase.bucket_id",
            "deep.scale.activation",
            "deep.scale.hidden_preact",
            "deep.scale.fc2_preact",
            "deep.phase.input",
            "deep.phase.preact",
            "deep.phase.logit_f32_bits",
            "deep.phase.sigmoid_f32_bits",
            "deep.phase.value_f32_bits",
            "deep.phase.channel_scale_f32_bits",
            "deep.main.raw",
            "deep.main.squared",
            "deep.cross.main_squared",
            "deep.cross.diff",
            "deep.cross.main_raw",
            "deep.cross.abs",
            "deep.cross.product_diff",
            "deep.cross.product_abs",
            "deep.cross.input",
            "deep.cross.preact",
            "deep.cross.output",
            "deep.fc1.input",
            "deep.fc1.preact",
            "deep.fc1.output",
            "deep.fc2.preact",
            "deep.fc2.output_preblend",
            "final.selected_bucket",
            "final.material_bucket",
            "final.scale.deep_output",
            "final.scale.bypass_input",
            "final.scale.bypass_output",
            "final.scale.blend_alpha",
            "final.scale.blend_numerator",
            "final.scale.network_output",
            "final.scale.eval_value",
            "final.scale.pytorch_nnue2score",
            "final.deep_output",
            "final.bypass.input",
            "final.bypass.preact",
            "final.bypass.scaled_numerator",
            "final.bypass.output",
            "final.blend.parameter_raw",
            "final.blend.alpha_q14",
            "final.blend.inv_alpha_q14",
            "final.blend.deep_term",
            "final.blend.bypass_term",
            "final.blend.numerator",
            "final.blend.output",
            "final.network_output",
            "final.fv_scale",
            "final.eval_before_clamp",
            "final.value_max_eval",
            "final.eval_after_clamp",
        )
    )

    for name in comparison_names:
        report_integer_comparison(
            name,
            get_trace_integers(entries, name),
            reference[name],
        )


def report_comparison(label, cpp_raw_values, cpp_scale, pytorch_values):
    cpp_raw = torch.tensor(cpp_raw_values, dtype=torch.int64)
    cpp_float = cpp_raw.to(torch.float64) / float(cpp_scale)
    pytorch_float = pytorch_values.detach().cpu().to(torch.float64).reshape(-1)
    if cpp_float.numel() != pytorch_float.numel():
        raise ValueError(
            f"{label}: C++ has {cpp_float.numel()} elements, "
            f"PyTorch has {pytorch_float.numel()}"
        )

    absolute_diff = torch.abs(cpp_float - pytorch_float)
    max_index = int(torch.argmax(absolute_diff).item())
    print(f"  {label}")
    print(f"    max abs diff : {absolute_diff[max_index].item():.9e}")
    print(f"    mean abs diff: {absolute_diff.mean().item():.9e}")
    print(f"    max diff index: {max_index}")
    if cpp_float.numel() == 4 * FM_DIM:
        print(
            f"    channel/index: {FM_NAMES[max_index // FM_DIM]} / "
            f"{max_index % FM_DIM}"
        )
    print(f"    C++ raw int  : {int(cpp_raw[max_index].item())}")
    print(f"    C++ / {cpp_scale:g} : {cpp_float[max_index].item():.9e}")
    print(f"    PyTorch float: {pytorch_float[max_index].item():.9e}")


def report_float_bits_comparison(
    label, cpp_raw_bits, pytorch_values, cpp_multiplier=1.0
):
    cpp_bits = torch.tensor(cpp_raw_bits, dtype=torch.int64).reshape(-1)
    cpp_float = (
        float32_from_bits(cpp_raw_bits).to(torch.float64)
        * float(cpp_multiplier)
    )
    pytorch_float = pytorch_values.detach().cpu().to(torch.float64).reshape(-1)
    if cpp_float.numel() != pytorch_float.numel():
        raise ValueError(
            f"{label}: C++ has {cpp_float.numel()} elements, "
            f"PyTorch has {pytorch_float.numel()}"
        )
    absolute_diff = torch.abs(cpp_float - pytorch_float)
    max_index = int(torch.argmax(absolute_diff).item())
    print(f"  {label}")
    print(f"    max abs diff : {absolute_diff[max_index].item():.9e}")
    print(f"    mean abs diff: {absolute_diff.mean().item():.9e}")
    print(f"    max diff index: {max_index}")
    print(f"    C++ raw bits : 0x{int(cpp_bits[max_index].item()):08x}")
    print(f"    C++ scaled   : {cpp_float[max_index].item():.9e}")
    print(f"    PyTorch float: {pytorch_float[max_index].item():.9e}")


def compare_main_ft(entries, t_w, t_b):
    print("[Main FT]")
    report_comparison(
        "C++ BLACK / 127 vs PyTorch t_w",
        get_trace_integers(entries, "ft.main.BLACK", M.L1_MAIN),
        CPP_FT_SCALE,
        t_w.squeeze(0),
    )
    report_comparison(
        "C++ WHITE / 127 vs PyTorch t_b",
        get_trace_integers(entries, "ft.main.WHITE", M.L1_MAIN),
        CPP_FT_SCALE,
        t_b.squeeze(0),
    )


def compare_native_pair_main(entries, nnue, t_w, t_b):
    pair_bucket = get_trace_integers(
        entries, "pair.bucket_id", expected_count=1
    )[0]
    native_weights = make_12_bucket_pair_softmax(nnue.pair_weights)[pair_bucket]

    print("[PairWeight / Main Path native float]")
    for term_index, term_name in enumerate(("mul", "diff", "sum")):
        report_comparison(
            f"pair weight {term_name}",
            get_trace_integers(entries, f"pair.weight.{term_name}", PAIR_DIM),
            PAIR_WEIGHT_SCALE,
            native_weights[:, term_index],
        )

    outputs = {}
    for perspective, transformed in (("BLACK", t_w), ("WHITE", t_b)):
        clipped = torch.clamp(transformed.squeeze(0), 0.0, 1.0)
        a_values = clipped[:PAIR_DIM]
        b_values = clipped[PAIR_DIM:]
        mul_term = a_values * b_values
        diff_sq_term = torch.pow(a_values - b_values, 2)
        sum_term = (a_values + b_values) * 0.5
        output = (
            native_weights[:, 0] * mul_term
            + native_weights[:, 1] * diff_sq_term
            + native_weights[:, 2] * sum_term
        ) * (127.0 / 128.0)
        outputs[perspective] = output

        report_comparison(
            f"C++ {perspective} main pair / 127 vs PyTorch float",
            get_trace_integers(
                entries, f"pair.main.{perspective}.output", PAIR_DIM
            ),
            CPP_FT_SCALE,
            output,
        )
    return outputs


def compare_fm_accumulators(entries, fm_by_perspective):
    print("[FM accumulator]")
    scales = {
        "halfka.sum_v": CPP_FM_SCALE,
        "halfka.sum_v2": CPP_FM_SCALE**2,
        "ksdg.sum_v": CPP_FM_SCALE,
        "ksdg.sum_v2": CPP_FM_SCALE**2,
    }
    for perspective in ("BLACK", "WHITE"):
        for name, scale in scales.items():
            report_comparison(
                f"{perspective} {name}",
                get_trace_integers(
                    entries,
                    f"fm.accumulator.{perspective}.{name}",
                    FM_DIM,
                ),
                scale,
                fm_by_perspective[perspective][name],
            )


def compare_fm_interactions(entries, fm_by_perspective):
    print("[FM interaction]")
    for perspective in ("BLACK", "WHITE"):
        for name in FM_NAMES:
            scale = CPP_FM_SCALE**2 if name in ("ih", "ik") else CPP_FM_SCALE
            report_comparison(
                f"{perspective} {name}",
                get_trace_integers(
                    entries,
                    f"fm.interaction.{perspective}.{name}",
                    FM_DIM,
                ),
                scale,
                fm_by_perspective[perspective][name],
            )


def make_raw_paths(inputs, fm_by_perspective):
    v_w_all = torch.stack(
        [fm_by_perspective["BLACK"][name] for name in FM_NAMES], dim=0
    ).unsqueeze(0)
    v_b_all = torch.stack(
        [fm_by_perspective["WHITE"][name] for name in FM_NAMES], dim=0
    ).unsqueeze(0)

    us_3d = inputs["us"].view(-1, 1, 1)
    them_3d = inputs["them"].view(-1, 1, 1)
    raw_diff = (
        us_3d * (v_w_all - v_b_all)
        + them_3d * (v_b_all - v_w_all)
    )
    raw_abs = us_3d * v_w_all + them_3d * v_b_all
    return raw_diff, raw_abs


def compare_raw_paths(entries, raw_diff, raw_abs):
    print("[FM raw diff / abs]")
    for path_name, pytorch_values in (("raw_diff", raw_diff), ("raw_abs", raw_abs)):
        for channel, name in enumerate(FM_NAMES):
            scale = CPP_FM_SCALE**2 if name in ("ih", "ik") else CPP_FM_SCALE
            report_comparison(
                f"{path_name} {name}",
                get_trace_integers(
                    entries, f"fm.{path_name}.{name}", FM_DIM
                ),
                scale,
                pytorch_values[0, channel],
            )


def make_native_scaled_paths(raw_diff, raw_abs):
    norm_diff = torch.tensor(
        [0.01, 0.01, 0.05, 0.05],
        dtype=raw_diff.dtype,
        device=raw_diff.device,
    ).view(1, 4, 1)
    norm_abs = torch.tensor(
        [0.004, 0.004, 0.02, 0.02],
        dtype=raw_abs.dtype,
        device=raw_abs.device,
    ).view(1, 4, 1)

    diff_input_scaled = torch.clamp(
        raw_diff * norm_diff + 0.5, 0.0, 1.0
    ).reshape(-1)
    abs_input_scaled = torch.clamp(
        raw_abs * norm_abs + 0.5, 0.0, 1.0
    ).reshape(-1)
    return diff_input_scaled, abs_input_scaled


def compare_scaled_paths(entries, raw_diff, raw_abs):
    diff_input_scaled, abs_input_scaled = make_native_scaled_paths(
        raw_diff, raw_abs
    )

    print("[FM scaled diff / abs]")
    report_comparison(
        "scaled diff",
        get_trace_integers(entries, "fm.scaled_diff", 4 * FM_DIM),
        CPP_FT_SCALE,
        diff_input_scaled,
    )
    report_comparison(
        "scaled abs",
        get_trace_integers(entries, "fm.scaled_abs", 4 * FM_DIM),
        CPP_FT_SCALE,
        abs_input_scaled,
    )


def compare_native_router(
    entries, nnue, raw_diff, raw_abs, main_pair_outputs, us_perspective
):
    diff_input, abs_input = make_native_scaled_paths(raw_diff, raw_abs)
    abs_centered = torch.clamp(abs_input - 0.5, 0.0, 1.0) * 2.0
    main_us = main_pair_outputs[us_perspective][:FM_DIM * 4]
    router_input = torch.cat((abs_centered, diff_input, main_us), dim=0)
    router_logits = nnue.layer_stacks.router(router_input.unsqueeze(0)).squeeze(0)
    native_selected_bucket = int(torch.argmax(router_logits).item())
    cpp_selected_bucket = get_trace_integers(
        entries, "router.selected_bucket", expected_count=1
    )[0]

    print("[Router native PyTorch float]")
    report_comparison(
        "C++ router input / 127 vs PyTorch float",
        get_trace_integers(entries, "router.input", ROUTER_INPUT_DIM),
        CPP_FT_SCALE,
        router_input,
    )
    report_comparison(
        "C++ router logits / 8128 vs PyTorch float",
        get_trace_integers(entries, "router.logits", ROUTER_OUTPUT_DIM),
        nnue.weight_scale_hidden * nnue.quantized_one,
        router_logits,
    )
    print(f"  C++ selected bucket   : {cpp_selected_bucket}")
    print(f"  PyTorch selected bucket: {native_selected_bucket}")
    probabilities = torch.softmax(router_logits, dim=0).detach().cpu()
    print(
        "  PyTorch softmax probs : ["
        + ", ".join(f"{value.item():.6f}" for value in probabilities)
        + "]"
    )


def compare_native_fm_path(entries, nnue, raw_diff, raw_abs):
    bucket = get_trace_integers(
        entries, "fm.path.selected_bucket", expected_count=1
    )[0]
    diff_input, abs_input = make_native_scaled_paths(raw_diff, raw_abs)
    start = bucket * 64
    end = start + 64

    diff_fc = F.linear(
        diff_input.unsqueeze(0),
        nnue.layer_stacks.fm_diff.weight[start:end],
        nnue.layer_stacks.fm_diff.bias[start:end],
    ).squeeze(0)
    abs_fc = F.linear(
        abs_input.unsqueeze(0),
        nnue.layer_stacks.fm_abs.weight[start:end],
        nnue.layer_stacks.fm_abs.bias[start:end],
    ).squeeze(0)

    diff_gate, diff_value = diff_fc[:FM_DIM], diff_fc[FM_DIM:]
    abs_gate, abs_value = abs_fc[:FM_DIM], abs_fc[FM_DIM:]
    diff_sum_sq = diff_value.pow(2).sum()
    diff_inv_rms = torch.rsqrt(diff_sum_sq / FM_DIM + 1e-8)
    diff_normalized = diff_value * diff_inv_rms
    diff_output = torch.clamp(diff_normalized * 0.2 + 0.5, 0.0, 1.0)
    main_gate = torch.sigmoid(diff_gate - 0.3)
    main_gate_multiplier = 0.5 + 0.5 * main_gate
    abs_gate_sigmoid = torch.sigmoid(abs_gate)
    abs_gated = abs_value * abs_gate_sigmoid
    abs_output = torch.clamp(abs_gated * 0.05 + 0.6, 0.0, 1.0)
    abs_squared_output = abs_output.pow(2.0)

    print("[FM FC / Gate native PyTorch float]")
    print(f"  selected bucket: {bucket}")
    report_comparison(
        "Diff input: C++ / 127 vs PyTorch",
        get_trace_integers(entries, "fm.path.diff.input", 4 * FM_DIM),
        CPP_FT_SCALE,
        diff_input,
    )
    report_comparison(
        "Abs input: C++ / 127 vs PyTorch",
        get_trace_integers(entries, "fm.path.abs.input", 4 * FM_DIM),
        CPP_FT_SCALE,
        abs_input,
    )
    report_comparison(
        "Diff FC pre-activation: C++ / 8128 vs PyTorch",
        get_trace_integers(entries, "fm.path.diff.fc_preact", 2 * FM_DIM),
        nnue.weight_scale_hidden * nnue.quantized_one,
        diff_fc,
    )
    report_comparison(
        "Abs FC pre-activation: C++ / 8128 vs PyTorch",
        get_trace_integers(entries, "fm.path.abs.fc_preact", 2 * FM_DIM),
        nnue.weight_scale_hidden * nnue.quantized_one,
        abs_fc,
    )
    report_comparison(
        "Diff gate pre-activation: C++ / 8128 vs PyTorch",
        get_trace_integers(entries, "fm.path.diff.gate_preact", FM_DIM),
        nnue.weight_scale_hidden * nnue.quantized_one,
        diff_gate,
    )
    report_comparison(
        "Diff value before RMSNorm: C++ / 8128 vs PyTorch",
        get_trace_integers(entries, "fm.path.diff.value_preact", FM_DIM),
        nnue.weight_scale_hidden * nnue.quantized_one,
        diff_value,
    )
    report_float_bits_comparison(
        "Diff RMS sum of squares: C++ / 8128^2 vs PyTorch",
        get_trace_integers(
            entries, "fm.path.diff.rms_sum_sq_f32_bits", 1
        ),
        diff_sum_sq.reshape(1),
        cpp_multiplier=1.0
        / (nnue.weight_scale_hidden * nnue.quantized_one) ** 2,
    )
    report_float_bits_comparison(
        "Diff inverse RMS: C++ * 8128 vs PyTorch",
        get_trace_integers(entries, "fm.path.diff.inv_rms_f32_bits", 1),
        diff_inv_rms.reshape(1),
        cpp_multiplier=nnue.weight_scale_hidden * nnue.quantized_one,
    )
    report_float_bits_comparison(
        "Diff after RMSNorm",
        get_trace_integers(
            entries, "fm.path.diff.normalized_f32_bits", FM_DIM
        ),
        diff_normalized,
    )
    report_comparison(
        "Diff centered mapping before +64: C++ / 25.4 vs PyTorch RMSNorm",
        get_trace_integers(
            entries, "fm.path.diff.normalized_scaled_centered", FM_DIM
        ),
        25.4,
        diff_normalized,
    )
    report_comparison(
        "Diff mapped output before LCA: C++ / 127 vs PyTorch",
        get_trace_integers(entries, "fm.path.diff.output_pre_lca", FM_DIM),
        CPP_FT_SCALE,
        diff_output,
    )
    report_comparison(
        "Diff-derived Main gate sigmoid: C++ Q6 vs PyTorch",
        get_trace_integers(entries, "fm.path.diff.main_gate_q64", FM_DIM),
        64.0,
        main_gate,
    )
    report_comparison(
        "Diff-derived Main gate multiplier: C++ Q7 vs PyTorch",
        get_trace_integers(
            entries, "fm.path.diff.main_gate_multiplier_q128", FM_DIM
        ),
        128.0,
        main_gate_multiplier,
    )
    report_comparison(
        "Abs gate pre-activation: C++ / 8128 vs PyTorch",
        get_trace_integers(entries, "fm.path.abs.gate_preact", FM_DIM),
        nnue.weight_scale_hidden * nnue.quantized_one,
        abs_gate,
    )
    report_comparison(
        "Abs value pre-activation: C++ / 8128 vs PyTorch",
        get_trace_integers(entries, "fm.path.abs.value_preact", FM_DIM),
        nnue.weight_scale_hidden * nnue.quantized_one,
        abs_value,
    )
    report_float_bits_comparison(
        "Abs sigmoid gate",
        get_trace_integers(
            entries, "fm.path.abs.gate_sigmoid_f32_bits", FM_DIM
        ),
        abs_gate_sigmoid,
    )
    report_comparison(
        "Abs gated value: C++ / 8128 vs PyTorch",
        get_trace_integers(entries, "fm.path.abs.gated_value", FM_DIM),
        nnue.weight_scale_hidden * nnue.quantized_one,
        abs_gated,
    )
    report_float_bits_comparison(
        "Abs mapped value before round: C++ / 127 vs PyTorch",
        get_trace_integers(
            entries, "fm.path.abs.scaled_before_round_f32_bits", FM_DIM
        ),
        abs_output,
        cpp_multiplier=1.0 / CPP_FT_SCALE,
    )
    report_comparison(
        "Abs mapped output: C++ / 127 vs PyTorch",
        get_trace_integers(entries, "fm.path.abs.output", FM_DIM),
        CPP_FT_SCALE,
        abs_output,
    )
    report_comparison(
        "Abs squared output: C++ / 127 vs PyTorch",
        get_trace_integers(entries, "fm.path.abs.squared_output", FM_DIM),
        CPP_FT_SCALE,
        abs_squared_output,
    )


def compare_native_lca(
    entries, nnue, raw_diff, raw_abs, main_pair_outputs, us_perspective
):
    bucket = get_trace_integers(
        entries, "lca.selected_bucket", expected_count=1
    )[0]
    diff_input, abs_input = make_native_scaled_paths(raw_diff, raw_abs)
    fm_start = bucket * 64
    fm_end = fm_start + 64
    main_start = bucket * FM_DIM
    main_end = main_start + FM_DIM

    diff_fc = F.linear(
        diff_input.unsqueeze(0),
        nnue.layer_stacks.fm_diff.weight[fm_start:fm_end],
        nnue.layer_stacks.fm_diff.bias[fm_start:fm_end],
    ).squeeze(0)
    abs_fc = F.linear(
        abs_input.unsqueeze(0),
        nnue.layer_stacks.fm_abs.weight[fm_start:fm_end],
        nnue.layer_stacks.fm_abs.bias[fm_start:fm_end],
    ).squeeze(0)
    diff_gate, diff_value = diff_fc[:FM_DIM], diff_fc[FM_DIM:]
    abs_gate, abs_value = abs_fc[:FM_DIM], abs_fc[FM_DIM:]
    diff_normalized = diff_value * torch.rsqrt(
        diff_value.pow(2).mean() + 1e-8
    )
    diff_pre_lca = torch.clamp(diff_normalized * 0.2 + 0.5, 0.0, 1.0)
    abs_output = torch.clamp(
        abs_value * torch.sigmoid(abs_gate) * 0.05 + 0.6, 0.0, 1.0
    )

    them_perspective = (
        "WHITE" if us_perspective == "BLACK" else "BLACK"
    )
    main_input = torch.cat(
        (
            main_pair_outputs[us_perspective],
            main_pair_outputs[them_perspective],
        ),
        dim=0,
    )
    main_fc_before_gate = (
        F.linear(
            main_input.unsqueeze(0),
            nnue.layer_stacks.l1.weight[main_start:main_end],
            nnue.layer_stacks.l1.bias[main_start:main_end],
        ).squeeze(0)
        + nnue.layer_stacks.l1_fact(main_input.unsqueeze(0)).squeeze(0)
    )
    main_gate_multiplier = 0.5 + 0.5 * torch.sigmoid(diff_gate - 0.3)
    main_fc_after_gate = main_fc_before_gate * main_gate_multiplier
    query_input = torch.clamp(main_fc_after_gate[:31], 0.0, 1.0)
    fm_input = torch.cat((diff_pre_lca, abs_output), dim=0)

    query = nnue.layer_stacks.q_proj(query_input.unsqueeze(0)).squeeze(0)
    key = nnue.layer_stacks.k_proj(fm_input.unsqueeze(0)).squeeze(0)
    value = nnue.layer_stacks.v_proj(fm_input.unsqueeze(0)).squeeze(0)
    dot_product = (query * key).sum()
    temperature = torch.clamp(nnue.layer_stacks.lca_temp, min=0.125)
    attention_logit = (dot_product / 5.656) / temperature
    attention_score = torch.sigmoid(attention_logit)
    value_full = torch.zeros(FM_DIM, dtype=value.dtype, device=value.device)
    value_indices = tuple(getattr(nnue, "lca_value_indices", range(FM_DIM)))
    value_full[list(value_indices)] = value
    value_clamped = torch.clamp(value_full * 0.4 + 0.5, 0.0, 1.0)
    output_post_lca = (
        diff_pre_lca * (1.0 - attention_score)
        + value_clamped * attention_score
    )
    correction = output_post_lca - diff_pre_lca

    print("[LCA native PyTorch float]")
    print(f"  selected bucket: {bucket}")
    report_comparison(
        "Main FC before Diff gate: C++ / 8128 vs PyTorch",
        get_trace_integers(
            entries, "lca.main_fc_preact_before_gate", FM_DIM
        ),
        nnue.weight_scale_hidden * nnue.quantized_one,
        main_fc_before_gate,
    )
    report_comparison(
        "Main FC after Diff gate: C++ / 8128 vs PyTorch",
        get_trace_integers(
            entries, "lca.main_fc_preact_after_gate", FM_DIM
        ),
        nnue.weight_scale_hidden * nnue.quantized_one,
        main_fc_after_gate,
    )
    report_comparison(
        "LCA Query input: C++ / 127 vs PyTorch",
        get_trace_integers(entries, "lca.query_input", 31),
        CPP_FT_SCALE,
        query_input,
    )
    report_comparison(
        "LCA Key/Value FM input: C++ / 127 vs PyTorch",
        get_trace_integers(entries, "lca.fm_input", 2 * FM_DIM),
        CPP_FT_SCALE,
        fm_input,
    )
    report_comparison(
        "LCA Query pre-activation: C++ / 8128 vs PyTorch",
        get_trace_integers(entries, "lca.query_preact", FM_DIM),
        nnue.weight_scale_hidden * nnue.quantized_one,
        F.pad(query, (0, FM_DIM - query.numel())),
    )
    report_comparison(
        "LCA Key pre-activation: C++ / 8128 vs PyTorch",
        get_trace_integers(entries, "lca.key_preact", FM_DIM),
        nnue.weight_scale_hidden * nnue.quantized_one,
        F.pad(key, (0, FM_DIM - key.numel())),
    )
    report_comparison(
        "LCA Value pre-activation: C++ / 8128 vs PyTorch",
        get_trace_integers(entries, "lca.value_preact", FM_DIM),
        nnue.weight_scale_hidden * nnue.quantized_one,
        F.pad(value, (0, FM_DIM - value.numel())),
    )
    report_float_bits_comparison(
        "LCA temperature",
        get_trace_integers(entries, "lca.temperature_f32_bits", 1),
        temperature.reshape(1),
    )
    report_float_bits_comparison(
        "LCA Q/K dot product",
        get_trace_integers(entries, "lca.dot_product_f32_bits", 1),
        dot_product.reshape(1),
    )
    report_float_bits_comparison(
        "LCA attention logit",
        get_trace_integers(entries, "lca.attention_logit_f32_bits", 1),
        attention_logit.reshape(1),
    )
    report_float_bits_comparison(
        "LCA attention score",
        get_trace_integers(entries, "lca.attention_score_f32_bits", 1),
        attention_score.reshape(1),
    )
    report_float_bits_comparison(
        "LCA Value mapped/clamped",
        get_trace_integers(entries, "lca.value_clamped_f32_bits", FM_DIM),
        value_clamped,
    )
    report_float_bits_comparison(
        "LCA correction",
        get_trace_integers(entries, "lca.correction_f32_bits", FM_DIM),
        correction,
    )
    report_float_bits_comparison(
        "Diff output post-LCA before integer conversion",
        get_trace_integers(
            entries, "lca.output_post_lca_f32_bits", FM_DIM
        ),
        output_post_lca,
    )
    report_comparison(
        "Diff output post-LCA: C++ / 127 vs PyTorch",
        get_trace_integers(entries, "lca.output_post_lca", FM_DIM),
        CPP_FT_SCALE,
        output_post_lca,
    )
    return {
        "main_input": main_input,
        "main_fc_after_gate": main_fc_after_gate,
        "diff_post_lca": output_post_lca,
        "abs_output": abs_output,
    }


def compare_native_deep(entries, nnue, raw_diff, raw_abs, lca_native):
    bucket = get_trace_integers(
        entries, "deep.selected_bucket", expected_count=1
    )[0]
    pair_bucket = get_trace_integers(
        entries, "pair.bucket_id", expected_count=1
    )[0]
    diff_input, abs_input = make_native_scaled_paths(raw_diff, raw_abs)
    main_input = lca_native["main_input"]
    main_raw = torch.clamp(
        lca_native["main_fc_after_gate"][:31], 0.0, 1.0
    )
    main_squared = main_raw.pow(2.0) * (127.0 / 128.0)
    diff_post_lca = lca_native["diff_post_lca"]
    abs_output = lca_native["abs_output"]
    abs_squared = abs_output.pow(2.0)

    phase_abs_input = torch.clamp(abs_input - 0.5, 0.0, 1.0) * 2.0
    phase_abs_input = phase_abs_input.clone()
    phase_abs_input[127] = pair_bucket / 11.0
    phase_input = torch.cat(
        (phase_abs_input, diff_input, main_input[:128]), dim=0
    )
    phase_preact = nnue.layer_stacks.phase_proj(
        phase_input.unsqueeze(0)
    ).squeeze(0)[:6]
    phase_logit = phase_preact * 3.0 + 1.0
    phase_sigmoid = torch.sigmoid(phase_logit)
    phase_value = 0.1 + 0.9 * phase_sigmoid
    phase_multipliers = torch.tensor(
        [1.3, 1.5, 1.0, 0.7, 0.88, 1.5],
        dtype=phase_value.dtype,
        device=phase_value.device,
    )
    channel_scales = (0.5 + 0.5 * phase_value) * phase_multipliers

    cross_main_squared = main_squared[:16]
    cross_diff = diff_post_lca[:16]
    cross_main_raw = main_raw[:16]
    cross_abs = abs_output[:16]
    cross_product_diff = cross_main_squared * cross_diff
    cross_product_abs = cross_main_raw * cross_abs
    cross_input = torch.cat((cross_product_diff, cross_product_abs), dim=0)
    cross_start = bucket * FM_DIM
    cross_end = cross_start + FM_DIM
    cross_preact = F.linear(
        cross_input.unsqueeze(0),
        nnue.layer_stacks.cross_proj.weight[cross_start:cross_end],
        nnue.layer_stacks.cross_proj.bias[cross_start:cross_end],
    ).squeeze(0)
    cross_output = torch.clamp(cross_preact, 0.0, 1.0)

    fc1_parts = [
        main_squared * channel_scales[0],
        main_raw * channel_scales[1],
        diff_post_lca * channel_scales[2],
        abs_output * channel_scales[3],
    ]
    if not nnue.remove_abs_sqr_l2:
        fc1_parts.append(abs_squared * channel_scales[4])
    fc1_parts.extend((
        cross_output * channel_scales[5],
        torch.zeros(2, dtype=main_raw.dtype, device=main_raw.device),
    ))
    fc1_input = torch.cat(tuple(fc1_parts), dim=0).clamp(0.0, 1.0)
    fc1_start = bucket * 96
    fc1_end = fc1_start + 96
    fc1_preact = F.linear(
        fc1_input.unsqueeze(0),
        nnue.layer_stacks.l2.weight[
            fc1_start:fc1_end, :nnue.layer_stacks.l2_in_total
        ],
        nnue.layer_stacks.l2.bias[fc1_start:fc1_end],
    ).squeeze(0)
    fc1_output = torch.clamp(fc1_preact, 0.0, 1.0)
    fc2_preact = F.linear(
        fc1_output.unsqueeze(0),
        nnue.layer_stacks.output.weight[bucket:bucket + 1],
        nnue.layer_stacks.output.bias[bucket:bucket + 1],
    ).reshape(-1)

    hidden_scale = nnue.weight_scale_hidden * nnue.quantized_one
    output_scale = nnue.weight_scale_out * nnue.nnue2score
    print("[Cross / bucket network native PyTorch float]")
    print(f"  selected bucket: {bucket}")
    print(f"  PyTorch phase bucket input: {pair_bucket}")
    report_comparison(
        "Phase input: C++ / 127 vs PyTorch",
        get_trace_integers(entries, "deep.phase.input", ROUTER_INPUT_DIM),
        CPP_FT_SCALE,
        phase_input,
    )
    report_comparison(
        "Phase pre-activation: C++ / 8128 vs PyTorch",
        get_trace_integers(entries, "deep.phase.preact", 6),
        hidden_scale,
        phase_preact,
    )
    for label, trace_name, native_value in (
        ("Phase logit", "deep.phase.logit_f32_bits", phase_logit),
        ("Phase sigmoid", "deep.phase.sigmoid_f32_bits", phase_sigmoid),
        ("Phase value", "deep.phase.value_f32_bits", phase_value),
        (
            "Phase channel scale",
            "deep.phase.channel_scale_f32_bits",
            channel_scales,
        ),
    ):
        report_float_bits_comparison(
            label, get_trace_integers(entries, trace_name, 6), native_value
        )
    for label, trace_name, native_value, count in (
        ("Main raw", "deep.main.raw", main_raw, 31),
        ("Main squared", "deep.main.squared", main_squared, 31),
        ("Cross Main squared", "deep.cross.main_squared", cross_main_squared, 16),
        ("Cross Diff", "deep.cross.diff", cross_diff, 16),
        ("Cross Main raw", "deep.cross.main_raw", cross_main_raw, 16),
        ("Cross Abs", "deep.cross.abs", cross_abs, 16),
        ("Cross product Diff", "deep.cross.product_diff", cross_product_diff, 16),
        ("Cross product Abs", "deep.cross.product_abs", cross_product_abs, 16),
        ("Cross input", "deep.cross.input", cross_input, 32),
    ):
        report_comparison(
            f"{label}: C++ / 127 vs PyTorch",
            get_trace_integers(entries, trace_name, count),
            CPP_FT_SCALE,
            native_value,
        )
    report_comparison(
        "Cross pre-activation: C++ / 8128 vs PyTorch",
        get_trace_integers(entries, "deep.cross.preact", FM_DIM),
        hidden_scale,
        cross_preact,
    )
    report_comparison(
        "Cross output: C++ / 127 vs PyTorch",
        get_trace_integers(entries, "deep.cross.output", FM_DIM),
        CPP_FT_SCALE,
        cross_output,
    )
    report_comparison(
        "FC1 input: C++ / 127 vs PyTorch",
        get_trace_integers(
            entries, "deep.fc1.input", nnue.layer_stacks.l2_in_total
        ),
        CPP_FT_SCALE,
        fc1_input,
    )
    report_comparison(
        "FC1 pre-activation: C++ / 8128 vs PyTorch",
        get_trace_integers(entries, "deep.fc1.preact", 96),
        hidden_scale,
        fc1_preact,
    )
    report_comparison(
        "FC1 activation: C++ / 127 vs PyTorch",
        get_trace_integers(entries, "deep.fc1.output", 96),
        CPP_FT_SCALE,
        fc1_output,
    )
    report_comparison(
        "FC2 pre-activation: C++ output scale vs PyTorch",
        get_trace_integers(entries, "deep.fc2.preact", 1),
        output_scale,
        fc2_preact,
    )
    return {
        "selected_bucket": bucket,
        "deep_output": fc2_preact,
        "bypass_output": lca_native["main_fc_after_gate"][31:32],
    }


def compare_native_final(entries, nnue, deep_native):
    bucket = deep_native["selected_bucket"]
    deep_output = deep_native["deep_output"]
    bypass_output = deep_native["bypass_output"]
    blend_parameter_raw = nnue.layer_stacks.blend[bucket]
    blend_alpha = torch.sigmoid(blend_parameter_raw).reshape(1)
    deep_term = deep_output * blend_alpha
    bypass_term = bypass_output * (1.0 - blend_alpha)
    blended_output = deep_term + bypass_term

    output_scale = get_trace_integers(
        entries, "final.scale.network_output", expected_count=1
    )[0]
    alpha_scale = get_trace_integers(
        entries, "final.scale.blend_alpha", expected_count=1
    )[0]
    numerator_scale = get_trace_integers(
        entries, "final.scale.blend_numerator", expected_count=1
    )[0]
    bypass_input_scale = get_trace_integers(
        entries, "final.scale.bypass_input", expected_count=1
    )[0]
    fv_scale = get_trace_integers(
        entries, "final.fv_scale", expected_count=1
    )[0]
    value_max_eval = get_trace_integers(
        entries, "final.value_max_eval", expected_count=1
    )[0]

    native_score = blended_output * nnue.nnue2score
    native_score_clamped = torch.clamp(
        native_score, -float(value_max_eval), float(value_max_eval)
    )

    print("[Bypass / blend / final eval native PyTorch float]")
    print(f"  selected Router bucket : {bucket}")
    print(
        "  material/pair bucket   : "
        f"{get_trace_integers(entries, 'final.material_bucket', 1)[0]}"
    )
    print(f"  checkpoint blend raw   : {blend_parameter_raw.item():.9e}")
    print(f"  sigmoid(raw) alpha     : {blend_alpha.item():.9e}")
    print(
        "  serialized alpha Q14   : "
        f"{get_trace_integers(entries, 'final.blend.alpha_q14', 1)[0]}"
        f" / {alpha_scale}"
    )
    print(f"  C++ FV_SCALE           : {fv_scale}")
    print(f"  PyTorch nnue2score     : {nnue.nnue2score:g}")
    print(
        "  C++ equivalent multiplier: "
        f"{float(output_scale) / float(fv_scale):.9e}"
    )

    report_comparison(
        "Deep output: C++ / output scale vs PyTorch",
        get_trace_integers(entries, "final.deep_output", 1),
        output_scale,
        deep_output,
    )
    report_comparison(
        "Bypass input/pre-activation: C++ / 8128 vs PyTorch",
        get_trace_integers(entries, "final.bypass.input", 1),
        bypass_input_scale,
        bypass_output,
    )
    report_comparison(
        "Bypass rescaled output: C++ / output scale vs PyTorch",
        get_trace_integers(entries, "final.bypass.output", 1),
        output_scale,
        bypass_output,
    )
    report_comparison(
        "Blend alpha: C++ Q14 vs PyTorch sigmoid(raw)",
        get_trace_integers(entries, "final.blend.alpha_q14", 1),
        alpha_scale,
        blend_alpha,
    )
    report_comparison(
        "Blend deep term: C++ numerator scale vs PyTorch",
        get_trace_integers(entries, "final.blend.deep_term", 1),
        numerator_scale,
        deep_term,
    )
    report_comparison(
        "Blend bypass term: C++ numerator scale vs PyTorch",
        get_trace_integers(entries, "final.blend.bypass_term", 1),
        numerator_scale,
        bypass_term,
    )
    report_comparison(
        "Blend numerator: C++ numerator scale vs PyTorch",
        get_trace_integers(entries, "final.blend.numerator", 1),
        numerator_scale,
        blended_output,
    )
    report_comparison(
        "Blend output: C++ / output scale vs PyTorch",
        get_trace_integers(entries, "final.blend.output", 1),
        output_scale,
        blended_output,
    )
    report_comparison(
        "Final network output: C++ / output scale vs PyTorch",
        get_trace_integers(entries, "final.network_output", 1),
        output_scale,
        blended_output,
    )
    report_comparison(
        "Final eval before clamp: C++ Value vs PyTorch scorenet*nnue2score",
        get_trace_integers(entries, "final.eval_before_clamp", 1),
        1,
        native_score,
    )
    report_comparison(
        "Final eval after clamp: C++ Value vs PyTorch clamped score",
        get_trace_integers(entries, "final.eval_after_clamp", 1),
        1,
        native_score_clamped,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare a native PyTorch NNUE checkpoint with NNUE_TRACE_V1"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Native .ckpt or architecture-tagged recovery .pt file")
    parser.add_argument("--trace", required=True, help="NNUE_TRACE_V1 TSV file")
    parser.add_argument(
        "--features",
        default="HalfKA_KSDG3",
        help="Feature set used by the checkpoint (default: HalfKA_KSDG3)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="CUDA device used by the custom Feature Transformer (default: cuda)",
    )
    args = parser.parse_args()

    entries = read_trace(args.trace)
    validate_trace_mapping(entries)
    us_perspective, them_perspective = validate_perspectives(entries)

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("The current custom Feature Transformer requires CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if device.index is not None:
        torch.cuda.set_device(device)

    feature_set = features.get_feature_set_from_name(args.features)
    if args.checkpoint.lower().endswith(".pt"):
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        architecture = saved.get("architecture", {}) if isinstance(saved, dict) else {}
        if not isinstance(saved, dict) or "state_dict" not in saved:
            raise ValueError("Recovery .pt must contain state_dict and architecture")
        nnue = M.NNUE(feature_set, **M.nnue_architecture_kwargs(architecture))
        nnue.load_state_dict(saved["state_dict"], strict=True)
    else:
        nnue = M.NNUE.load_from_checkpoint(
            args.checkpoint,
            feature_set=feature_set,
            map_location="cpu",
        )
    nnue.eval()
    if nnue.input.num_inputs != feature_set.num_features:
        raise ValueError(
            f"Checkpoint input size {nnue.input.num_inputs} does not match "
            f"feature set size {feature_set.num_features}"
        )
    if nnue.input.num_outputs != M.L1_MAIN or nnue.input.factor_dim != FM_DIM:
        raise ValueError(
            "Checkpoint Feature Transformer shape does not match "
            f"L1_MAIN={M.L1_MAIN}, FM_DIM={FM_DIM}"
        )
    if (
        nnue.layer_stacks.router.in_features != ROUTER_INPUT_DIM
        or nnue.layer_stacks.router.out_features != ROUTER_OUTPUT_DIM
    ):
        raise ValueError(
            "Checkpoint Router shape does not match "
            f"{ROUTER_INPUT_DIM}->{ROUTER_OUTPUT_DIM}"
        )

    # NNUEWriter quantizes on the CPU model loaded from the checkpoint.
    integer_reference = make_integer_reference(entries, nnue)
    nnue.to(device)

    inputs = make_sparse_inputs(entries, feature_set, device)
    print("[Trace metadata]")
    print(f"  SFEN             : {get_trace_string(entries, 'meta.sfen')}")
    print(f"  us / them        : {us_perspective} / {them_perspective}")
    print(
        "  pair bucket      : "
        f"{get_trace_integers(entries, 'pair.bucket_id', 1)[0]}"
    )
    print(
        "  router bucket    : "
        f"{get_trace_integers(entries, 'router.selected_bucket', 1)[0]}"
    )
    print("  C++ BLACK        : PyTorch white_indices / t_w / v_w")
    print("  C++ WHITE        : PyTorch black_indices / t_b / v_b")
    print(
        f"  active features  : C++ BLACK={inputs['cpp_black_count']}, "
        f"C++ WHITE={inputs['cpp_white_count']}"
    )
    print(
        f"  PyTorch entries  : white={inputs['pytorch_white_count']}, "
        f"black={inputs['pytorch_black_count']}"
    )

    compare_integer_reference(entries, integer_reference)

    with torch.no_grad():
        t_w, t_b, v_w, v_b = nnue.input(
            inputs["white_indices"],
            inputs["white_values"],
            inputs["black_indices"],
            inputs["black_values"],
        )

        fm_by_perspective = {
            "BLACK": process_fm(v_w, inputs["white_indices"]),
            "WHITE": process_fm(v_b, inputs["black_indices"]),
        }
        raw_diff, raw_abs = make_raw_paths(inputs, fm_by_perspective)

        print("[Native PyTorch float reference]")
        compare_main_ft(entries, t_w, t_b)
        main_pair_outputs = compare_native_pair_main(entries, nnue, t_w, t_b)
        compare_fm_accumulators(entries, fm_by_perspective)
        compare_fm_interactions(entries, fm_by_perspective)
        compare_raw_paths(entries, raw_diff, raw_abs)
        compare_scaled_paths(entries, raw_diff, raw_abs)
        compare_native_router(
            entries,
            nnue,
            raw_diff,
            raw_abs,
            main_pair_outputs,
            us_perspective,
        )
        compare_native_fm_path(entries, nnue, raw_diff, raw_abs)
        lca_native = compare_native_lca(
            entries,
            nnue,
            raw_diff,
            raw_abs,
            main_pair_outputs,
            us_perspective,
        )
        deep_native = compare_native_deep(
            entries, nnue, raw_diff, raw_abs, lca_native
        )
        compare_native_final(entries, nnue, deep_native)


if __name__ == "__main__":
    main()

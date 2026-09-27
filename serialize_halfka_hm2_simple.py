"""Dedicated nn.bin serializer/reader for Experiment 85 simple SFNN."""

import argparse
import io
import struct
from pathlib import Path

import numpy as np
import torch

import features
from serialize import encode_leb_128_array
from simple_halfka_hm2_model import (
    FEATURE_NAME, FT_INPUTS, FT_WIDTH, LAYER_STACKS,
    SimpleHalfKAHM2NNUE,
)
from simple_pp3wide import (
    PP3WIDE_FEATURES, PP3WIDE_QUANT_SCALE, PP3WIDE_TYPE,
    PP3WIDE64_TYPE, PP3WIDE64_WIDTH,
)
from simple_local_pair64 import (
    LOCALPAIR64_FEATURES, LOCALPAIR64_TYPE, LOCALPAIR64_WIDTH,
)
from simple_ksg_local_pair64 import (
    KSG_LOCALPAIR64_FEATURES, KSG_LOCALPAIR64_TYPE,
    KSG_LOCALPAIR64_WIDTH,
)
from simple_gs_local_pair64 import (
    GS_LOCALPAIR64_FEATURES, GS_LOCALPAIR64_TYPE, GS_LOCALPAIR64_WIDTH,
)
from simple_gs_local_pair32 import (
    GS_LOCALPAIR32_FEATURES, GS_LOCALPAIR32_TYPE, GS_LOCALPAIR32_WIDTH,
)
from simple_gs_local_pair32_d1 import (
    GS_LOCALPAIR32_D1_FEATURES, GS_LOCALPAIR32_D1_TYPE,
    GS_LOCALPAIR32_D1_WIDTH,
)


VERSION = 0x7AF32F16
OUTER_HASH = 0x74517600
FT_HASH = 0x7F234CB8 ^ FT_WIDTH
NETWORK_HASH = 0x6333718A ^ 0x484D3202
DESCRIPTION = (
    "ModelType=SFNNWithoutPsqt;"
    "Features=HalfKA_hm2_NoDG(Friend)[73305->1536x2],"
    "Network=SFNN-1536-HalfKAHM2-NoDG-v2{LayerStack=9}"
)
FT_HASH_PP3WIDE = FT_HASH ^ 0x50335731
NETWORK_HASH_PP3WIDE = NETWORK_HASH ^ 0x50335031
DESCRIPTION_PP3WIDE = (
    "ModelType=SFNNWithoutPsqt;"
    "Features=HalfKA_hm2_NoDG(Friend)+PP3WidePL[73305+15552->1536x2],"
    "Network=SFNN-1536-HalfKAHM2-NoDG-PP3WPL-v3{LayerStack=9}"
)
FT_HASH_PP3WIDE64 = FT_HASH ^ 0x50335764
NETWORK_HASH_PP3WIDE64 = NETWORK_HASH ^ 0x50335064
DESCRIPTION_PP3WIDE64 = (
    "ModelType=SFNNWithoutPsqt;"
    "Features=HalfKA_hm2_NoDG(Friend)+PP3WidePL64"
    "[73305->1536x2;15552->64x2->EWM64->Proj16],"
    "Network=SFNN-1536-HalfKAHM2-NoDG-PP3WPL64-v4{LayerStack=9}"
)
FT_HASH_LOCALPAIR64 = FT_HASH ^ 0x4C503634
NETWORK_HASH_LOCALPAIR64 = NETWORK_HASH ^ 0x4C503634
DESCRIPTION_LOCALPAIR64 = (
    "ModelType=SFNNWithoutPsqt;"
    "Features=HalfKA_hm2_NoDG(Friend)+LocalPair64-L4"
    "[73305->1536x2;184320->64x2->EWM64->Proj16],"
    "Network=SFNN-1536-HalfKAHM2-NoDG-LocalPair64-v1{LayerStack=9}"
)
FT_HASH_KSG_LOCALPAIR64 = FT_HASH ^ 0x4B534732
NETWORK_HASH_KSG_LOCALPAIR64 = NETWORK_HASH ^ 0x4B534732
DESCRIPTION_KSG_LOCALPAIR64 = (
    "ModelType=SFNNWithoutPsqt;"
    "Features=HalfKA_hm2_NoDG(Friend)+KSGLocalPair64-R2"
    "[73305->1536x2;25920->64x2->EWM64->Proj16],"
    "Network=SFNN-1536-HalfKAHM2-NoDG-KSGLocalPair64-v1{LayerStack=9}"
)
FT_HASH_GS_LOCALPAIR64 = FT_HASH ^ 0x47535235
NETWORK_HASH_GS_LOCALPAIR64 = NETWORK_HASH ^ 0x47535235
DESCRIPTION_GS_LOCALPAIR64 = (
    "ModelType=SFNNWithoutPsqt;"
    "Features=HalfKA_hm2_NoDG(Friend)+GSLocalPair64-R5"
    "[73305->1536x2;11520->64x2->EWM64->Proj16],"
    "Network=SFNN-1536-HalfKAHM2-NoDG-GSLocalPair64-R5-v1{LayerStack=9}"
)
FT_HASH_GS_LOCALPAIR32 = FT_HASH ^ 0x47533332
NETWORK_HASH_GS_LOCALPAIR32 = NETWORK_HASH ^ 0x47533332
DESCRIPTION_GS_LOCALPAIR32 = (
    "ModelType=SFNNWithoutPsqt;"
    "Features=HalfKA_hm2_NoDG(Friend)+GSLocalPair32-R5"
    "[73305->1536x2;11520->32x2->EWM32->Proj16],"
    "Network=SFNN-1536-HalfKAHM2-NoDG-GSLocalPair32-R5-v1{LayerStack=9}"
)
FT_HASH_GS_LOCALPAIR32_D1 = FT_HASH ^ 0x47334431
NETWORK_HASH_GS_LOCALPAIR32_D1 = NETWORK_HASH ^ 0x47334431
DESCRIPTION_GS_LOCALPAIR32_D1 = (
    "ModelType=SFNNWithoutPsqt;"
    "Features=HalfKA_hm2_NoDG(Friend)+GSLocalPair32-R5-D1"
    "[73305->1536x2;4352->32x2->EWM32->Proj16],"
    "Network=SFNN-1536-HalfKAHM2-NoDG-GSLocalPair32-R5-D1-v1"
    "{LayerStack=9}"
)
Q_ONE = 127.0
HIDDEN_WEIGHT_SCALE = 64.0
LEB128_MAGIC = b"COMPRESSED_LEB128"


def _u32(buf, value):
    buf.extend(struct.pack("<I", int(value) & 0xFFFFFFFF))


def _write_tensor(buf, tensor, compression):
    values = tensor.detach().cpu().numpy().reshape(-1)
    if compression == "none":
        buf.extend(values.tobytes())
    elif compression == "leb128":
        encoded = bytes(encode_leb_128_array(values))
        buf.extend(LEB128_MAGIC)
        _u32(buf, len(encoded))
        buf.extend(encoded)
    else:
        raise ValueError(f"unsupported FT compression: {compression}")


def _write_fc(buf, layer):
    weight_scale = HIDDEN_WEIGHT_SCALE
    bias_scale = HIDDEN_WEIGHT_SCALE * Q_ONE
    limit = Q_ONE / weight_scale
    weight = layer.weight.detach().clamp(-limit, limit).mul(weight_scale).round().to(torch.int8)
    bias = layer.bias.detach().mul(bias_scale).round().to(torch.int32)
    outputs, inputs = weight.shape
    # Explicit-dimension C++ layers serialize logical output rows (unlike the
    # legacy complex writer, which pads several physical output matrices).
    padded_outputs = outputs
    padded_inputs = (inputs + 31) // 32 * 32
    pb = torch.zeros(padded_outputs, dtype=torch.int32)
    pw = torch.zeros((padded_outputs, padded_inputs), dtype=torch.int8)
    pb[:outputs] = bias.cpu()
    pw[:outputs, :inputs] = weight.cpu()
    buf.extend(pb.numpy().tobytes())
    buf.extend(pw.numpy().tobytes())


def serialize_model(model, output, ft_compression="none"):
    if not isinstance(model, SimpleHalfKAHM2NNUE):
        raise TypeError("simple serializer accepts only SimpleHalfKAHM2NNUE")
    if getattr(model, "use_side_input", False):
        raise ValueError(
            "ply/material Simple side-input is Python-training-only; its C++ "
            "serializer/schema has intentionally not been implemented")
    buf = bytearray()
    pp_type = getattr(model, "simple_local_pair_feature", "off")
    pp_enabled = pp_type == PP3WIDE_TYPE
    pp64_enabled = pp_type == PP3WIDE64_TYPE
    local64_enabled = pp_type == LOCALPAIR64_TYPE
    ksg64_enabled = pp_type == KSG_LOCALPAIR64_TYPE
    gs64_enabled = pp_type == GS_LOCALPAIR64_TYPE
    gs32_enabled = pp_type == GS_LOCALPAIR32_TYPE
    gs32_d1_enabled = pp_type == GS_LOCALPAIR32_D1_TYPE
    _u32(buf, VERSION)
    _u32(buf, OUTER_HASH)
    desc = (DESCRIPTION_GS_LOCALPAIR32_D1 if gs32_d1_enabled
            else DESCRIPTION_GS_LOCALPAIR32 if gs32_enabled
            else DESCRIPTION_GS_LOCALPAIR64 if gs64_enabled
            else DESCRIPTION_KSG_LOCALPAIR64 if ksg64_enabled
            else DESCRIPTION_LOCALPAIR64 if local64_enabled
            else DESCRIPTION_PP3WIDE64 if pp64_enabled
            else DESCRIPTION_PP3WIDE if pp_enabled else DESCRIPTION).encode("utf-8")
    _u32(buf, len(desc))
    buf.extend(desc)
    _u32(buf, FT_HASH_GS_LOCALPAIR32_D1 if gs32_d1_enabled
         else FT_HASH_GS_LOCALPAIR32 if gs32_enabled
         else FT_HASH_GS_LOCALPAIR64 if gs64_enabled
         else FT_HASH_KSG_LOCALPAIR64 if ksg64_enabled
         else FT_HASH_LOCALPAIR64 if local64_enabled
         else FT_HASH_PP3WIDE64 if pp64_enabled
         else FT_HASH_PP3WIDE if pp_enabled else FT_HASH)
    _write_tensor(
        buf, model.input.bias.mul(Q_ONE).round().to(torch.int16),
        ft_compression)
    # Virtual factorization is a training-only parameterization.  Export a
    # fresh coalesced tensor without modifying either specific or virtual
    # parameters; repeated serialization must therefore be byte-identical.
    effective_ft_weight = model.input.effective_weight()
    _write_tensor(
        buf, effective_ft_weight.mul(Q_ONE).round().to(torch.int16),
        ft_compression)
    if pp_enabled or pp64_enabled or local64_enabled or ksg64_enabled or gs64_enabled or gs32_enabled or gs32_d1_enabled:
        if model.pp3wide is None:
            raise ValueError("PP3Wide architecture is missing its component")
        pp = model.pp3wide.weight.detach().mul(PP3WIDE_QUANT_SCALE) \
            .round().clamp(-127, 127).to(torch.int8).cpu().numpy()
        buf.extend(pp.tobytes())
        if pp64_enabled or local64_enabled or ksg64_enabled or gs64_enabled or gs32_enabled or gs32_d1_enabled:
            projection = model.pp3wide.projection.weight.detach() \
                .mul(HIDDEN_WEIGHT_SCALE).round().clamp(-127, 127) \
                .to(torch.int8).cpu().numpy()
            buf.extend(projection.tobytes())
    for stack in model.layer_stacks:
        _u32(buf, NETWORK_HASH_GS_LOCALPAIR32_D1 if gs32_d1_enabled
             else NETWORK_HASH_GS_LOCALPAIR32 if gs32_enabled
             else NETWORK_HASH_GS_LOCALPAIR64 if gs64_enabled
             else NETWORK_HASH_KSG_LOCALPAIR64 if ksg64_enabled
             else NETWORK_HASH_LOCALPAIR64 if local64_enabled
             else NETWORK_HASH_PP3WIDE64 if pp64_enabled
             else NETWORK_HASH_PP3WIDE if pp_enabled else NETWORK_HASH)
        _write_fc(buf, stack.fc0)
        _write_fc(buf, stack.fc1)
        _write_fc(buf, stack.fc2)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(buf)
    return bytes(buf)


def _read_u32(stream):
    raw = stream.read(4)
    if len(raw) != 4:
        raise EOFError
    return struct.unpack("<I", raw)[0]


def _read_leb(stream, count):
    if stream.read(len(LEB128_MAGIC)) != LEB128_MAGIC:
        raise ValueError("missing LEB128 marker")
    size = _read_u32(stream)
    raw = stream.read(size)
    out = np.empty(count, dtype=np.int16)
    pos = 0
    for i in range(count):
        value = shift = 0
        while True:
            byte = raw[pos]; pos += 1
            value |= (byte & 0x7f) << shift
            shift += 7
            if not byte & 0x80:
                if byte & 0x40:
                    value |= -(1 << shift)
                out[i] = value
                break
    if pos != len(raw):
        raise ValueError("LEB128 payload has trailing data")
    return out


def _read_array(stream, dtype, count):
    dtype = np.dtype(dtype).newbyteorder("<")
    raw = stream.read(dtype.itemsize * count)
    if len(raw) != dtype.itemsize * count:
        raise EOFError
    return np.frombuffer(raw, dtype=dtype).copy()


def _read_ft_tensor(stream, count):
    start = stream.tell()
    compressed = stream.read(len(LEB128_MAGIC)) == LEB128_MAGIC
    stream.seek(start)
    if compressed:
        return _read_leb(stream, count)
    return _read_array(stream, np.int16, count)


def _copy_parameter(parameter, values, scale):
    tensor = torch.from_numpy(values.astype(np.float32, copy=False))
    tensor = tensor.reshape(parameter.shape).div(float(scale))
    with torch.no_grad():
        parameter.copy_(tensor)


def _read_fc(stream, layer):
    weight_scale = HIDDEN_WEIGHT_SCALE
    bias_scale = HIDDEN_WEIGHT_SCALE * Q_ONE
    outputs, inputs = layer.weight.shape
    padded_inputs = (inputs + 31) // 32 * 32
    bias = _read_array(stream, np.int32, outputs)
    weight = _read_array(stream, np.int8, outputs * padded_inputs)
    weight = weight.reshape(outputs, padded_inputs)[:, :inputs].copy()
    _copy_parameter(layer.bias, bias, bias_scale)
    _copy_parameter(layer.weight, weight, weight_scale)


def deserialize_model(source, feature_set):
    with Path(source).open("rb") as stream:
        if _read_u32(stream) != VERSION:
            raise ValueError("unsupported simple nn.bin version")
        if _read_u32(stream) != OUTER_HASH:
            raise ValueError("HalfKA_HM2 simple outer hash mismatch")
        description_size = _read_u32(stream)
        description = stream.read(description_size).decode("utf-8")
        if description not in (
                DESCRIPTION, DESCRIPTION_PP3WIDE, DESCRIPTION_PP3WIDE64,
                DESCRIPTION_LOCALPAIR64, DESCRIPTION_KSG_LOCALPAIR64,
                DESCRIPTION_GS_LOCALPAIR64, DESCRIPTION_GS_LOCALPAIR32,
                DESCRIPTION_GS_LOCALPAIR32_D1):
            raise ValueError(
                "HalfKA_HM2 simple architecture description mismatch: "
                f"{description!r}")
        pp_enabled = description == DESCRIPTION_PP3WIDE
        pp64_enabled = description == DESCRIPTION_PP3WIDE64
        local64_enabled = description == DESCRIPTION_LOCALPAIR64
        ksg64_enabled = description == DESCRIPTION_KSG_LOCALPAIR64
        gs64_enabled = description == DESCRIPTION_GS_LOCALPAIR64
        gs32_enabled = description == DESCRIPTION_GS_LOCALPAIR32
        gs32_d1_enabled = description == DESCRIPTION_GS_LOCALPAIR32_D1
        expected_ft_hash = (FT_HASH_GS_LOCALPAIR32_D1 if gs32_d1_enabled
                            else FT_HASH_GS_LOCALPAIR32 if gs32_enabled
                            else FT_HASH_GS_LOCALPAIR64 if gs64_enabled
                            else FT_HASH_KSG_LOCALPAIR64 if ksg64_enabled
                            else FT_HASH_LOCALPAIR64 if local64_enabled
                            else FT_HASH_PP3WIDE64 if pp64_enabled
                            else FT_HASH_PP3WIDE if pp_enabled else FT_HASH)
        if _read_u32(stream) != expected_ft_hash:
            raise ValueError("HalfKA_HM2 simple FT hash mismatch")

        model = SimpleHalfKAHM2NNUE(
            feature_set=feature_set,
            simple_local_pair_feature=(
                GS_LOCALPAIR32_D1_TYPE if gs32_d1_enabled
                else GS_LOCALPAIR32_TYPE if gs32_enabled
                else GS_LOCALPAIR64_TYPE if gs64_enabled
                else KSG_LOCALPAIR64_TYPE if ksg64_enabled
                else LOCALPAIR64_TYPE if local64_enabled
                else PP3WIDE64_TYPE if pp64_enabled
                else PP3WIDE_TYPE if pp_enabled else "off"))
        bias = _read_ft_tensor(stream, FT_WIDTH)
        weight = _read_ft_tensor(stream, FT_INPUTS * FT_WIDTH)
        _copy_parameter(model.input.bias, bias, Q_ONE)
        _copy_parameter(model.input.weight, weight, Q_ONE)
        if pp_enabled or pp64_enabled or local64_enabled or ksg64_enabled or gs64_enabled or gs32_enabled or gs32_d1_enabled:
            feature_count = (
                GS_LOCALPAIR32_D1_FEATURES if gs32_d1_enabled
                else GS_LOCALPAIR32_FEATURES if gs32_enabled
                else GS_LOCALPAIR64_FEATURES if gs64_enabled
                else KSG_LOCALPAIR64_FEATURES if ksg64_enabled
                else LOCALPAIR64_FEATURES if local64_enabled
                else PP3WIDE_FEATURES)
            pp = _read_array(
                stream, np.int8,
                feature_count * (
                    GS_LOCALPAIR32_D1_WIDTH if gs32_d1_enabled
                    else GS_LOCALPAIR32_WIDTH if gs32_enabled
                    else GS_LOCALPAIR64_WIDTH if gs64_enabled
                    else KSG_LOCALPAIR64_WIDTH if ksg64_enabled
                    else LOCALPAIR64_WIDTH if local64_enabled
                    else PP3WIDE64_WIDTH if pp64_enabled else FT_WIDTH))
            _copy_parameter(model.pp3wide.weight, pp, PP3WIDE_QUANT_SCALE)
            if pp64_enabled or local64_enabled or ksg64_enabled or gs64_enabled or gs32_enabled or gs32_d1_enabled:
                projection = _read_array(
                    stream, np.int8, 16 * (
                        GS_LOCALPAIR32_D1_WIDTH if gs32_d1_enabled
                        else GS_LOCALPAIR32_WIDTH if gs32_enabled
                        else PP3WIDE64_WIDTH))
                _copy_parameter(
                    model.pp3wide.projection.weight, projection,
                    HIDDEN_WEIGHT_SCALE)

        for stack in model.layer_stacks:
            expected_network_hash = (
                NETWORK_HASH_GS_LOCALPAIR32_D1 if gs32_d1_enabled
                else NETWORK_HASH_GS_LOCALPAIR32 if gs32_enabled
                else NETWORK_HASH_GS_LOCALPAIR64 if gs64_enabled
                else NETWORK_HASH_KSG_LOCALPAIR64 if ksg64_enabled
                else NETWORK_HASH_LOCALPAIR64 if local64_enabled
                else NETWORK_HASH_PP3WIDE64 if pp64_enabled
                else NETWORK_HASH_PP3WIDE if pp_enabled else NETWORK_HASH)
            if _read_u32(stream) != expected_network_hash:
                raise ValueError("HalfKA_HM2 simple network hash mismatch")
            _read_fc(stream, stack.fc0)
            _read_fc(stream, stack.fc1)
            _read_fc(stream, stack.fc2)
        if stream.read(1):
            raise ValueError("trailing bytes in simple nn.bin")
    model.eval()
    return model


def validate_roundtrip(blob, ft_compression="none"):
    stream = io.BytesIO(blob)
    assert _read_u32(stream) == VERSION
    assert _read_u32(stream) == OUTER_HASH
    n = _read_u32(stream)
    description = stream.read(n).decode("utf-8")
    assert description in (
        DESCRIPTION, DESCRIPTION_PP3WIDE, DESCRIPTION_PP3WIDE64,
        DESCRIPTION_LOCALPAIR64, DESCRIPTION_KSG_LOCALPAIR64,
        DESCRIPTION_GS_LOCALPAIR64, DESCRIPTION_GS_LOCALPAIR32,
        DESCRIPTION_GS_LOCALPAIR32_D1)
    pp_enabled = description == DESCRIPTION_PP3WIDE
    pp64_enabled = description == DESCRIPTION_PP3WIDE64
    local64_enabled = description == DESCRIPTION_LOCALPAIR64
    ksg64_enabled = description == DESCRIPTION_KSG_LOCALPAIR64
    gs64_enabled = description == DESCRIPTION_GS_LOCALPAIR64
    gs32_enabled = description == DESCRIPTION_GS_LOCALPAIR32
    gs32_d1_enabled = description == DESCRIPTION_GS_LOCALPAIR32_D1
    assert _read_u32(stream) == (
        FT_HASH_GS_LOCALPAIR32_D1 if gs32_d1_enabled
        else FT_HASH_GS_LOCALPAIR32 if gs32_enabled
        else FT_HASH_GS_LOCALPAIR64 if gs64_enabled
        else FT_HASH_KSG_LOCALPAIR64 if ksg64_enabled
        else FT_HASH_LOCALPAIR64 if local64_enabled
        else FT_HASH_PP3WIDE64 if pp64_enabled
        else FT_HASH_PP3WIDE if pp_enabled else FT_HASH)
    if ft_compression == "none":
        ft_bytes = (FT_WIDTH + FT_INPUTS * FT_WIDTH) * 2
        if len(stream.read(ft_bytes)) != ft_bytes:
            raise EOFError
    elif ft_compression == "leb128":
        _read_leb(stream, FT_WIDTH)
        _read_leb(stream, FT_INPUTS * FT_WIDTH)
    else:
        raise ValueError(f"unsupported FT compression: {ft_compression}")
    if pp_enabled or pp64_enabled or local64_enabled or ksg64_enabled or gs64_enabled or gs32_enabled or gs32_d1_enabled:
        pp_bytes = (
            GS_LOCALPAIR32_D1_FEATURES if gs32_d1_enabled
            else GS_LOCALPAIR32_FEATURES if gs32_enabled
            else GS_LOCALPAIR64_FEATURES if gs64_enabled
            else KSG_LOCALPAIR64_FEATURES if ksg64_enabled
            else LOCALPAIR64_FEATURES if local64_enabled
            else PP3WIDE_FEATURES) * (
            GS_LOCALPAIR32_D1_WIDTH if gs32_d1_enabled
            else GS_LOCALPAIR32_WIDTH if gs32_enabled
            else GS_LOCALPAIR64_WIDTH if gs64_enabled
            else KSG_LOCALPAIR64_WIDTH if ksg64_enabled
            else LOCALPAIR64_WIDTH if local64_enabled
            else PP3WIDE64_WIDTH if pp64_enabled else FT_WIDTH)
        if len(stream.read(pp_bytes)) != pp_bytes:
            raise EOFError
        projection_width = (GS_LOCALPAIR32_D1_WIDTH if gs32_d1_enabled
                            else GS_LOCALPAIR32_WIDTH if gs32_enabled
                            else PP3WIDE64_WIDTH)
        if (pp64_enabled or local64_enabled or ksg64_enabled or gs64_enabled or gs32_enabled or gs32_d1_enabled) and len(stream.read(16 * projection_width)) \
                != 16 * projection_width:
            raise EOFError
    # Validate exact tensor order/size for all nine stacks.
    fc_sizes = (
        16 * 4 + 16 * FT_WIDTH,
        32 * 4 + 32 * 32,
        1 * 4 + 1 * 32,
    )
    for _ in range(LAYER_STACKS):
        assert _read_u32(stream) == (
            NETWORK_HASH_GS_LOCALPAIR32_D1 if gs32_d1_enabled
            else NETWORK_HASH_GS_LOCALPAIR32 if gs32_enabled
            else NETWORK_HASH_GS_LOCALPAIR64 if gs64_enabled
            else NETWORK_HASH_KSG_LOCALPAIR64 if ksg64_enabled
            else NETWORK_HASH_LOCALPAIR64 if local64_enabled
            else NETWORK_HASH_PP3WIDE64 if pp64_enabled
            else NETWORK_HASH_PP3WIDE if pp_enabled else NETWORK_HASH)
        for size in fc_sizes:
            if len(stream.read(size)) != size:
                raise EOFError
    if stream.read(1):
        raise ValueError("trailing bytes in simple nn.bin")
    return True


def main():
    parser = argparse.ArgumentParser(
        description=("Convert HalfKA_HM2 simple networks between .ckpt, .pt "
                     "and .nnue/.bin."))
    parser.add_argument(
        "--ft_compression", "--ft-compression",
        choices=("none", "leb128"), default="none",
        help=("Compression for FT bias/weights. The HalfKA_HM2 simple "
              "default is 'none' so nn.bin remains directly inspectable."))
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()
    feature_set = features.get_feature_set_from_name(FEATURE_NAME)
    source = args.source
    output = args.output
    if source.endswith(".ckpt"):
        model = SimpleHalfKAHM2NNUE.load_from_checkpoint(
            source, feature_set=feature_set, map_location="cpu")
    elif source.endswith(".pt"):
        saved = torch.load(source, map_location="cpu", weights_only=False)
        if not isinstance(saved, SimpleHalfKAHM2NNUE):
            raise TypeError("simple .pt must contain SimpleHalfKAHM2NNUE")
        model = saved
        model.set_feature_set(feature_set)
    elif source.endswith((".nnue", ".bin")):
        model = deserialize_model(source, feature_set)
    else:
        raise ValueError("source must be .ckpt, .pt, .nnue or .bin")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output.endswith(".pt"):
        torch.save(model.eval(), output_path)
        print(f"wrote {output}")
    elif output.endswith((".nnue", ".bin")):
        blob = serialize_model(
            model.eval(), output, ft_compression=args.ft_compression)
        validate_roundtrip(blob, ft_compression=args.ft_compression)
        print(f"wrote {output}: {len(blob):,} bytes")
        print(f"FT compression: {args.ft_compression}")
        pp_type = getattr(model, "simple_local_pair_feature", "off")
        print(DESCRIPTION_GS_LOCALPAIR32_D1
              if pp_type == GS_LOCALPAIR32_D1_TYPE
              else DESCRIPTION_GS_LOCALPAIR32 if pp_type == GS_LOCALPAIR32_TYPE
              else DESCRIPTION_GS_LOCALPAIR64 if pp_type == GS_LOCALPAIR64_TYPE
              else DESCRIPTION_KSG_LOCALPAIR64 if pp_type == KSG_LOCALPAIR64_TYPE
              else DESCRIPTION_LOCALPAIR64 if pp_type == LOCALPAIR64_TYPE
              else DESCRIPTION_PP3WIDE64 if pp_type == PP3WIDE64_TYPE
              else DESCRIPTION_PP3WIDE if pp_type == PP3WIDE_TYPE
              else DESCRIPTION)
    else:
        raise ValueError("output must be .pt, .nnue or .bin")


if __name__ == "__main__":
    main()

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


VERSION = 0x7AF32F16
OUTER_HASH = 0x74517600
FT_HASH = 0x7F234CB8 ^ FT_WIDTH
NETWORK_HASH = 0x6333718A ^ 0x484D3202
DESCRIPTION = (
    "ModelType=SFNNWithoutPsqt;"
    "Features=HalfKA_hm2_NoDG(Friend)[73305->1536x2],"
    "Network=SFNN-1536-HalfKAHM2-NoDG-v2{LayerStack=9}"
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
    _u32(buf, VERSION)
    _u32(buf, OUTER_HASH)
    desc = DESCRIPTION.encode("utf-8")
    _u32(buf, len(desc))
    buf.extend(desc)
    _u32(buf, FT_HASH)
    _write_tensor(
        buf, model.input.bias.mul(Q_ONE).round().to(torch.int16),
        ft_compression)
    _write_tensor(
        buf, model.input.weight.mul(Q_ONE).round().to(torch.int16),
        ft_compression)
    for stack in model.layer_stacks:
        _u32(buf, NETWORK_HASH)
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
        if description != DESCRIPTION:
            raise ValueError(
                "HalfKA_HM2 simple architecture description mismatch: "
                f"{description!r}")
        if _read_u32(stream) != FT_HASH:
            raise ValueError("HalfKA_HM2 simple FT hash mismatch")

        model = SimpleHalfKAHM2NNUE(feature_set=feature_set)
        bias = _read_ft_tensor(stream, FT_WIDTH)
        weight = _read_ft_tensor(stream, FT_INPUTS * FT_WIDTH)
        _copy_parameter(model.input.bias, bias, Q_ONE)
        _copy_parameter(model.input.weight, weight, Q_ONE)

        for stack in model.layer_stacks:
            if _read_u32(stream) != NETWORK_HASH:
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
    assert stream.read(n).decode("utf-8") == DESCRIPTION
    assert _read_u32(stream) == FT_HASH
    if ft_compression == "none":
        ft_bytes = (FT_WIDTH + FT_INPUTS * FT_WIDTH) * 2
        if len(stream.read(ft_bytes)) != ft_bytes:
            raise EOFError
    elif ft_compression == "leb128":
        _read_leb(stream, FT_WIDTH)
        _read_leb(stream, FT_INPUTS * FT_WIDTH)
    else:
        raise ValueError(f"unsupported FT compression: {ft_compression}")
    # Validate exact tensor order/size for all nine stacks.
    fc_sizes = (
        16 * 4 + 16 * FT_WIDTH,
        32 * 4 + 32 * 32,
        1 * 4 + 1 * 32,
    )
    for _ in range(LAYER_STACKS):
        assert _read_u32(stream) == NETWORK_HASH
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
        print(DESCRIPTION)
    else:
        raise ValueError("output must be .pt, .nnue or .bin")


if __name__ == "__main__":
    main()

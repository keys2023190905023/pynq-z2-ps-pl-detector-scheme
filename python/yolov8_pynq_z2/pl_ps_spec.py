from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _as_int_tuple(values: Iterable[int]) -> Tuple[int, ...]:
    return tuple(int(v) for v in values)


def _as_4d_weights(values: Sequence[Sequence[Sequence[Sequence[int]]]]) -> Tuple[Tuple[Tuple[Tuple[int, ...], ...], ...], ...]:
    return tuple(
        tuple(
            tuple(tuple(int(weight) for weight in row) for row in in_channel)
            for in_channel in out_channel
        )
        for out_channel in values
    )


def _check_signed(name: str, value: int, bits: int) -> int:
    lo = -(1 << (bits - 1))
    hi = (1 << (bits - 1)) - 1
    value = int(value)
    if value < lo or value > hi:
        raise ValueError(f"{name}={value} is outside signed {bits}-bit range [{lo}, {hi}]")
    return value


@dataclass(frozen=True)
class ConvLayerSpec:
    name: str
    in_channels: int
    out_channels: int
    weights: Tuple[Tuple[Tuple[Tuple[int, ...], ...], ...], ...]
    bias: Tuple[int, ...]
    quant_scale: int = 1
    quant_shift: int = 0
    input_zp: int = 0
    output_zp: int = 0
    stride: int = 1
    padding: int = 1
    activation: str = "relu"
    description: str = ""

    def __post_init__(self) -> None:
        in_channels = int(self.in_channels)
        out_channels = int(self.out_channels)
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        if int(self.stride) != 1:
            raise ValueError("this project currently supports stride=1 only")
        if int(self.padding) != 1:
            raise ValueError("this project currently supports padding=1 only")
        if self.activation not in {"identity", "relu"}:
            raise ValueError("activation must be 'identity' or 'relu'")

        weights = _as_4d_weights(self.weights)
        if len(weights) != out_channels:
            raise ValueError(f"weights outer dimension must equal out_channels={out_channels}, got {len(weights)}")
        for oc, out_channel in enumerate(weights):
            if len(out_channel) != in_channels:
                raise ValueError(f"weights[{oc}] must contain {in_channels} input channels, got {len(out_channel)}")
            for ic, kernel in enumerate(out_channel):
                if len(kernel) != 3 or any(len(row) != 3 for row in kernel):
                    raise ValueError(f"weights[{oc}][{ic}] must be a 3x3 kernel")
                for ky, row in enumerate(kernel):
                    for kx, weight in enumerate(row):
                        _check_signed(f"weights[{oc}][{ic}][{ky}][{kx}]", weight, 8)

        bias = _as_int_tuple(self.bias)
        if len(bias) != out_channels:
            raise ValueError(f"bias must contain {out_channels} values, got {len(bias)}")
        for oc, value in enumerate(bias):
            _check_signed(f"bias[{oc}]", value, 32)

        _check_signed("quant_scale", self.quant_scale, 16)
        if not 0 <= int(self.quant_shift) <= 31:
            raise ValueError("quant_shift must be in [0, 31]")
        _check_signed("input_zp", self.input_zp, 8)
        _check_signed("output_zp", self.output_zp, 8)

        object.__setattr__(self, "in_channels", in_channels)
        object.__setattr__(self, "out_channels", out_channels)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "bias", bias)
        object.__setattr__(self, "quant_scale", int(self.quant_scale))
        object.__setattr__(self, "quant_shift", int(self.quant_shift))
        object.__setattr__(self, "input_zp", int(self.input_zp))
        object.__setattr__(self, "output_zp", int(self.output_zp))
        object.__setattr__(self, "stride", int(self.stride))
        object.__setattr__(self, "padding", int(self.padding))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "weights": [[[list(row) for row in kernel] for kernel in out_channel] for out_channel in self.weights],
            "bias": list(self.bias),
            "quant_scale": self.quant_scale,
            "quant_shift": self.quant_shift,
            "input_zp": self.input_zp,
            "output_zp": self.output_zp,
            "stride": self.stride,
            "padding": self.padding,
            "activation": self.activation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConvLayerSpec":
        return cls(
            name=str(data["name"]),
            description=str(data.get("description", "")),
            in_channels=int(data["in_channels"]),
            out_channels=int(data["out_channels"]),
            weights=data["weights"],
            bias=data["bias"],
            quant_scale=int(data.get("quant_scale", 1)),
            quant_shift=int(data.get("quant_shift", 0)),
            input_zp=int(data.get("input_zp", 0)),
            output_zp=int(data.get("output_zp", 0)),
            stride=int(data.get("stride", 1)),
            padding=int(data.get("padding", 1)),
            activation=str(data.get("activation", "relu")),
        )


@dataclass(frozen=True)
class ModelSpec:
    name: str
    input_channels: int
    layers: Tuple[ConvLayerSpec, ...]
    class_names: Tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        layers = tuple(self.layers)
        if int(self.input_channels) <= 0:
            raise ValueError("input_channels must be positive")
        prev_channels = int(self.input_channels)
        for idx, layer in enumerate(layers):
            if layer.in_channels != prev_channels:
                raise ValueError(
                    f"layer {idx} '{layer.name}' expects {layer.in_channels} input channels, "
                    f"but previous layer produces {prev_channels}"
                )
            prev_channels = layer.out_channels
        object.__setattr__(self, "input_channels", int(self.input_channels))
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "class_names", tuple(str(name) for name in self.class_names))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_channels": self.input_channels,
            "class_names": list(self.class_names),
            "layers": [layer.to_dict() for layer in self.layers],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelSpec":
        return cls(
            name=str(data["name"]),
            description=str(data.get("description", "")),
            input_channels=int(data["input_channels"]),
            class_names=tuple(str(name) for name in data.get("class_names", ())),
            layers=tuple(ConvLayerSpec.from_dict(layer) for layer in data["layers"]),
        )


@dataclass(frozen=True)
class HardwareConfig:
    output_channel_parallelism: int = 4
    input_channel_tile: int = 4
    scratch_buffers: int = 2
    line_buffer_max_width: int = 640

    def __post_init__(self) -> None:
        if int(self.output_channel_parallelism) <= 0:
            raise ValueError("output_channel_parallelism must be positive")
        if int(self.input_channel_tile) <= 0:
            raise ValueError("input_channel_tile must be positive")
        if int(self.scratch_buffers) <= 0:
            raise ValueError("scratch_buffers must be positive")
        if int(self.line_buffer_max_width) < 3:
            raise ValueError("line_buffer_max_width must be at least 3")
        object.__setattr__(self, "output_channel_parallelism", int(self.output_channel_parallelism))
        object.__setattr__(self, "input_channel_tile", int(self.input_channel_tile))
        object.__setattr__(self, "scratch_buffers", int(self.scratch_buffers))
        object.__setattr__(self, "line_buffer_max_width", int(self.line_buffer_max_width))


@dataclass(frozen=True)
class ExecutionStep:
    layer_name: str
    output_tile_index: int
    input_tile_index: int
    output_channel_start: int
    output_channel_count: int
    input_channel_start: int
    input_channel_count: int
    clear_accumulator: bool
    write_output: bool
    apply_activation: bool
    scratch_buffer_index: int
    description: str = ""


def make_demo_model_spec() -> ModelSpec:
    stem_weights = []
    for oc in range(4):
        out_channel = []
        for ic in range(3):
            if oc == ic:
                kernel = (
                    (0, 0, 0),
                    (0, 1, 0),
                    (0, 0, 0),
                )
            else:
                kernel = (
                    (0, 0, 0),
                    (0, 0, 0),
                    (0, 0, 0),
                )
            out_channel.append(kernel)
        stem_weights.append(out_channel)

    head_weights = []
    for oc in range(3):
        out_channel = []
        for ic in range(4):
            center = 1 if (ic % 3) == oc else 0
            out_channel.append(
                (
                    (0, 0, 0),
                    (0, center, 0),
                    (0, 0, 0),
                )
            )
        head_weights.append(out_channel)

    return ModelSpec(
        name="pl_ps_tiny_demo",
        description="Two-layer multi-channel demo graph for the PL-conv / PS-scheduler project path.",
        input_channels=3,
        class_names=("helmet", "person", "vest"),
        layers=(
            ConvLayerSpec(
                name="stem_conv",
                description="RGB stem projection into 4 channels",
                in_channels=3,
                out_channels=4,
                weights=tuple(stem_weights),
                bias=(0, 0, 0, 0),
                activation="relu",
            ),
            ConvLayerSpec(
                name="head_logits",
                description="Demo 3-channel detector head logits",
                in_channels=4,
                out_channels=3,
                weights=tuple(head_weights),
                bias=(0, 0, 0),
                activation="identity",
            ),
        ),
    )


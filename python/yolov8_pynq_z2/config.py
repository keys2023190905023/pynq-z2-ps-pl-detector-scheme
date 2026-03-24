from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple


def _as_tuple(values: Iterable[int]) -> Tuple[int, ...]:
    return tuple(int(v) for v in values)


def _check_signed(name: str, value: int, bits: int) -> int:
    lo = -(1 << (bits - 1))
    hi = (1 << (bits - 1)) - 1
    value = int(value)
    if value < lo or value > hi:
        raise ValueError(f"{name}={value} is outside signed {bits}-bit range [{lo}, {hi}]")
    return value


@dataclass(frozen=True)
class QuantizedConvConfig:
    name: str
    weights: Tuple[int, ...]
    bias: int = 0
    quant_scale: int = 1
    quant_shift: int = 0
    input_zp: int = 0
    output_zp: int = 0
    description: str = ""

    def __post_init__(self) -> None:
        weights = _as_tuple(self.weights)
        if len(weights) != 9:
            raise ValueError(f"weights must contain exactly 9 values, got {len(weights)}")
        for idx, weight in enumerate(weights):
            _check_signed(f"weights[{idx}]", weight, 8)
        _check_signed("bias", self.bias, 32)
        _check_signed("quant_scale", self.quant_scale, 16)
        if not 0 <= int(self.quant_shift) <= 31:
            raise ValueError("quant_shift must be in [0, 31]")
        _check_signed("input_zp", self.input_zp, 8)
        _check_signed("output_zp", self.output_zp, 8)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "quant_shift", int(self.quant_shift))
        object.__setattr__(self, "bias", int(self.bias))
        object.__setattr__(self, "quant_scale", int(self.quant_scale))
        object.__setattr__(self, "input_zp", int(self.input_zp))
        object.__setattr__(self, "output_zp", int(self.output_zp))

    @property
    def kernel(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]:
        return (
            (self.weights[0], self.weights[1], self.weights[2]),
            (self.weights[3], self.weights[4], self.weights[5]),
            (self.weights[6], self.weights[7], self.weights[8]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "weights": list(self.weights),
            "bias": self.bias,
            "quant_scale": self.quant_scale,
            "quant_shift": self.quant_shift,
            "input_zp": self.input_zp,
            "output_zp": self.output_zp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuantizedConvConfig":
        return cls(
            name=str(data["name"]),
            description=str(data.get("description", "")),
            weights=_as_tuple(data["weights"]),
            bias=int(data.get("bias", 0)),
            quant_scale=int(data.get("quant_scale", 1)),
            quant_shift=int(data.get("quant_shift", 0)),
            input_zp=int(data.get("input_zp", 0)),
            output_zp=int(data.get("output_zp", 0)),
        )

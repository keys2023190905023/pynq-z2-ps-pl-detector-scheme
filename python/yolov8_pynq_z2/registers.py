from __future__ import annotations

from typing import Iterable, Tuple

from .config import QuantizedConvConfig


CTRL_OFFSET = 0x00
IMAGE_SHAPE_OFFSET = 0x04
BIAS_OFFSET = 0x08
QUANT_OFFSET = 0x0C
WEIGHTS0_OFFSET = 0x10
WEIGHTS1_OFFSET = 0x14
WEIGHTS2_OFFSET = 0x18
SCRATCH_OFFSET = 0x1C

CTRL_START_BIT = 0
CTRL_SOFT_RESET_BIT = 1
CTRL_RELU_ENABLE_BIT = 4
CTRL_CHANNEL_FIRST_BIT = 5
CTRL_CHANNEL_LAST_BIT = 6
CTRL_INPUT_ZP_SHIFT = 8
CTRL_OUTPUT_ZP_SHIFT = 16

MICROKERNEL_INPUT_WIDTH = 5
MICROKERNEL_VALID_OUTPUT_WIDTH = 3


def _u8(value: int) -> int:
    return int(value) & 0xFF


def _u16(value: int) -> int:
    return int(value) & 0xFFFF


def _u32(value: int) -> int:
    return int(value) & 0xFFFFFFFF


def pack_ctrl(
    *,
    start: bool,
    soft_reset: bool,
    input_zp: int = 0,
    output_zp: int = 0,
    relu_enable: bool = False,
    channel_first: bool = True,
    channel_last: bool = True,
) -> int:
    ctrl = 0
    ctrl |= (1 if start else 0) << CTRL_START_BIT
    ctrl |= (1 if soft_reset else 0) << CTRL_SOFT_RESET_BIT
    ctrl |= (1 if relu_enable else 0) << CTRL_RELU_ENABLE_BIT
    ctrl |= (1 if channel_first else 0) << CTRL_CHANNEL_FIRST_BIT
    ctrl |= (1 if channel_last else 0) << CTRL_CHANNEL_LAST_BIT
    ctrl |= _u8(input_zp) << CTRL_INPUT_ZP_SHIFT
    ctrl |= _u8(output_zp) << CTRL_OUTPUT_ZP_SHIFT
    return _u32(ctrl)


def pack_image_shape(width: int, height: int) -> int:
    if not 0 <= int(width) <= 0xFFFF:
        raise ValueError("width must fit in 16 bits")
    if not 0 <= int(height) <= 0xFFFF:
        raise ValueError("height must fit in 16 bits")
    return _u32((_u16(height) << 16) | _u16(width))


def pack_quant(quant_scale: int, quant_shift: int) -> int:
    if not 0 <= int(quant_shift) <= 0xFF:
        raise ValueError("quant_shift must fit in 8 bits")
    return _u32((_u8(quant_shift) << 16) | _u16(quant_scale))


def pack_weights(weights: Iterable[int]) -> Tuple[int, int, int]:
    values = tuple(int(v) for v in weights)
    if len(values) != 9:
        raise ValueError("weights must contain exactly 9 signed bytes")
    return (
        _u32(_u8(values[0]) | (_u8(values[1]) << 8) | (_u8(values[2]) << 16) | (_u8(values[3]) << 24)),
        _u32(_u8(values[4]) | (_u8(values[5]) << 8) | (_u8(values[6]) << 16) | (_u8(values[7]) << 24)),
        _u32(_u8(values[8])),
    )


def pack_config(config: QuantizedConvConfig, *, width: int, height: int, start: bool, soft_reset: bool) -> Tuple[int, ...]:
    weights0, weights1, weights2 = pack_weights(config.weights)
    return (
        pack_ctrl(start=start, soft_reset=soft_reset, input_zp=config.input_zp, output_zp=config.output_zp),
        pack_image_shape(width, height),
        _u32(config.bias),
        pack_quant(config.quant_scale, config.quant_shift),
        weights0,
        weights1,
        weights2,
    )

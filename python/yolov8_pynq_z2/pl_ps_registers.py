from __future__ import annotations


CTRL_OFFSET = 0x00
IMAGE_SHAPE_OFFSET = 0x04
CHANNEL_CONFIG_OFFSET = 0x08
TILE_CONFIG_OFFSET = 0x0C
QUANT_CONFIG_OFFSET = 0x10
DMA_IFM_ADDR_OFFSET = 0x14
DMA_OFM_ADDR_OFFSET = 0x18
DMA_WEIGHT_ADDR_OFFSET = 0x1C
DMA_BIAS_ADDR_OFFSET = 0x20
DMA_ACCUM_ADDR_OFFSET = 0x24

CTRL_START_BIT = 0
CTRL_SOFT_RESET_BIT = 1
CTRL_CLEAR_ACCUM_BIT = 2
CTRL_WRITE_OUTPUT_BIT = 3
CTRL_ACT_RELU_BIT = 4


def _u8(value: int) -> int:
    return int(value) & 0xFF


def _u16(value: int) -> int:
    return int(value) & 0xFFFF


def _u32(value: int) -> int:
    return int(value) & 0xFFFFFFFF


def pack_ctrl(*, start: bool, soft_reset: bool, clear_accumulator: bool, write_output: bool, relu_enable: bool) -> int:
    value = 0
    value |= (1 if start else 0) << CTRL_START_BIT
    value |= (1 if soft_reset else 0) << CTRL_SOFT_RESET_BIT
    value |= (1 if clear_accumulator else 0) << CTRL_CLEAR_ACCUM_BIT
    value |= (1 if write_output else 0) << CTRL_WRITE_OUTPUT_BIT
    value |= (1 if relu_enable else 0) << CTRL_ACT_RELU_BIT
    return _u32(value)


def pack_image_shape(width: int, height: int) -> int:
    return _u32((_u16(height) << 16) | _u16(width))


def pack_channel_config(in_channels: int, out_channels: int) -> int:
    return _u32((_u16(out_channels) << 16) | _u16(in_channels))


def pack_tile_config(input_channel_start: int, input_channel_count: int, output_channel_start: int, output_channel_count: int) -> int:
    return _u32(
        _u8(input_channel_start)
        | (_u8(input_channel_count) << 8)
        | (_u8(output_channel_start) << 16)
        | (_u8(output_channel_count) << 24)
    )


def pack_quant_config(quant_scale: int, quant_shift: int, input_zp: int, output_zp: int) -> int:
    return _u32(_u8(input_zp) | (_u8(output_zp) << 8) | (_u8(quant_shift) << 16) | (_u8(quant_scale) << 24))

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .pl_ps_registers import (
    CHANNEL_CONFIG_OFFSET,
    CTRL_OFFSET,
    DMA_ACCUM_ADDR_OFFSET,
    DMA_BIAS_ADDR_OFFSET,
    DMA_IFM_ADDR_OFFSET,
    DMA_OFM_ADDR_OFFSET,
    DMA_WEIGHT_ADDR_OFFSET,
    IMAGE_SHAPE_OFFSET,
    QUANT_CONFIG_OFFSET,
    TILE_CONFIG_OFFSET,
    pack_channel_config,
    pack_ctrl,
    pack_image_shape,
    pack_quant_config,
    pack_tile_config,
)
from .pl_ps_scheduler import build_layer_execution_steps
from .pl_ps_spec import ConvLayerSpec, ExecutionStep, HardwareConfig, ModelSpec


def align_up(value: int, alignment: int) -> int:
    value = int(value)
    alignment = int(alignment)
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return ((value + alignment - 1) // alignment) * alignment


def feature_map_bytes(channels: int, height: int, width: int) -> int:
    return int(channels) * int(height) * int(width)


def scratch_buffer_bytes(output_channel_parallelism: int, height: int, width: int) -> int:
    return int(output_channel_parallelism) * int(height) * int(width) * 4


def _pack_i8(value: int) -> int:
    return int(value) & 0xFF


def pack_weight_tile_bytes(layer: ConvLayerSpec, step: ExecutionStep) -> bytes:
    blob = bytearray()
    for oc in range(step.output_channel_start, step.output_channel_start + step.output_channel_count):
        for ic in range(step.input_channel_start, step.input_channel_start + step.input_channel_count):
            kernel = layer.weights[oc][ic]
            for row in kernel:
                for value in row:
                    blob.append(_pack_i8(value))
    return bytes(blob)


def pack_bias_tile_bytes(layer: ConvLayerSpec, step: ExecutionStep) -> bytes:
    blob = bytearray()
    for oc in range(step.output_channel_start, step.output_channel_start + step.output_channel_count):
        blob.extend(int(layer.bias[oc]).to_bytes(4, byteorder="little", signed=True))
    return bytes(blob)


@dataclass(frozen=True)
class CompiledStep:
    step: ExecutionStep
    ifm_addr: int
    ofm_addr: int
    accum_addr: int
    weight_addr: int
    bias_addr: int
    weight_size: int
    bias_size: int
    ctrl_idle: int
    ctrl_start: int
    image_shape_word: int
    channel_config_word: int
    tile_config_word: int
    quant_config_word: int

    def register_sequence(self) -> Tuple[Tuple[int, int], ...]:
        return (
            (IMAGE_SHAPE_OFFSET, self.image_shape_word),
            (CHANNEL_CONFIG_OFFSET, self.channel_config_word),
            (TILE_CONFIG_OFFSET, self.tile_config_word),
            (QUANT_CONFIG_OFFSET, self.quant_config_word),
            (DMA_IFM_ADDR_OFFSET, self.ifm_addr),
            (DMA_OFM_ADDR_OFFSET, self.ofm_addr),
            (DMA_WEIGHT_ADDR_OFFSET, self.weight_addr),
            (DMA_BIAS_ADDR_OFFSET, self.bias_addr),
            (DMA_ACCUM_ADDR_OFFSET, self.accum_addr),
            (CTRL_OFFSET, self.ctrl_idle),
            (CTRL_OFFSET, self.ctrl_start),
            (CTRL_OFFSET, self.ctrl_idle),
        )

    def to_dict(self) -> Dict[str, int | str | bool]:
        return {
            "layer_name": self.step.layer_name,
            "description": self.step.description,
            "output_tile_index": self.step.output_tile_index,
            "input_tile_index": self.step.input_tile_index,
            "output_channel_start": self.step.output_channel_start,
            "output_channel_count": self.step.output_channel_count,
            "input_channel_start": self.step.input_channel_start,
            "input_channel_count": self.step.input_channel_count,
            "clear_accumulator": self.step.clear_accumulator,
            "write_output": self.step.write_output,
            "apply_activation": self.step.apply_activation,
            "scratch_buffer_index": self.step.scratch_buffer_index,
            "ifm_addr": self.ifm_addr,
            "ofm_addr": self.ofm_addr,
            "accum_addr": self.accum_addr,
            "weight_addr": self.weight_addr,
            "bias_addr": self.bias_addr,
            "weight_size": self.weight_size,
            "bias_size": self.bias_size,
            "ctrl_idle": self.ctrl_idle,
            "ctrl_start": self.ctrl_start,
            "image_shape_word": self.image_shape_word,
            "channel_config_word": self.channel_config_word,
            "tile_config_word": self.tile_config_word,
            "quant_config_word": self.quant_config_word,
        }


@dataclass(frozen=True)
class CompiledModelProgram:
    model_name: str
    width: int
    height: int
    feature_map_addrs: Tuple[int, int]
    feature_map_bytes: int
    scratch_base_addr: int
    scratch_buffer_bytes: int
    weight_base_addr: int
    bias_base_addr: int
    weight_blob: bytes
    bias_blob: bytes
    steps: Tuple[CompiledStep, ...]
    final_output_addr: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_name": self.model_name,
            "width": self.width,
            "height": self.height,
            "feature_map_addrs": list(self.feature_map_addrs),
            "feature_map_bytes": self.feature_map_bytes,
            "scratch_base_addr": self.scratch_base_addr,
            "scratch_buffer_bytes": self.scratch_buffer_bytes,
            "weight_base_addr": self.weight_base_addr,
            "bias_base_addr": self.bias_base_addr,
            "weight_blob_size": len(self.weight_blob),
            "bias_blob_size": len(self.bias_blob),
            "final_output_addr": self.final_output_addr,
            "steps": [step.to_dict() for step in self.steps],
        }


def _feature_map_pool_addrs(
    model: ModelSpec,
    height: int,
    width: int,
    *,
    base_addr: int,
    alignment: int,
) -> Tuple[Tuple[int, int], int]:
    max_channels = max([model.input_channels] + [layer.out_channels for layer in model.layers])
    buffer_bytes = align_up(feature_map_bytes(max_channels, height, width), alignment)
    return (int(base_addr), int(base_addr) + buffer_bytes), buffer_bytes


def _append_aligned(blob: bytearray, payload: bytes, alignment: int) -> int:
    offset = align_up(len(blob), alignment)
    if len(blob) < offset:
        blob.extend(b"\x00" * (offset - len(blob)))
    blob.extend(payload)
    return offset


def build_compiled_model_program(
    model: ModelSpec,
    hw: HardwareConfig,
    *,
    width: int,
    height: int,
    feature_base_addr: int = 0x1000_0000,
    weight_base_addr: int = 0x1100_0000,
    bias_base_addr: int = 0x1200_0000,
    scratch_base_addr: int = 0x1300_0000,
    alignment: int = 64,
) -> CompiledModelProgram:
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    feature_map_addrs, feature_bytes = _feature_map_pool_addrs(
        model,
        height,
        width,
        base_addr=feature_base_addr,
        alignment=alignment,
    )
    scratch_bytes = align_up(scratch_buffer_bytes(hw.output_channel_parallelism, height, width), alignment)
    weight_blob = bytearray()
    bias_blob = bytearray()
    compiled_steps: List[CompiledStep] = []

    current_ifm_addr, current_ofm_addr = feature_map_addrs
    for layer in model.layers:
        for step in build_layer_execution_steps(layer, hw):
            weight_payload = pack_weight_tile_bytes(layer, step)
            weight_offset = _append_aligned(weight_blob, weight_payload, alignment)
            if step.write_output:
                bias_payload = pack_bias_tile_bytes(layer, step)
                bias_offset = _append_aligned(bias_blob, bias_payload, alignment)
                bias_addr = int(bias_base_addr) + bias_offset
                bias_size = len(bias_payload)
            else:
                bias_addr = 0
                bias_size = 0

            ctrl_kwargs = dict(
                soft_reset=False,
                clear_accumulator=step.clear_accumulator,
                write_output=step.write_output,
                relu_enable=step.apply_activation,
            )
            compiled_steps.append(
                CompiledStep(
                    step=step,
                    ifm_addr=current_ifm_addr,
                    ofm_addr=current_ofm_addr,
                    accum_addr=int(scratch_base_addr) + step.scratch_buffer_index * scratch_bytes,
                    weight_addr=int(weight_base_addr) + weight_offset,
                    bias_addr=bias_addr,
                    weight_size=len(weight_payload),
                    bias_size=bias_size,
                    ctrl_idle=pack_ctrl(start=False, **ctrl_kwargs),
                    ctrl_start=pack_ctrl(start=True, **ctrl_kwargs),
                    image_shape_word=pack_image_shape(width=width, height=height),
                    channel_config_word=pack_channel_config(layer.in_channels, layer.out_channels),
                    tile_config_word=pack_tile_config(
                        input_channel_start=step.input_channel_start,
                        input_channel_count=step.input_channel_count,
                        output_channel_start=step.output_channel_start,
                        output_channel_count=step.output_channel_count,
                    ),
                    quant_config_word=pack_quant_config(
                        quant_scale=layer.quant_scale,
                        quant_shift=layer.quant_shift,
                        input_zp=layer.input_zp,
                        output_zp=layer.output_zp,
                    ),
                )
            )
        current_ifm_addr, current_ofm_addr = current_ofm_addr, current_ifm_addr

    final_output_addr = current_ifm_addr
    return CompiledModelProgram(
        model_name=model.name,
        width=width,
        height=height,
        feature_map_addrs=feature_map_addrs,
        feature_map_bytes=feature_bytes,
        scratch_base_addr=int(scratch_base_addr),
        scratch_buffer_bytes=scratch_bytes,
        weight_base_addr=int(weight_base_addr),
        bias_base_addr=int(bias_base_addr),
        weight_blob=bytes(weight_blob),
        bias_blob=bytes(bias_blob),
        steps=tuple(compiled_steps),
        final_output_addr=final_output_addr,
    )


def program_compiled_step(mmio: object, compiled_step: CompiledStep) -> None:
    for offset, value in compiled_step.register_sequence():
        mmio.write(offset, value)


def program_compiled_steps(mmio: object, steps: Sequence[CompiledStep]) -> None:
    for compiled_step in steps:
        program_compiled_step(mmio, compiled_step)

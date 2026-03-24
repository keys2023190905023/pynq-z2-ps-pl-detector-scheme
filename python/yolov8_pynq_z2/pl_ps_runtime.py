from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .pl_ps_model import ensure_nchw_int8, quantize_conv_output
from .pl_ps_scheduler import build_layer_execution_steps
from .pl_ps_spec import ConvLayerSpec, ExecutionStep, HardwareConfig, ModelSpec


@dataclass
class LayerRuntimeResult:
    output: np.ndarray
    steps: List[ExecutionStep]
    partial_accumulator_shape: Tuple[int, int, int]


def accumulate_layer_tile(input_tensor: np.ndarray, layer: ConvLayerSpec, step: ExecutionStep, accumulator: np.ndarray) -> None:
    src = ensure_nchw_int8(input_tensor).astype(np.int16)
    padded = np.pad(src, ((0, 0), (1, 1), (1, 1)), mode="constant", constant_values=int(layer.input_zp))
    weights = np.asarray(layer.weights, dtype=np.int16)
    height, width = src.shape[1:]

    tile_acc = accumulator[
        step.output_channel_start : step.output_channel_start + step.output_channel_count,
        :,
        :,
    ]
    if step.clear_accumulator:
        tile_acc.fill(0)

    for local_oc, oc in enumerate(range(step.output_channel_start, step.output_channel_start + step.output_channel_count)):
        for ic in range(step.input_channel_start, step.input_channel_start + step.input_channel_count):
            kernel = weights[oc, ic]
            for y in range(height):
                for x in range(width):
                    tile_acc[local_oc, y, x] += int(np.sum(padded[ic, y : y + 3, x : x + 3] * kernel, dtype=np.int32))


def run_layer_reference(input_tensor: np.ndarray, layer: ConvLayerSpec, hw: HardwareConfig) -> LayerRuntimeResult:
    src = ensure_nchw_int8(input_tensor)
    height, width = src.shape[1:]
    accumulator = np.zeros((layer.out_channels, height, width), dtype=np.int32)
    output = np.zeros((layer.out_channels, height, width), dtype=np.int8)
    steps = build_layer_execution_steps(layer, hw)

    for step in steps:
        accumulate_layer_tile(src, layer, step, accumulator)
        if step.write_output:
            oc_slice = slice(step.output_channel_start, step.output_channel_start + step.output_channel_count)
            quantized = quantize_conv_output(
                accumulator[oc_slice],
                layer,
                bias_override=np.asarray(layer.bias[oc_slice], dtype=np.int64),
            )
            output[oc_slice] = quantized

    return LayerRuntimeResult(output=output, steps=steps, partial_accumulator_shape=accumulator.shape)


def run_model_reference_with_schedule(input_tensor: np.ndarray, model: ModelSpec, hw: HardwareConfig) -> Dict[str, LayerRuntimeResult]:
    results: Dict[str, LayerRuntimeResult] = {}
    current = ensure_nchw_int8(input_tensor)
    for layer in model.layers:
        result = run_layer_reference(current, layer, hw)
        results[layer.name] = result
        current = result.output
    return results

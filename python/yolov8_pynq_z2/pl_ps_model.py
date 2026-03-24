from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .pl_ps_spec import ConvLayerSpec, ModelSpec


def ensure_nchw_int8(tensor: np.ndarray) -> np.ndarray:
    array = np.asarray(tensor)
    if array.ndim != 3:
        raise ValueError(f"expected a 3D CHW tensor, got shape {array.shape}")
    return array.astype(np.int8, copy=False)


def quantize_conv_output(accumulator: np.ndarray, layer: ConvLayerSpec, *, bias_override: np.ndarray | None = None) -> np.ndarray:
    values = accumulator.astype(np.int64)
    if bias_override is None:
        bias = np.asarray(layer.bias, dtype=np.int64).reshape(-1, 1, 1)
    else:
        bias = np.asarray(bias_override, dtype=np.int64).reshape(-1, 1, 1)
    values = values + bias
    values = values * int(layer.quant_scale)
    if layer.quant_shift > 0:
        values = (values + (1 << (layer.quant_shift - 1))) >> int(layer.quant_shift)
    values = values + int(layer.output_zp)
    if layer.activation == "relu":
        values = np.maximum(values, int(layer.output_zp))
    return np.clip(values, -128, 127).astype(np.int8)


def conv2d_same_nchw_reference(input_tensor: np.ndarray, layer: ConvLayerSpec) -> np.ndarray:
    src = ensure_nchw_int8(input_tensor).astype(np.int16)
    in_channels, height, width = src.shape
    if in_channels != layer.in_channels:
        raise ValueError(f"layer '{layer.name}' expects {layer.in_channels} channels, got {in_channels}")

    padded = np.pad(src, ((0, 0), (1, 1), (1, 1)), mode="constant", constant_values=int(layer.input_zp))
    weights = np.asarray(layer.weights, dtype=np.int16)
    accumulator = np.zeros((layer.out_channels, height, width), dtype=np.int32)

    for oc in range(layer.out_channels):
        for ic in range(layer.in_channels):
            kernel = weights[oc, ic]
            for y in range(height):
                for x in range(width):
                    window = padded[ic, y : y + 3, x : x + 3]
                    accumulator[oc, y, x] += int(np.sum(window * kernel, dtype=np.int32))

    return quantize_conv_output(accumulator, layer)


def run_model_reference(input_tensor: np.ndarray, model: ModelSpec) -> Dict[str, np.ndarray]:
    activations: Dict[str, np.ndarray] = {}
    current = ensure_nchw_int8(input_tensor)
    if current.shape[0] != model.input_channels:
        raise ValueError(f"model '{model.name}' expects {model.input_channels} input channels, got {current.shape[0]}")

    for layer in model.layers:
        current = conv2d_same_nchw_reference(current, layer)
        activations[layer.name] = current
    return activations


def make_demo_input(height: int = 32, width: int = 32) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    red = ((xx * 255) // max(width - 1, 1) - 128).astype(np.int8)
    green = ((yy * 255) // max(height - 1, 1) - 128).astype(np.int8)
    blue = np.where(((xx // 4) ^ (yy // 4)) & 1, 64, -32).astype(np.int8)
    return np.stack([red, green, blue], axis=0)


def summarize_tensor(name: str, tensor: np.ndarray) -> str:
    array = np.asarray(tensor)
    return f"{name}: shape={array.shape}, dtype={array.dtype}, min={array.min()}, max={array.max()}, mean={float(array.mean()):.2f}"

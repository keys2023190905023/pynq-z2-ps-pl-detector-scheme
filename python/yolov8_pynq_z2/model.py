from __future__ import annotations

import numpy as np

from .config import QuantizedConvConfig
from .registers import MICROKERNEL_INPUT_WIDTH, MICROKERNEL_VALID_OUTPUT_WIDTH


def ensure_int8_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"expected a 2D grayscale image, got shape {array.shape}")
    return array.astype(np.int8, copy=False)


def to_signed_image(image_u8: np.ndarray) -> np.ndarray:
    image = np.asarray(image_u8, dtype=np.uint8)
    return (image.astype(np.int16) - 128).astype(np.int8)


def to_display_image(image_i8: np.ndarray) -> np.ndarray:
    image = np.asarray(image_i8, dtype=np.int16) + 128
    return np.clip(image, 0, 255).astype(np.uint8)


def quantize_accumulator(acc: np.ndarray, config: QuantizedConvConfig) -> np.ndarray:
    values = acc.astype(np.int64)
    values = (values + int(config.bias)) * int(config.quant_scale)
    if config.quant_shift > 0:
        values = (values + (1 << (config.quant_shift - 1))) >> config.quant_shift
    values = values + int(config.output_zp)
    return np.clip(values, -128, 127).astype(np.int8)


def conv2d_same_reference(image: np.ndarray, config: QuantizedConvConfig) -> np.ndarray:
    src = ensure_int8_image(image).astype(np.int16)
    kernel = np.asarray(config.kernel, dtype=np.int16)
    height, width = src.shape
    padded = np.pad(src, ((1, 1), (1, 1)), mode="constant")
    acc = np.zeros((height, width), dtype=np.int32)
    for y in range(height):
        for x in range(width):
            window = padded[y : y + 3, x : x + 3]
            acc[y, x] = int(np.sum(window * kernel, dtype=np.int32))
    return quantize_accumulator(acc, config)


def build_strip(image: np.ndarray, x_start: int, strip_width: int = MICROKERNEL_INPUT_WIDTH) -> np.ndarray:
    src = ensure_int8_image(image)
    height, width = src.shape
    if strip_width != MICROKERNEL_INPUT_WIDTH:
        raise ValueError("this bitstream only supports a 5-column strip input")
    strip = np.zeros((height, strip_width), dtype=np.int8)
    for col in range(strip_width):
        source_x = x_start + col - 1
        if 0 <= source_x < width:
            strip[:, col] = src[:, source_x]
    return strip


def build_native_strip(image: np.ndarray, x_start: int, strip_width: int = MICROKERNEL_INPUT_WIDTH) -> np.ndarray:
    src = ensure_int8_image(image)
    height, width = src.shape
    if strip_width != MICROKERNEL_INPUT_WIDTH:
        raise ValueError("this bitstream only supports a 5-column strip input")
    strip = np.zeros((height, strip_width), dtype=np.int8)
    for col in range(strip_width):
        source_x = x_start + col
        if 0 <= source_x < width:
            strip[:, col] = src[:, source_x]
    return strip


def run_strip_reference(strip: np.ndarray, config: QuantizedConvConfig) -> np.ndarray:
    strip_image = ensure_int8_image(strip)
    if strip_image.shape[1] != MICROKERNEL_INPUT_WIDTH:
        raise ValueError(f"strip width must be {MICROKERNEL_INPUT_WIDTH}, got {strip_image.shape[1]}")
    return conv2d_same_reference(strip_image, config)


def run_tiled_reference(image: np.ndarray, config: QuantizedConvConfig) -> np.ndarray:
    src = ensure_int8_image(image)
    height, width = src.shape
    output = np.zeros((height, width), dtype=np.int8)
    for x_start in range(0, width, MICROKERNEL_VALID_OUTPUT_WIDTH):
        block_width = min(MICROKERNEL_VALID_OUTPUT_WIDTH, width - x_start)
        strip = build_strip(src, x_start)
        strip_output = run_strip_reference(strip, config)
        output[:, x_start : x_start + block_width] = strip_output[:, 1 : 1 + block_width]
    return output


def run_native_tiled_reference(image: np.ndarray, config: QuantizedConvConfig) -> np.ndarray:
    src = ensure_int8_image(image)
    height, width = src.shape
    output = np.zeros((height, width), dtype=np.int8)
    if width == 0:
        return output

    strip_start = 0
    write_x = 0
    first_strip = True

    while write_x < width:
        strip = build_native_strip(src, strip_start)
        strip_output = run_strip_reference(strip, config)
        if first_strip:
            take = min(MICROKERNEL_VALID_OUTPUT_WIDTH, width - write_x)
            output[:, write_x : write_x + take] = strip_output[:, :take]
            write_x += take
            first_strip = False
        else:
            take = min(MICROKERNEL_VALID_OUTPUT_WIDTH - 1, width - write_x)
            output[:, write_x : write_x + take] = strip_output[:, 1 : 1 + take]
            write_x += take
        strip_start += MICROKERNEL_VALID_OUTPUT_WIDTH - 1
    return output


def make_demo_image(width: int = 160, height: int = 96) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    gradient = (xx * 255 // max(width - 1, 1)).astype(np.uint8)
    checker = ((((xx // 8) ^ (yy // 8)) & 1) * 70).astype(np.uint8)
    image = np.clip(gradient // 2 + checker, 0, 255).astype(np.uint8)
    image[height // 5 : 4 * height // 5, width // 3 : width // 3 + 5] = 255
    image[height // 2 - 3 : height // 2 + 3, width // 6 : 5 * width // 6] = 0
    return image


def describe_array(name: str, image: np.ndarray) -> str:
    array = np.asarray(image)
    return f"{name}: shape={array.shape}, dtype={array.dtype}, min={array.min()}, max={array.max()}"

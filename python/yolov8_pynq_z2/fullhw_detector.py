from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .detections import Detection, non_max_suppression


CTRL_OFFSET = 0x00
SPEC_OFFSET = 0x04
OUTPUT_BYTES_OFFSET = 0x08
INPUT_BYTES_OFFSET = 0x0C
RECORD_BYTES_OFFSET = 0x10
STATUS_OFFSET = 0x14
IN_COUNT_OFFSET = 0x18
OUT_COUNT_OFFSET = 0x1C

DET1_MAGIC = b"DET1"


@dataclass(frozen=True)
class FullHwDetectorSpec:
    img_width: int
    img_height: int
    in_channels: int
    num_anchors: int
    box_params: int
    num_classes: int
    record_bytes: int
    input_bytes: int
    output_bytes: int

    @property
    def grid_width(self) -> int:
        return max(self.img_width - 2, 0)

    @property
    def grid_height(self) -> int:
        return max(self.img_height - 2, 0)

    @property
    def num_records(self) -> int:
        return self.grid_width * self.grid_height * self.num_anchors


@dataclass(frozen=True)
class ParsedDet1Packet:
    spec: FullHwDetectorSpec
    records_u8: np.ndarray
    packet: np.ndarray

    @property
    def records_i8(self) -> np.ndarray:
        return self.records_u8.view(np.int8)


def _clamp_int8(value: int | float) -> int:
    return int(max(-128, min(127, int(value))))


def _int8_to_u8(value: int | float) -> int:
    return int(np.uint8(np.int8(_clamp_int8(value))))


def _safe_sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(values, -32.0, 32.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def head_center_coeff(out_channel: int, in_channel: int) -> int:
    if out_channel == 0:
        return 1 if in_channel == 0 else 0
    if out_channel == 1:
        return 0 if in_channel == 0 else 1
    if out_channel == 2:
        return 1
    if out_channel == 3:
        return 1 if in_channel == 0 else 2
    if out_channel == 4:
        return -1 if in_channel == 0 else 1
    if out_channel == 5:
        return 1 if in_channel == 0 else -1
    if out_channel == 6:
        return 0 if in_channel == 0 else 2
    if out_channel == 7:
        return 2 if in_channel == 0 else 0
    return 0


def head_bias_value(out_channel: int) -> int:
    if out_channel == 0:
        return 10
    if out_channel == 1:
        return 20
    if out_channel == 2:
        return 0
    if out_channel == 3:
        return -10
    if out_channel == 4:
        return 64
    if out_channel == 5:
        return 32
    if out_channel == 6:
        return 1
    if out_channel == 7:
        return 2
    return 0


def make_synthetic_input(spec: FullHwDetectorSpec) -> np.ndarray:
    channels = []
    channel_pixels = spec.img_width * spec.img_height
    for channel_index in range(spec.in_channels):
        base = channel_index * 20
        values = (np.arange(channel_pixels, dtype=np.int16) + base).reshape(spec.img_height, spec.img_width)
        channels.append(values.astype(np.int8))
    return np.stack(channels, axis=0)


def pack_input_tensor(input_tensor_chw: np.ndarray) -> np.ndarray:
    tensor = np.asarray(input_tensor_chw, dtype=np.int8)
    if tensor.ndim != 3:
        raise ValueError(f"expected CHW int8 tensor, got shape {tensor.shape}")
    return tensor.reshape(-1).copy()


def prepare_camera_tensor(
    frame_bgr: np.ndarray,
    *,
    process_width: int,
    process_height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"expected BGR frame, got {frame_bgr.shape}")
    gray_u8 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray_u8 = cv2.resize(gray_u8, (process_width, process_height), interpolation=cv2.INTER_AREA)
    gray_i8 = np.clip(gray_u8.astype(np.int16) - 128, -128, 127).astype(np.int8)
    sobel_x = cv2.Sobel(gray_u8, cv2.CV_16S, 1, 0, ksize=3)
    sobel_i8 = np.clip(sobel_x, -128, 127).astype(np.int8)
    tensor = np.stack([gray_i8, sobel_i8], axis=0)
    return tensor, gray_u8


def prepare_camera_rgb_tensor(
    frame_bgr: np.ndarray,
    *,
    process_width: int,
    process_height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"expected BGR frame, got {frame_bgr.shape}")
    resized_bgr = cv2.resize(frame_bgr, (process_width, process_height), interpolation=cv2.INTER_AREA)
    rgb_u8 = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    rgb_i8 = np.clip(rgb_u8.astype(np.int16) - 128, -128, 127).astype(np.int8)
    tensor = np.transpose(rgb_i8, (2, 0, 1)).copy()
    return tensor, rgb_u8


def build_pl_only_demo_stem_tensor(input_tensor_chw: np.ndarray) -> np.ndarray:
    tensor = np.asarray(input_tensor_chw, dtype=np.int8)
    if tensor.ndim != 3:
        raise ValueError(f"expected CHW int8 tensor, got shape {tensor.shape}")
    if tensor.shape[0] != 3:
        raise ValueError(f"pl_only_demo stem expects 3 input channels, got {tensor.shape[0]}")
    _, img_height, img_width = tensor.shape
    if img_width < 3 or img_height < 3:
        raise ValueError(f"pl_only_demo stem requires at least 3x3 input, got {img_width}x{img_height}")
    # The PL stem uses a center-only 3x3 kernel with relu_enable=1, so its
    # effective software reference is "crop then ReLU" in int8 space.
    cropped = np.maximum(tensor[:, 1:-1, 1:-1].astype(np.int16), 0).astype(np.int8)
    zero_channel = np.zeros((1, cropped.shape[1], cropped.shape[2]), dtype=np.int8)
    return np.concatenate([cropped, zero_channel], axis=0)


def build_fullhw_reference_packet(
    input_tensor_chw: np.ndarray,
    *,
    num_anchors: int = 1,
    box_params: int = 5,
    num_classes: int = 3,
) -> np.ndarray:
    tensor = np.asarray(input_tensor_chw, dtype=np.int8)
    if tensor.ndim != 3:
        raise ValueError(f"expected CHW int8 tensor, got shape {tensor.shape}")

    in_channels, img_height, img_width = tensor.shape
    grid_width = max(img_width - 2, 0)
    grid_height = max(img_height - 2, 0)
    record_bytes = num_anchors * (box_params + num_classes)
    head_channels = record_bytes

    header = bytearray()
    header.extend(DET1_MAGIC)
    header.extend(np.uint16(grid_width).tobytes())
    header.extend(np.uint16(grid_height).tobytes())
    header.extend(np.uint16(num_anchors).tobytes())
    header.extend(np.uint16(box_params).tobytes())
    header.extend(np.uint16(num_classes).tobytes())
    header.extend(np.uint16(record_bytes).tobytes())

    payload = bytearray()
    for row in range(grid_height):
        for col in range(grid_width):
            center = tensor[:, row + 1, col + 1]
            for out_channel in range(head_channels):
                acc = head_bias_value(out_channel)
                for in_channel in range(in_channels):
                    acc += head_center_coeff(out_channel, in_channel) * int(center[in_channel])
                payload.append(_int8_to_u8(acc))

    return np.frombuffer(bytes(header + payload), dtype=np.uint8)


def build_pl_only_demo_reference_packet(input_tensor_chw: np.ndarray) -> np.ndarray:
    stem_tensor = build_pl_only_demo_stem_tensor(input_tensor_chw)
    return build_fullhw_reference_packet(stem_tensor)


def parse_det1_packet(packet_bytes: np.ndarray, *, in_channels: int) -> ParsedDet1Packet:
    packet = np.asarray(packet_bytes, dtype=np.uint8).reshape(-1)
    if packet.size < 16:
        raise ValueError("DET1 packet is shorter than the 16-byte header")
    if bytes(packet[:4].tolist()) != DET1_MAGIC:
        raise ValueError("invalid DET1 packet magic")

    grid_width = int(np.frombuffer(packet[4:6].tobytes(), dtype=np.uint16)[0])
    grid_height = int(np.frombuffer(packet[6:8].tobytes(), dtype=np.uint16)[0])
    num_anchors = int(np.frombuffer(packet[8:10].tobytes(), dtype=np.uint16)[0])
    box_params = int(np.frombuffer(packet[10:12].tobytes(), dtype=np.uint16)[0])
    num_classes = int(np.frombuffer(packet[12:14].tobytes(), dtype=np.uint16)[0])
    record_bytes = int(np.frombuffer(packet[14:16].tobytes(), dtype=np.uint16)[0])
    num_records = grid_width * grid_height * num_anchors
    payload_bytes = num_records * record_bytes
    expected_size = 16 + payload_bytes
    if packet.size != expected_size:
        raise ValueError(f"DET1 packet size mismatch: expected {expected_size}, got {packet.size}")

    records_u8 = packet[16:].reshape(num_records, record_bytes)
    spec = FullHwDetectorSpec(
        img_width=grid_width + 2,
        img_height=grid_height + 2,
        in_channels=in_channels,
        num_anchors=num_anchors,
        box_params=box_params,
        num_classes=num_classes,
        record_bytes=record_bytes,
        input_bytes=(grid_width + 2) * (grid_height + 2) * in_channels,
        output_bytes=packet.size,
    )
    return ParsedDet1Packet(spec=spec, records_u8=records_u8, packet=packet)


def det1_packet_to_detections(
    parsed: ParsedDet1Packet,
    *,
    original_shape: Tuple[int, int],
    class_names: Optional[Sequence[str]] = None,
    score_threshold: float = 0.35,
    max_detections: int = 20,
    iou_threshold: float = 0.25,
    class_agnostic_nms: bool = False,
) -> List[Detection]:
    grid_width = parsed.spec.grid_width
    grid_height = parsed.spec.grid_height
    if grid_width == 0 or grid_height == 0:
        return []

    out_h, out_w = original_shape
    scale_x = float(out_w) / float(grid_width)
    scale_y = float(out_h) / float(grid_height)

    records_u8 = parsed.records_u8.astype(np.float32, copy=False)
    records_i8 = parsed.records_i8.astype(np.float32, copy=False)
    num_records = records_u8.shape[0]
    if num_records == 0:
        return []

    obj_scores = _safe_sigmoid(records_i8[:, 4]).astype(np.float32, copy=False)
    if parsed.spec.num_classes > 0:
        class_logits = records_i8[:, 5 : 5 + parsed.spec.num_classes]
        class_probs = _safe_sigmoid(class_logits).astype(np.float32, copy=False)
        class_ids = class_probs.argmax(axis=1).astype(np.int32, copy=False)
        class_scores = class_probs[np.arange(num_records), class_ids]
    else:
        class_ids = np.zeros((num_records,), dtype=np.int32)
        class_scores = np.ones((num_records,), dtype=np.float32)

    scores = (obj_scores * class_scores).astype(np.float32, copy=False)
    keep_mask = scores >= float(score_threshold)
    if not np.any(keep_mask):
        return []

    keep_indices = np.flatnonzero(keep_mask)
    scores = scores[keep_mask]
    class_ids = class_ids[keep_mask]
    records_u8 = records_u8[keep_mask]

    cell_x = (keep_indices % grid_width).astype(np.float32, copy=False)
    cell_y = (keep_indices // grid_width).astype(np.float32, copy=False)
    x_center = (cell_x + 0.5) * scale_x + (records_u8[:, 0] / 255.0 - 0.5) * scale_x
    y_center = (cell_y + 0.5) * scale_y + (records_u8[:, 1] / 255.0 - 0.5) * scale_y
    box_w = np.maximum(scale_x, (records_u8[:, 2] / 255.0) * float(out_w))
    box_h = np.maximum(scale_y, (records_u8[:, 3] / 255.0) * float(out_h))

    x1 = np.clip(x_center - box_w * 0.5, 0.0, float(out_w - 1))
    y1 = np.clip(y_center - box_h * 0.5, 0.0, float(out_h - 1))
    x2 = np.clip(x_center + box_w * 0.5, 0.0, float(out_w - 1))
    y2 = np.clip(y_center + box_h * 0.5, 0.0, float(out_h - 1))

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32, copy=False)
    keep_indices = non_max_suppression(
        boxes_xyxy,
        scores,
        class_ids,
        iou_threshold=float(iou_threshold),
        max_detections=max_detections,
        class_agnostic=bool(class_agnostic_nms),
    )

    detections: List[Detection] = []
    for idx in keep_indices:
        class_id = int(class_ids[idx])
        if class_names and 0 <= class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = f"class_{class_id}"
        detections.append(
            Detection(
                x1=float(x1[idx]),
                y1=float(y1[idx]),
                x2=float(x2[idx]),
                y2=float(y2[idx]),
                score=float(scores[idx]),
                class_id=class_id,
                class_name=class_name,
            )
        )

    return detections


class FullHwDetectorOverlay:
    dma_name = "axi_dma_0"
    ip_name = "YOLO_Engine_AXI_0"

    def __init__(self, bitfile: str | Path, *, download: bool = True):
        try:
            from pynq import Overlay, allocate
        except ImportError as exc:
            raise RuntimeError("pynq is not installed in this Python environment") from exc

        self._allocate = allocate
        self.bitfile = str(bitfile)
        self.overlay = Overlay(self.bitfile, download=download)
        self.dma = getattr(self.overlay, self.dma_name)
        self.ip = getattr(self.overlay, self.ip_name)
        self.spec = self._read_spec()
        self._in_buf = self._allocate(shape=(self.spec.input_bytes,), dtype=np.int8)
        self._out_buf = self._allocate(shape=(self.spec.output_bytes,), dtype=np.uint8)
        self._closed = False

    def _read_spec(self) -> FullHwDetectorSpec:
        spec_reg = int(self.ip.read(SPEC_OFFSET))
        img_width = spec_reg & 0xFFFF
        img_height = (spec_reg >> 16) & 0xFFFF
        input_bytes = int(self.ip.read(INPUT_BYTES_OFFSET))
        output_bytes = int(self.ip.read(OUTPUT_BYTES_OFFSET))
        record_bytes = int(self.ip.read(RECORD_BYTES_OFFSET)) & 0xFFFF
        if img_width <= 0 or img_height <= 0:
            raise RuntimeError(f"invalid detector spec width/height from register: {img_width}x{img_height}")
        pixels = img_width * img_height
        if pixels <= 0 or input_bytes % pixels != 0:
            raise RuntimeError(f"cannot infer channel count from input_bytes={input_bytes}, pixels={pixels}")
        in_channels = input_bytes // pixels
        num_anchors = 1
        box_params = 5
        num_classes = max(record_bytes - box_params, 0)
        return FullHwDetectorSpec(
            img_width=img_width,
            img_height=img_height,
            in_channels=in_channels,
            num_anchors=num_anchors,
            box_params=box_params,
            num_classes=num_classes,
            record_bytes=record_bytes,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
        )

    def _ensure_dma_started(self) -> None:
        send = self.dma.sendchannel
        recv = self.dma.recvchannel
        try:
            send.start()
        except RuntimeError:
            pass
        try:
            recv.start()
        except RuntimeError:
            pass

    def soft_reset(self) -> None:
        self.ip.write(CTRL_OFFSET, 0x00000002)
        self.ip.write(CTRL_OFFSET, 0x00000000)

    def start(self) -> None:
        self.ip.write(CTRL_OFFSET, 0x00000001)
        self.ip.write(CTRL_OFFSET, 0x00000000)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._in_buf.close()
        finally:
            self._out_buf.close()
            self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def run_tensor(self, input_tensor_chw: np.ndarray) -> ParsedDet1Packet:
        tensor = np.asarray(input_tensor_chw, dtype=np.int8)
        expected_shape = (self.spec.in_channels, self.spec.img_height, self.spec.img_width)
        if tensor.shape != expected_shape:
            raise ValueError(f"expected input tensor shape {expected_shape}, got {tensor.shape}")

        flat_input = pack_input_tensor(tensor)
        if flat_input.size != self.spec.input_bytes:
            raise ValueError(f"expected {self.spec.input_bytes} input bytes, got {flat_input.size}")

        self._in_buf[:] = flat_input
        self._out_buf[:] = 0
        if hasattr(self._in_buf, "flush"):
            self._in_buf.flush()

        self._ensure_dma_started()
        self.soft_reset()
        self.dma.recvchannel.transfer(self._out_buf)
        self.dma.sendchannel.transfer(self._in_buf)
        self.start()
        self.dma.recvchannel.wait()
        self.dma.sendchannel.wait()
        if hasattr(self._out_buf, "invalidate"):
            self._out_buf.invalidate()
        packet = np.array(self._out_buf, dtype=np.uint8)
        return parse_det1_packet(packet, in_channels=self.spec.in_channels)

    def read_counters(self) -> dict:
        return {
            "status_reg": int(self.ip.read(STATUS_OFFSET)),
            "in_count_reg": int(self.ip.read(IN_COUNT_OFFSET)),
            "out_count_reg": int(self.ip.read(OUT_COUNT_OFFSET)),
        }

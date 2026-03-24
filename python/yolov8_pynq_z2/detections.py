from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    class_name: str

    def to_dict(self) -> dict:
        return {
            "bbox_xyxy": [float(self.x1), float(self.y1), float(self.x2), float(self.y2)],
            "score": float(self.score),
            "class_id": int(self.class_id),
            "class_name": self.class_name,
        }


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def standardize_yolov8_predictions(predictions: np.ndarray) -> np.ndarray:
    array = np.asarray(predictions)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"expected a 2D YOLO prediction tensor after squeeze, got shape {array.shape}")

    rows, cols = array.shape
    if rows >= 5 and cols >= 5:
        if rows <= 256 and cols > rows:
            return array.T.astype(np.float32, copy=False)
        if cols <= 256:
            return array.astype(np.float32, copy=False)
        return array.astype(np.float32, copy=False)
    if cols >= 5:
        return array.astype(np.float32, copy=False)
    if rows >= 5:
        return array.T.astype(np.float32, copy=False)

    raise ValueError(f"unable to interpret YOLO predictions with shape {array.shape}")


def xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes_xywh, dtype=np.float32)
    xyxy = np.empty_like(boxes, dtype=np.float32)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] * 0.5
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] * 0.5
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] * 0.5
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] * 0.5
    return xyxy


def clip_boxes_xyxy(boxes_xyxy: np.ndarray, *, width: int, height: int) -> np.ndarray:
    boxes = np.asarray(boxes_xyxy, dtype=np.float32).copy()
    boxes[:, 0] = np.clip(boxes[:, 0], 0, max(width - 1, 0))
    boxes[:, 1] = np.clip(boxes[:, 1], 0, max(height - 1, 0))
    boxes[:, 2] = np.clip(boxes[:, 2], 0, max(width - 1, 0))
    boxes[:, 3] = np.clip(boxes[:, 3], 0, max(height - 1, 0))
    return boxes


def scale_boxes_xyxy(
    boxes_xyxy: np.ndarray,
    *,
    input_shape: Tuple[int, int],
    original_shape: Tuple[int, int],
) -> np.ndarray:
    boxes = np.asarray(boxes_xyxy, dtype=np.float32).copy()
    in_h, in_w = input_shape
    out_h, out_w = original_shape
    if in_w <= 0 or in_h <= 0:
        raise ValueError(f"invalid input shape {input_shape}")
    scale_x = float(out_w) / float(in_w)
    scale_y = float(out_h) / float(in_h)
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y
    return clip_boxes_xyxy(boxes, width=out_w, height=out_h)


def compute_iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_box = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area_boxes = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = np.maximum(area_box + area_boxes - inter, 1e-9)
    return inter / union


def non_max_suppression(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    *,
    iou_threshold: float,
    max_detections: int,
    class_agnostic: bool = False,
) -> List[int]:
    boxes = np.asarray(boxes_xyxy, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    class_ids = np.asarray(class_ids, dtype=np.int32)
    order = scores.argsort()[::-1]
    keep: List[int] = []

    while order.size > 0 and len(keep) < max_detections:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break

        remaining = order[1:]
        ious = compute_iou_xyxy(boxes[current], boxes[remaining])
        if class_agnostic:
            suppress = ious > float(iou_threshold)
        else:
            suppress = (ious > float(iou_threshold)) & (class_ids[remaining] == class_ids[current])
        order = remaining[~suppress]

    return keep


def decode_yolov8_detections(
    predictions: np.ndarray,
    *,
    input_shape: Tuple[int, int],
    original_shape: Optional[Tuple[int, int]] = None,
    score_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 100,
    class_names: Optional[Sequence[str]] = None,
    class_agnostic_nms: bool = False,
) -> List[Detection]:
    rows = standardize_yolov8_predictions(predictions)
    if rows.shape[1] < 5:
        raise ValueError(f"expected at least 5 channels per prediction row, got shape {rows.shape}")

    boxes_xywh = rows[:, :4]
    class_scores = rows[:, 4:]
    if class_scores.shape[1] == 0:
        raise ValueError("prediction tensor does not contain class scores")

    if class_scores.min() < 0.0 or class_scores.max() > 1.0:
        class_scores = _sigmoid(class_scores)

    class_ids = class_scores.argmax(axis=1).astype(np.int32)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
    keep_mask = scores >= float(score_threshold)
    if not np.any(keep_mask):
        return []

    boxes_xyxy = xywh_to_xyxy(boxes_xywh[keep_mask])
    scores = scores[keep_mask].astype(np.float32)
    class_ids = class_ids[keep_mask]

    if original_shape is not None:
        boxes_xyxy = scale_boxes_xyxy(boxes_xyxy, input_shape=input_shape, original_shape=original_shape)
        out_h, out_w = original_shape
    else:
        out_h, out_w = input_shape
        boxes_xyxy = clip_boxes_xyxy(boxes_xyxy, width=out_w, height=out_h)

    keep_indices = non_max_suppression(
        boxes_xyxy,
        scores,
        class_ids,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
        class_agnostic=class_agnostic_nms,
    )

    detections: List[Detection] = []
    for idx in keep_indices:
        class_id = int(class_ids[idx])
        if class_names and 0 <= class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = f"class_{class_id}"
        x1, y1, x2, y2 = boxes_xyxy[idx]
        detections.append(
            Detection(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                score=float(scores[idx]),
                class_id=class_id,
                class_name=class_name,
            )
        )
    return detections


def draw_detections(
    image_bgr: np.ndarray,
    detections: Iterable[Detection],
    *,
    line_thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    canvas = np.asarray(image_bgr).copy()
    for det in detections:
        color = (
            int((37 * (det.class_id + 1)) % 255),
            int((17 * (det.class_id + 11)) % 255),
            int((29 * (det.class_id + 23)) % 255),
        )
        pt1 = (int(round(det.x1)), int(round(det.y1)))
        pt2 = (int(round(det.x2)), int(round(det.y2)))
        cv2.rectangle(canvas, pt1, pt2, color, thickness=line_thickness)
        label = f"{det.class_name} {det.score:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        box_y1 = max(0, pt1[1] - text_h - baseline - 4)
        box_y2 = box_y1 + text_h + baseline + 4
        box_x2 = min(canvas.shape[1], pt1[0] + text_w + 6)
        cv2.rectangle(canvas, (pt1[0], box_y1), (box_x2, box_y2), color, thickness=-1)
        cv2.putText(
            canvas,
            label,
            (pt1[0] + 3, box_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def save_detections_json(path: Path, detections: Sequence[Detection]) -> Path:
    payload = {"detections": [det.to_dict() for det in detections], "count": len(detections)}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path

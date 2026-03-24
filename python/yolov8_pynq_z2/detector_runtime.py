from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
from urllib import request

import cv2
import numpy as np

from .detections import Detection, decode_yolov8_detections


@dataclass(frozen=True)
class DetectorConfig:
    backend: str = "helmet_heuristic"
    model_path: Optional[str] = None
    class_names: Sequence[str] = ("helmet",)
    score_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 50
    input_width: int = 0
    input_height: int = 0
    service_url: Optional[str] = None
    request_timeout: float = 10.0


class BaseDetector:
    backend_name = "none"

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        raise NotImplementedError


class NoOpDetector(BaseDetector):
    backend_name = "none"

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        return []


class OpenCVDnnYoloDetector(BaseDetector):
    backend_name = "opencv_dnn"

    def __init__(self, config: DetectorConfig):
        if not config.model_path:
            raise ValueError("opencv_dnn backend requires model_path")
        model_path = Path(config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        if config.input_width <= 0 or config.input_height <= 0:
            raise ValueError("opencv_dnn backend requires positive input_width and input_height")

        self.config = config
        self.net = cv2.dnn.readNet(str(model_path))
        self.output_names = self._get_output_names()
        self.preferred_output_name = self._select_output_name(self.output_names)

    def _get_output_names(self) -> tuple[str, ...]:
        try:
            names = self.net.getUnconnectedOutLayersNames()
        except AttributeError:
            return ()
        return tuple(str(name) for name in names)

    @staticmethod
    def _select_output_name(output_names: Sequence[str]) -> Optional[str]:
        if not output_names:
            return None

        exact_priority = ("predictions", "output0", "output")
        lowered = {name.lower(): name for name in output_names}
        for candidate in exact_priority:
            if candidate in lowered:
                return lowered[candidate]

        for name in output_names:
            lowered_name = name.lower()
            if "prediction" in lowered_name or "detect" in lowered_name:
                return name

        return output_names[-1]

    @staticmethod
    def _select_prediction_blob(outputs: object) -> np.ndarray:
        if isinstance(outputs, (list, tuple)):
            arrays = [np.asarray(output) for output in outputs]
            for array in arrays:
                if np.squeeze(array).ndim == 2:
                    return array
            return arrays[-1]
        return np.asarray(outputs)

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        height, width = image_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image_bgr,
            scalefactor=1.0 / 255.0,
            size=(self.config.input_width, self.config.input_height),
            mean=(0.0, 0.0, 0.0),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        if self.preferred_output_name:
            outputs = self.net.forward(self.preferred_output_name)
        elif self.output_names:
            outputs = self.net.forward(self.output_names)
        else:
            outputs = self.net.forward()
        outputs = self._select_prediction_blob(outputs)
        return decode_yolov8_detections(
            outputs,
            input_shape=(self.config.input_height, self.config.input_width),
            original_shape=(height, width),
            score_threshold=self.config.score_threshold,
            iou_threshold=self.config.iou_threshold,
            max_detections=self.config.max_detections,
            class_names=self.config.class_names,
        )


class RemoteHttpDetector(BaseDetector):
    backend_name = "remote_http"

    def __init__(self, config: DetectorConfig):
        if not config.service_url:
            raise ValueError("remote_http backend requires service_url")
        self.config = config

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise RuntimeError("failed to JPEG-encode frame for remote detector")

        req = request.Request(
            self.config.service_url,
            data=encoded.tobytes(),
            headers={"Content-Type": "image/jpeg"},
            method="POST",
        )
        with request.urlopen(req, timeout=float(self.config.request_timeout)) as response:
            payload = json.loads(response.read().decode("utf-8"))

        raw_detections = payload.get("detections", [])
        detections: list[Detection] = []
        for item in raw_detections:
            bbox = item.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])
            class_id = int(item.get("class_id", 0))
            if item.get("class_name"):
                class_name = str(item["class_name"])
            elif 0 <= class_id < len(self.config.class_names):
                class_name = str(self.config.class_names[class_id])
            else:
                class_name = f"class_{class_id}"

            detections.append(
                Detection(
                    x1=float(bbox[0]),
                    y1=float(bbox[1]),
                    x2=float(bbox[2]),
                    y2=float(bbox[3]),
                    score=float(item.get("score", 0.0)),
                    class_id=class_id,
                    class_name=class_name,
                )
            )
        return detections[: self.config.max_detections]


class HelmetHeuristicDetector(BaseDetector):
    backend_name = "helmet_heuristic"

    def __init__(self, config: DetectorConfig):
        self.config = config

    def _color_masks(self, hsv: np.ndarray) -> list[np.ndarray]:
        masks = []
        # Yellow
        masks.append(cv2.inRange(hsv, (18, 60, 60), (40, 255, 255)))
        # Orange
        masks.append(cv2.inRange(hsv, (5, 90, 60), (18, 255, 255)))
        # Blue
        masks.append(cv2.inRange(hsv, (90, 70, 50), (130, 255, 255)))
        # Red, split around hue wrap
        red1 = cv2.inRange(hsv, (0, 90, 60), (8, 255, 255))
        red2 = cv2.inRange(hsv, (170, 90, 60), (179, 255, 255))
        masks.append(cv2.bitwise_or(red1, red2))
        return masks

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        frame = np.asarray(image_bgr)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"expected BGR image, got {frame.shape}")

        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros((height, width), dtype=np.uint8)
        for partial in self._color_masks(hsv):
            mask = cv2.bitwise_or(mask, partial)

        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[Detection] = []
        image_area = float(height * width)
        class_name = self.config.class_names[0] if self.config.class_names else "helmet_candidate"

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < max(24.0, image_area * 0.0006) or area > image_area * 0.20:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if h <= 0 or w <= 0:
                continue
            aspect = float(w) / float(h)
            if not 0.45 <= aspect <= 2.2:
                continue
            if y > int(height * 0.92):
                continue

            perimeter = max(float(cv2.arcLength(contour, True)), 1e-6)
            circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
            if circularity < 0.15:
                continue

            roi = hsv[y : y + h, x : x + w]
            if roi.size == 0:
                continue
            mean_sat = float(roi[..., 1].mean()) / 255.0
            area_ratio = min(area / max(image_area * 0.02, 1.0), 1.0)
            score = max(0.05, min(0.99, 0.45 * mean_sat + 0.35 * max(circularity, 0.0) + 0.20 * area_ratio))
            if score < self.config.score_threshold:
                continue

            detections.append(
                Detection(
                    x1=float(x),
                    y1=float(y),
                    x2=float(x + w),
                    y2=float(y + h),
                    score=float(score),
                    class_id=0,
                    class_name=class_name,
                )
            )

        detections.sort(key=lambda det: det.score, reverse=True)
        return detections[: self.config.max_detections]


def create_detector(config: DetectorConfig) -> BaseDetector:
    backend = str(config.backend or "none").lower()
    if backend == "none":
        return NoOpDetector()
    if backend == "helmet_heuristic":
        return HelmetHeuristicDetector(config)
    if backend == "remote_http":
        return RemoteHttpDetector(config)
    if backend == "opencv_dnn":
        return OpenCVDnnYoloDetector(config)
    raise ValueError(f"unsupported detector backend: {config.backend}")

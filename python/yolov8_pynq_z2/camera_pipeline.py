from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np

from .config import QuantizedConvConfig
from .detections import Detection, draw_detections
from .detector_runtime import BaseDetector
from .model import run_tiled_reference, to_display_image, to_signed_image
from .overlay import YoloPynqZ2Overlay


def open_camera(
    camera_index: int,
    *,
    capture_width: int,
    capture_height: int,
    requested_fps: float = 0.0,
) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open camera index {camera_index}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if capture_width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(capture_width))
    if capture_height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(capture_height))
    if requested_fps > 0:
        cap.set(cv2.CAP_PROP_FPS, float(requested_fps))
    return cap


def preprocess_camera_frame(frame_bgr: np.ndarray, *, process_width: int, process_height: int) -> np.ndarray:
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"expected a BGR frame with shape (H, W, 3), got {frame_bgr.shape}")
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (process_width, process_height), interpolation=cv2.INTER_AREA)


def build_preview_frame(
    input_u8: np.ndarray,
    output_i8: Optional[np.ndarray],
    *,
    preset_name: str,
    backend: str,
    frame_index: int,
    inference_ms: float,
    detect_ms: float,
    loop_fps: float,
    compatibility_mode: Optional[str],
    preview_scale: int,
    source_bgr: Optional[np.ndarray] = None,
    detections: Optional[Iterable[Detection]] = None,
) -> np.ndarray:
    if source_bgr is not None:
        input_bgr = np.asarray(source_bgr).copy()
    else:
        input_bgr = cv2.cvtColor(input_u8, cv2.COLOR_GRAY2BGR)
    if detections:
        input_bgr = draw_detections(input_bgr, list(detections))
    if output_i8 is None:
        output_bgr = np.zeros_like(input_bgr)
        cv2.putText(output_bgr, "No feature map", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        output_bgr = cv2.cvtColor(to_display_image(output_i8), cv2.COLOR_GRAY2BGR)
    if preview_scale > 1:
        size = (input_bgr.shape[1] * preview_scale, input_bgr.shape[0] * preview_scale)
        input_bgr = cv2.resize(input_bgr, size, interpolation=cv2.INTER_NEAREST)
        output_bgr = cv2.resize(output_bgr, size, interpolation=cv2.INTER_NEAREST)

    preview = np.hstack([input_bgr, output_bgr])
    cv2.rectangle(preview, (0, 0), (preview.shape[1], 66), (18, 18, 18), thickness=-1)
    cv2.putText(preview, "Input", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        preview,
        f"Output ({preset_name if output_i8 is not None else 'bypass'})",
        (input_bgr.shape[1] + 12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    status = f"backend={backend} frame={frame_index} det={detect_ms:.1f}ms infer={inference_ms:.1f}ms loop_fps={loop_fps:.2f}"
    cv2.putText(preview, status, (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 255, 180), 1, cv2.LINE_AA)

    if compatibility_mode:
        mode_text = f"mode={compatibility_mode}"
        cv2.putText(
            preview,
            mode_text,
            (input_bgr.shape[1] + 12, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (120, 220, 255),
            1,
            cv2.LINE_AA,
        )
    return preview


def create_video_writer(output_path: Path, frame_shape: Tuple[int, int, int], fps: float) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame_shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        max(float(fps), 1.0),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {output_path}")
    return writer


def save_latest_artifacts(
    output_dir: Path,
    *,
    input_u8: np.ndarray,
    output_i8: Optional[np.ndarray],
    preview_bgr: np.ndarray,
    metadata: dict,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "camera_input_latest.png"
    output_path = output_dir / "camera_output_latest.png"
    preview_path = output_dir / "camera_preview_latest.jpg"
    metadata_path = output_dir / "camera_latest_metadata.json"

    cv2.imwrite(str(input_path), input_u8)
    if output_i8 is None:
        cv2.imwrite(str(output_path), np.zeros_like(input_u8))
    else:
        cv2.imwrite(str(output_path), to_display_image(output_i8))
    cv2.imwrite(str(preview_path), preview_bgr)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "preview_path": str(preview_path),
        "metadata_path": str(metadata_path),
    }


def run_camera_pipeline(
    *,
    preset: QuantizedConvConfig,
    backend: str,
    overlay_bitfile: Optional[Path | str],
    camera_index: int,
    capture_width: int,
    capture_height: int,
    process_width: int,
    process_height: int,
    requested_fps: float,
    preview_scale: int,
    max_frames: int,
    warmup_frames: int,
    display_window: bool,
    download_overlay: bool,
    save_every: int,
    output_dir: Path,
    save_video_path: Optional[Path],
    detector: Optional[BaseDetector] = None,
) -> dict:
    if backend not in {"board", "reference", "passthrough"}:
        raise ValueError(f"unsupported backend {backend}")
    if process_width <= 0 or process_height <= 0:
        raise ValueError("process dimensions must be positive")
    if preview_scale <= 0:
        raise ValueError("preview_scale must be positive")
    if max_frames < 0:
        raise ValueError("max_frames must be non-negative")
    if warmup_frames < 0:
        raise ValueError("warmup_frames must be non-negative")
    if save_every < 0:
        raise ValueError("save_every must be non-negative")

    if display_window and not os.environ.get("DISPLAY"):
        display_window = False

    overlay: Optional[YoloPynqZ2Overlay]
    if backend == "board":
        overlay = YoloPynqZ2Overlay(bitfile=overlay_bitfile, download=download_overlay)
    else:
        overlay = None

    cap = open_camera(
        camera_index,
        capture_width=capture_width,
        capture_height=capture_height,
        requested_fps=requested_fps,
    )
    writer: Optional[cv2.VideoWriter] = None
    processed_frames = 0
    capture_failures = 0
    start_time = time.perf_counter()
    last_frame_time = start_time
    latest_paths: dict = {}

    try:
        for _ in range(warmup_frames):
            ok, _ = cap.read()
            if not ok:
                capture_failures += 1

        while max_frames == 0 or processed_frames < max_frames:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                capture_failures += 1
                if capture_failures >= 3:
                    raise RuntimeError("camera capture failed repeatedly")
                continue

            frame_index = processed_frames + 1
            need_artifacts = save_every > 0 and frame_index % save_every == 0
            need_preview = display_window or save_video_path is not None or need_artifacts
            need_feature_map = backend in {"board", "reference"}
            need_gray_input = need_feature_map or need_artifacts or need_preview

            detect_start = time.perf_counter()
            detections = detector.detect(frame_bgr) if detector is not None else []
            detect_end = time.perf_counter()

            gray_u8: Optional[np.ndarray] = None
            if need_gray_input:
                gray_u8 = preprocess_camera_frame(
                    frame_bgr,
                    process_width=process_width,
                    process_height=process_height,
                )

            infer_start = time.perf_counter()
            output_i8: Optional[np.ndarray]
            if backend == "passthrough":
                output_i8 = None
                compatibility_mode = None
            else:
                assert gray_u8 is not None
                signed = to_signed_image(gray_u8)
                if overlay is not None:
                    output_i8 = overlay.run_tiled(signed, preset)
                    compatibility_mode = overlay.compatibility_mode
                else:
                    output_i8 = run_tiled_reference(signed, preset)
                    compatibility_mode = None
            infer_end = time.perf_counter()

            now = infer_end
            loop_fps = 1.0 / max(now - last_frame_time, 1e-9)
            last_frame_time = now

            preview_bgr: Optional[np.ndarray] = None
            if need_preview:
                assert gray_u8 is not None
                preview_bgr = build_preview_frame(
                    gray_u8,
                    output_i8,
                    preset_name=preset.name,
                    backend=backend,
                    frame_index=frame_index,
                    inference_ms=(infer_end - infer_start) * 1000.0,
                    detect_ms=(detect_end - detect_start) * 1000.0,
                    loop_fps=loop_fps,
                    compatibility_mode=compatibility_mode,
                    preview_scale=preview_scale,
                    source_bgr=cv2.resize(frame_bgr, (gray_u8.shape[1], gray_u8.shape[0]), interpolation=cv2.INTER_AREA),
                    detections=detections,
                )

            if preview_bgr is not None and writer is None and save_video_path is not None:
                writer = create_video_writer(save_video_path, preview_bgr.shape, fps=max(loop_fps, requested_fps, 1.0))
            if writer is not None and preview_bgr is not None:
                writer.write(preview_bgr)

            metadata = {
                "backend": backend,
                "preset": preset.name,
                "frame_index": frame_index,
                "loop_fps": loop_fps,
                "detect_ms": (detect_end - detect_start) * 1000.0,
                "inference_ms": (infer_end - infer_start) * 1000.0,
                "compatibility_mode": compatibility_mode,
                "camera_index": camera_index,
                "capture_size": [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))],
                "process_size": [process_height, process_width],
                "detections": [det.to_dict() for det in detections],
            }
            if need_artifacts:
                assert gray_u8 is not None
                assert preview_bgr is not None
                latest_paths = save_latest_artifacts(
                    output_dir,
                    input_u8=gray_u8,
                    output_i8=output_i8,
                    preview_bgr=preview_bgr,
                    metadata=metadata,
                )

            if display_window:
                assert preview_bgr is not None
                cv2.imshow("yolov8_pynq_z2_camera", preview_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    processed_frames = frame_index
                    break

            processed_frames = frame_index
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display_window:
            cv2.destroyAllWindows()

    total_time = time.perf_counter() - start_time
    average_fps = processed_frames / max(total_time, 1e-9)
    result = {
        "backend": backend,
        "preset": preset.name,
        "frames_processed": processed_frames,
        "capture_failures": capture_failures,
        "average_fps": average_fps,
        "elapsed_seconds": total_time,
        "camera_index": camera_index,
        "capture_size": [int(capture_width), int(capture_height)],
        "process_size": [int(process_height), int(process_width)],
        "display_window": bool(display_window),
        "save_every": int(save_every),
        "save_video_path": str(save_video_path) if save_video_path is not None else None,
        "latest_artifacts": latest_paths,
        "compatibility_mode": overlay.compatibility_mode if overlay is not None else None,
        "detector_backend": getattr(detector, "backend_name", "none"),
    }
    return result

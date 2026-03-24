from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from yolov8_pynq_z2.camera_pipeline import create_video_writer, open_camera, save_latest_artifacts
from yolov8_pynq_z2.detections import draw_detections
from yolov8_pynq_z2.fullhw_detector import (
    FullHwDetectorOverlay,
    det1_packet_to_detections,
    prepare_camera_rgb_tensor,
    prepare_camera_tensor,
)


def filter_preview_detections(
    detections,
    *,
    image_shape,
    min_area_ratio: float,
    max_area_ratio: float,
    max_aspect_ratio: float,
    border_margin_ratio: float,
    top_k: int,
):
    if not detections:
        return []

    img_h, img_w = image_shape
    frame_area = max(float(img_h * img_w), 1.0)
    margin_x = float(img_w) * max(float(border_margin_ratio), 0.0)
    margin_y = float(img_h) * max(float(border_margin_ratio), 0.0)
    kept = []

    for det in detections:
        width = max(float(det.x2 - det.x1), 0.0)
        height = max(float(det.y2 - det.y1), 0.0)
        if width <= 1.0 or height <= 1.0:
            continue
        area_ratio = (width * height) / frame_area
        if area_ratio < float(min_area_ratio) or area_ratio > float(max_area_ratio):
            continue
        aspect_ratio = max(width / height, height / width)
        if aspect_ratio > float(max_aspect_ratio):
            continue
        if (
            det.x1 <= margin_x
            or det.y1 <= margin_y
            or det.x2 >= float(img_w - 1) - margin_x
            or det.y2 >= float(img_h - 1) - margin_y
        ):
            continue
        kept.append(det)

    kept.sort(key=lambda det: det.score * max(float(det.x2 - det.x1), 0.0) * max(float(det.y2 - det.y1), 0.0), reverse=True)
    if top_k > 0:
        kept = kept[:top_k]
    return kept


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PS/PL partitioned fullhw detector overlay with a realtime camera on PYNQ-Z2.")
    parser.add_argument(
        "--bitfile",
        type=Path,
        default=REPO_ROOT / "hardware" / "overlay" / "yolo_pynq_z2_fullhw_plonly_demo_cam32.bit",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--capture-width", type=int, default=640)
    parser.add_argument("--capture-height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument("--preview-scale", type=float, default=1.0)
    parser.add_argument("--preview-layout", choices=("source_only", "panels"), default="source_only")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means run until interrupted.")
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--score-threshold", type=float, default=0.60)
    parser.add_argument("--iou-threshold", type=float, default=0.20)
    parser.add_argument("--max-detections", type=int, default=6)
    parser.add_argument("--class-agnostic-nms", action="store_true")
    parser.add_argument("--class-names", nargs="*", default=["class_0", "class_1", "class_2"])
    parser.add_argument("--preview-min-area-ratio", type=float, default=0.03)
    parser.add_argument("--preview-max-area-ratio", type=float, default=0.40)
    parser.add_argument("--preview-max-aspect-ratio", type=float, default=3.0)
    parser.add_argument("--preview-border-margin-ratio", type=float, default=0.02)
    parser.add_argument("--preview-top-k", type=int, default=2)
    parser.add_argument("--display-window", action="store_true")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--preview-only-artifacts", action="store_true")
    parser.add_argument("--preview-jpeg-quality", type=int, default=80)
    parser.add_argument("--download", dest="download", action="store_true")
    parser.add_argument("--no-download", dest="download", action="store_false")
    parser.set_defaults(download=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "demo_output" / "camera_ps_pl_partitioned")
    parser.add_argument(
        "--save-video",
        type=Path,
        default=REPO_ROOT / "demo_output" / "camera_ps_pl_partitioned" / "camera_preview.avi",
    )
    parser.add_argument("--no-save-video", dest="save_video", action="store_const", const=None)
    return parser.parse_args()


def build_preview(
    source_bgr,
    gray_u8,
    edge_i8,
    *,
    detections,
    frame_index: int,
    inference_ms: float,
    loop_fps: float,
    preview_scale: float,
    preview_layout: str,
):
    preview = draw_detections(source_bgr, detections)
    edge_u8 = cv2.convertScaleAbs(edge_i8.astype("float32"), alpha=1.0, beta=128.0)
    if preview_layout == "panels":
        gray_bgr = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
        edge_bgr = cv2.cvtColor(edge_u8, cv2.COLOR_GRAY2BGR)
        thumb_h = max(80, preview.shape[0] // 4)
        thumb_w = max(80, int(round(thumb_h * float(gray_u8.shape[1]) / max(gray_u8.shape[0], 1))))
        gray_thumb = cv2.resize(gray_bgr, (thumb_w, thumb_h), interpolation=cv2.INTER_NEAREST)
        edge_thumb = cv2.resize(edge_bgr, (thumb_w, thumb_h), interpolation=cv2.INTER_NEAREST)
        panel = cv2.vconcat([gray_thumb, edge_thumb])
        y1 = 8
        x1 = max(8, preview.shape[1] - panel.shape[1] - 8)
        y2 = min(preview.shape[0], y1 + panel.shape[0])
        x2 = min(preview.shape[1], x1 + panel.shape[1])
        if y2 > y1 and x2 > x1:
            cv2.rectangle(preview, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (18, 18, 18), thickness=-1)
            preview[y1:y2, x1:x2] = panel[: y2 - y1, : x2 - x1]
            cv2.putText(preview, "gray", (x1 + 6, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(
                preview,
                "edge",
                (x1 + 6, y1 + gray_thumb.shape[0] + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    if abs(float(preview_scale) - 1.0) > 1e-6:
        stacked = cv2.resize(
            preview,
            (
                max(1, int(round(preview.shape[1] * float(preview_scale)))),
                max(1, int(round(preview.shape[0] * float(preview_scale)))),
            ),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        stacked = preview
    cv2.rectangle(stacked, (0, 0), (stacked.shape[1], 56), (18, 18, 18), thickness=-1)
    cv2.putText(stacked, "FullHW Detector", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    status = f"frame={frame_index} infer={inference_ms:.1f}ms fps={loop_fps:.2f} dets={len(detections)}"
    cv2.putText(stacked, status, (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 255, 140), 1, cv2.LINE_AA)
    return stacked


def save_preview_metadata(
    output_dir: Path,
    *,
    preview_bgr,
    metadata: dict,
    jpeg_quality: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = output_dir / "camera_preview_latest.jpg"
    metadata_path = output_dir / "camera_latest_metadata.json"
    quality = int(max(30, min(95, int(jpeg_quality))))
    cv2.imwrite(str(preview_path), preview_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return {
        "preview_path": str(preview_path),
        "metadata_path": str(metadata_path),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay = FullHwDetectorOverlay(args.bitfile, download=args.download)
    spec = overlay.spec
    display_window = bool(args.display_window and os.environ.get("DISPLAY"))
    cap = open_camera(
        args.camera_index,
        capture_width=args.capture_width,
        capture_height=args.capture_height,
        requested_fps=args.fps,
    )

    writer = None
    start_time = time.perf_counter()
    last_frame_time = start_time
    processed_frames = 0
    capture_failures = 0
    latest_paths = {}

    try:
        for _ in range(args.warmup_frames):
            ok, _ = cap.read()
            if not ok:
                capture_failures += 1

        while args.max_frames == 0 or processed_frames < args.max_frames:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                capture_failures += 1
                if capture_failures >= 3:
                    raise RuntimeError("camera capture failed repeatedly")
                continue

            frame_index = processed_frames + 1
            infer_start = time.perf_counter()
            if spec.in_channels == 2:
                tensor_chw, gray_u8 = prepare_camera_tensor(
                    frame_bgr,
                    process_width=spec.img_width,
                    process_height=spec.img_height,
                )
                output_plane = tensor_chw[1]
            elif spec.in_channels == 3:
                tensor_chw, rgb_u8 = prepare_camera_rgb_tensor(
                    frame_bgr,
                    process_width=spec.img_width,
                    process_height=spec.img_height,
                )
                gray_u8 = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
                output_plane = tensor_chw[0]
            else:
                raise RuntimeError(f"camera pipeline supports only 2 or 3 input channels, overlay reports {spec.in_channels}")
            parsed = overlay.run_tensor(tensor_chw)
            detections = det1_packet_to_detections(
                parsed,
                original_shape=frame_bgr.shape[:2],
                class_names=tuple(args.class_names),
                score_threshold=args.score_threshold,
                max_detections=args.max_detections,
                iou_threshold=args.iou_threshold,
                class_agnostic_nms=args.class_agnostic_nms,
            )
            preview_detections = filter_preview_detections(
                detections,
                image_shape=frame_bgr.shape[:2],
                min_area_ratio=args.preview_min_area_ratio,
                max_area_ratio=args.preview_max_area_ratio,
                max_aspect_ratio=args.preview_max_aspect_ratio,
                border_margin_ratio=args.preview_border_margin_ratio,
                top_k=args.preview_top_k,
            )
            infer_end = time.perf_counter()

            now = infer_end
            loop_fps = 1.0 / max(now - last_frame_time, 1e-9)
            last_frame_time = now

            preview_bgr = build_preview(
                frame_bgr,
                gray_u8,
                output_plane,
                detections=preview_detections,
                frame_index=frame_index,
                inference_ms=(infer_end - infer_start) * 1000.0,
                loop_fps=loop_fps,
                preview_scale=args.preview_scale,
                preview_layout=args.preview_layout,
            )

            if writer is None and args.save_video is not None:
                writer = create_video_writer(args.save_video, preview_bgr.shape, fps=max(loop_fps, args.fps, 1.0))
            if writer is not None:
                writer.write(preview_bgr)

            metadata = {
                "bitfile": str(args.bitfile),
                "frame_index": frame_index,
                "inference_ms": (infer_end - infer_start) * 1000.0,
                "loop_fps": loop_fps,
                "capture_size": [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))],
                "process_size": [spec.img_height, spec.img_width],
                "detector_spec": {
                    "img_width": spec.img_width,
                    "img_height": spec.img_height,
                    "in_channels": spec.in_channels,
                    "record_bytes": spec.record_bytes,
                    "output_bytes": spec.output_bytes,
                },
                "dma_counters": overlay.read_counters(),
                "raw_detection_count": len(detections),
                "preview_detection_count": len(preview_detections),
                "detections": [det.to_dict() for det in preview_detections],
            }

            if args.save_every > 0 and frame_index % args.save_every == 0:
                if args.preview_only_artifacts:
                    latest_paths = save_preview_metadata(
                        args.output_dir,
                        preview_bgr=preview_bgr,
                        metadata=metadata,
                        jpeg_quality=args.preview_jpeg_quality,
                    )
                else:
                    latest_paths = save_latest_artifacts(
                        args.output_dir,
                        input_u8=gray_u8,
                        output_i8=output_plane,
                        preview_bgr=preview_bgr,
                        metadata=metadata,
                    )

            if display_window:
                cv2.imshow("fullhw_detector_camera", preview_bgr)
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
        overlay.close()

    total_time = time.perf_counter() - start_time
    result = {
        "bitfile": str(args.bitfile),
        "frames_processed": processed_frames,
        "capture_failures": capture_failures,
        "average_fps": processed_frames / max(total_time, 1e-9),
        "elapsed_seconds": total_time,
        "camera_index": args.camera_index,
        "capture_size": [args.capture_height, args.capture_width],
        "process_size": [spec.img_height, spec.img_width],
        "display_window": display_window,
        "latest_artifacts": latest_paths,
    }
    result_path = args.output_dir / "camera_run_result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=True))
    print(f"saved_result={result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

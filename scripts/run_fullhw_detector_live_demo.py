from __future__ import annotations

import argparse
import json
import posixpath
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import paramiko


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display the board-side fullhw detector camera preview live on the host PC."
    )
    parser.add_argument("--board-host", default="192.168.2.99")
    parser.add_argument("--board-user", default="xilinx")
    parser.add_argument("--board-password", default="xilinx")
    parser.add_argument("--remote-base", default="/home/xilinx/jupyter_notebooks/ps_pl_partitioned_detector_scheme")
    parser.add_argument(
        "--bitfile",
        default="hardware/overlay/yolo_pynq_z2_fullhw_plonly_demo_cam32.bit",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--capture-width", type=int, default=640)
    parser.add_argument("--capture-height", type=int, default=480)
    parser.add_argument("--warmup-frames", type=int, default=1)
    parser.add_argument("--score-threshold", type=float, default=0.60)
    parser.add_argument("--iou-threshold", type=float, default=0.20)
    parser.add_argument("--max-detections", type=int, default=6)
    parser.add_argument("--preview-min-area-ratio", type=float, default=0.03)
    parser.add_argument("--preview-max-area-ratio", type=float, default=0.40)
    parser.add_argument("--preview-max-aspect-ratio", type=float, default=3.0)
    parser.add_argument("--preview-border-margin-ratio", type=float, default=0.02)
    parser.add_argument("--preview-top-k", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--preview-scale", type=float, default=1.0)
    parser.add_argument("--poll-interval", type=float, default=0.05)
    parser.add_argument("--window-name", default="ps_pl_partitioned_live")
    parser.add_argument("--ui-mode", choices=("dashboard", "plain"), default="dashboard")
    parser.add_argument("--output-dir-name", default="camera_ps_pl_partitioned_live")
    parser.add_argument(
        "--local-output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "demo_output" / "camera_ps_pl_partitioned_live",
    )
    return parser.parse_args()


def _drain_stream(stream, prefix: str, stop_flag: dict[str, bool]) -> None:
    while not stop_flag["stop"]:
        if stream.channel.exit_status_ready() and not stream.channel.recv_ready() and not stream.channel.recv_stderr_ready():
            break
        line = stream.readline()
        if not line:
            time.sleep(0.05)
            continue
        text = line.rstrip()
        if text:
            print(f"{prefix}{text}")


def _read_remote_bytes(sftp: paramiko.SFTPClient, remote_path: str) -> bytes | None:
    try:
        with sftp.open(remote_path, "rb") as fh:
            return fh.read()
    except OSError:
        return None


def _decode_image(encoded: bytes | None) -> np.ndarray | None:
    if not encoded:
        return None
    arr = np.frombuffer(encoded, dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _load_remote_metadata(sftp: paramiko.SFTPClient, remote_path: str) -> dict | None:
    raw = _read_remote_bytes(sftp, remote_path)
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def _format_header(metadata: dict | None) -> str:
    if not metadata:
        return "ps/pl live waiting for first frame"
    frame_index = int(metadata.get("frame_index", 0))
    inference_ms = float(metadata.get("inference_ms", 0.0))
    detections = metadata.get("detections") or []
    return f"ps/pl live frame={frame_index} infer={inference_ms:.1f}ms dets={len(detections)}"


def _draw_header(frame_bgr: np.ndarray, header: str) -> np.ndarray:
    canvas = np.asarray(frame_bgr).copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 36), (18, 18, 18), thickness=-1)
    cv2.putText(canvas, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 255, 220), 2, cv2.LINE_AA)
    return canvas


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _draw_gradient_background(canvas: np.ndarray) -> None:
    height, width = canvas.shape[:2]
    top_color = np.array([18.0, 24.0, 33.0], dtype=np.float32)
    bottom_color = np.array([42.0, 70.0, 89.0], dtype=np.float32)
    alpha = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    gradient = ((1.0 - alpha) * top_color + alpha * bottom_color).astype(np.uint8)
    canvas[:] = gradient
    for x0 in range(-height, width, 88):
        cv2.line(canvas, (x0, 0), (x0 + height, height), (28, 42, 56), 1, cv2.LINE_AA)


def _draw_metric_card(
    canvas: np.ndarray,
    *,
    x: int,
    y: int,
    w: int,
    h: int,
    label: str,
    value: str,
    accent_bgr: tuple[int, int, int],
) -> None:
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (24, 33, 44), thickness=-1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), accent_bgr, thickness=1)
    cv2.rectangle(canvas, (x, y), (x + 6, y + h), accent_bgr, thickness=-1)
    cv2.putText(canvas, label, (x + 12, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (166, 188, 208), 1, cv2.LINE_AA)
    cv2.putText(canvas, value, (x + 12, y + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (238, 248, 255), 2, cv2.LINE_AA)


def _draw_detection_rows(canvas: np.ndarray, panel_x: int, start_y: int, panel_w: int, detections: list[dict]) -> int:
    row_x = panel_x + 14
    row_w = panel_w - 28
    row_h = 48
    row_gap = 8
    max_rows = 6
    shown = detections[:max_rows]

    if not shown:
        cv2.rectangle(canvas, (row_x, start_y), (row_x + row_w, start_y + row_h), (27, 36, 47), thickness=-1)
        cv2.putText(
            canvas,
            "No detection rows yet",
            (row_x + 12, start_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (148, 167, 183),
            1,
            cv2.LINE_AA,
        )
        return start_y + row_h + row_gap

    for idx, det in enumerate(shown):
        y = start_y + idx * (row_h + row_gap)
        cv2.rectangle(canvas, (row_x, y), (row_x + row_w, y + row_h), (27, 36, 47), thickness=-1)
        cls_name = str(det.get("class_name", f"class_{_safe_int(det.get('class_id'), -1)}"))
        score = max(0.0, min(1.0, _safe_float(det.get("score"), 0.0)))
        bbox = det.get("bbox_xyxy") or [0.0, 0.0, 0.0, 0.0]
        if len(bbox) >= 4:
            x1 = _safe_int(round(float(bbox[0])))
            y1 = _safe_int(round(float(bbox[1])))
            x2 = _safe_int(round(float(bbox[2])))
            y2 = _safe_int(round(float(bbox[3])))
        else:
            x1 = y1 = x2 = y2 = 0

        cv2.putText(
            canvas,
            f"{idx + 1:02d}  {cls_name}",
            (row_x + 10, y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (226, 238, 248),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"score {score:.2f}",
            (row_x + 10, y + 39),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (138, 255, 209),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"[{x1},{y1},{x2},{y2}]",
            (row_x + row_w - 170, y + 39),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (144, 170, 191),
            1,
            cv2.LINE_AA,
        )
        bar_x = row_x + 102
        bar_y = y + 26
        bar_w = row_w - 286
        bar_h = 10
        if bar_w > 6:
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (41, 50, 60), thickness=-1)
            fill_w = max(1, int(round(bar_w * score)))
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (84, 240, 189), thickness=-1)
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (75, 90, 108), thickness=1)
    return start_y + len(shown) * (row_h + row_gap)


def _compose_dashboard(
    preview_bgr: np.ndarray | None,
    metadata: dict | None,
    *,
    board_host: str,
    status_text: str,
) -> np.ndarray:
    waiting = preview_bgr is None
    if preview_bgr is None:
        preview_bgr = np.zeros((480, 640, 3), dtype=np.uint8)

    video = np.asarray(preview_bgr).copy()
    video_h, video_w = video.shape[:2]
    if video_h < 360:
        scale = 360.0 / max(float(video_h), 1.0)
        video = cv2.resize(
            video,
            (max(1, int(round(video_w * scale))), 360),
            interpolation=cv2.INTER_LINEAR,
        )
        video_h, video_w = video.shape[:2]

    margin = 18
    title_h = 76
    panel_w = 380
    canvas_h = title_h + margin * 2 + video_h + margin
    canvas_w = margin * 3 + video_w + panel_w
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    _draw_gradient_background(canvas)

    cv2.rectangle(canvas, (0, 0), (canvas_w, title_h), (12, 17, 24), thickness=-1)
    cv2.putText(
        canvas,
        "PS/PL Detector Dashboard",
        (margin, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.86,
        (239, 248, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"{status_text}  |  {time.strftime('%Y-%m-%d %H:%M:%S')}",
        (margin, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (162, 187, 205),
        1,
        cv2.LINE_AA,
    )

    video_x = margin
    video_y = title_h + margin
    cv2.rectangle(canvas, (video_x - 3, video_y - 3), (video_x + video_w + 3, video_y + video_h + 3), (10, 12, 16), thickness=-1)
    cv2.rectangle(canvas, (video_x - 3, video_y - 3), (video_x + video_w + 3, video_y + video_h + 3), (58, 86, 109), thickness=1)
    canvas[video_y : video_y + video_h, video_x : video_x + video_w] = video

    if waiting:
        cv2.rectangle(canvas, (video_x, video_y), (video_x + video_w, video_y + video_h), (6, 10, 14), thickness=-1)
        cv2.putText(
            canvas,
            "Waiting for first frame from board...",
            (video_x + 32, video_y + video_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (182, 209, 229),
            2,
            cv2.LINE_AA,
        )

    panel_x = video_x + video_w + margin
    panel_y = video_y
    panel_h = video_h
    cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (16, 24, 34), thickness=-1)
    cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (53, 74, 92), thickness=1)

    dets = list((metadata or {}).get("detections") or [])
    frame_index = _safe_int((metadata or {}).get("frame_index"), 0)
    infer_ms = _safe_float((metadata or {}).get("inference_ms"), 0.0)
    loop_fps = _safe_float((metadata or {}).get("loop_fps"), 0.0)
    preview_count = _safe_int((metadata or {}).get("preview_detection_count"), len(dets))
    raw_count = _safe_int((metadata or {}).get("raw_detection_count"), preview_count)
    process_size = (metadata or {}).get("process_size") or [0, 0]
    proc_h = _safe_int(process_size[0] if len(process_size) > 0 else 0)
    proc_w = _safe_int(process_size[1] if len(process_size) > 1 else 0)

    gap = 10
    card_w = (panel_w - gap * 3) // 2
    card_h = 82
    cards_y = panel_y + 12
    _draw_metric_card(canvas, x=panel_x + gap, y=cards_y, w=card_w, h=card_h, label="INFERENCE", value=f"{infer_ms:5.1f} ms", accent_bgr=(95, 202, 244))
    _draw_metric_card(
        canvas,
        x=panel_x + gap * 2 + card_w,
        y=cards_y,
        w=card_w,
        h=card_h,
        label="FPS",
        value=f"{loop_fps:5.2f}",
        accent_bgr=(82, 226, 172),
    )
    _draw_metric_card(
        canvas,
        x=panel_x + gap,
        y=cards_y + card_h + gap,
        w=card_w,
        h=card_h,
        label="PREVIEW DETS",
        value=str(preview_count),
        accent_bgr=(121, 180, 255),
    )
    _draw_metric_card(
        canvas,
        x=panel_x + gap * 2 + card_w,
        y=cards_y + card_h + gap,
        w=card_w,
        h=card_h,
        label="RAW DETS",
        value=str(raw_count),
        accent_bgr=(255, 181, 112),
    )

    title_y = cards_y + 2 * (card_h + gap) + 26
    cv2.putText(canvas, "TOP DETECTIONS", (panel_x + 14, title_y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (198, 219, 235), 1, cv2.LINE_AA)
    next_y = _draw_detection_rows(canvas, panel_x, title_y + 10, panel_w, dets)

    dma = (metadata or {}).get("dma_counters") or {}
    status_reg = _safe_int(dma.get("status_reg"), 0)
    in_count = _safe_int(dma.get("in_count_reg"), 0)
    out_count = _safe_int(dma.get("out_count_reg"), 0)
    footer_y = min(panel_y + panel_h - 56, max(next_y + 12, panel_y + panel_h - 56))
    cv2.line(canvas, (panel_x + 10, footer_y - 8), (panel_x + panel_w - 10, footer_y - 8), (56, 79, 99), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"frame={frame_index}  proc={proc_w}x{proc_h}",
        (panel_x + 14, footer_y + 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (164, 185, 203),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"DMA in={in_count} out={out_count} status=0x{status_reg:08X}",
        (panel_x + 14, footer_y + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (146, 171, 192),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"board={board_host}",
        (panel_x + 14, footer_y + 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (128, 157, 179),
        1,
        cv2.LINE_AA,
    )

    return canvas


def _ensure_local_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_local(path: Path, data: bytes | None) -> None:
    if data is None:
        return
    path.write_bytes(data)


def main() -> int:
    args = parse_args()
    _ensure_local_dir(args.local_output_dir)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        args.board_host,
        username=args.board_user,
        password=args.board_password,
        timeout=10,
        banner_timeout=10,
        auth_timeout=10,
    )

    cleanup_cmd = "echo xilinx | sudo -S pkill -f run_fullhw_detector_camera.py"
    cleanup_stdin, cleanup_stdout, cleanup_stderr = client.exec_command(cleanup_cmd, get_pty=True, timeout=60)
    cleanup_stdout.read()
    cleanup_stderr.read()
    time.sleep(0.5)

    remote_output_dir = f"demo_output/{args.output_dir_name}"
    remote_preview_path = posixpath.join(args.remote_base, remote_output_dir, "camera_preview_latest.jpg")
    remote_metadata_path = posixpath.join(args.remote_base, remote_output_dir, "camera_latest_metadata.json")
    remote_result_path = posixpath.join(args.remote_base, remote_output_dir, "camera_run_result.json")

    cmd = (
        f"cd {args.remote_base} && "
        f"mkdir -p {remote_output_dir} && "
        "echo xilinx | sudo -S env XILINX_XRT=/usr "
        "/usr/local/share/pynq-venv/bin/python scripts/run_fullhw_detector_camera.py "
        f"--bitfile {args.bitfile} "
        f"--camera-index {int(args.camera_index)} "
        f"--capture-width {int(args.capture_width)} "
        f"--capture-height {int(args.capture_height)} "
        f"--warmup-frames {int(args.warmup_frames)} "
        f"--score-threshold {float(args.score_threshold)} "
        f"--iou-threshold {float(args.iou_threshold)} "
        f"--max-detections {int(args.max_detections)} "
        f"--preview-min-area-ratio {float(args.preview_min_area_ratio)} "
        f"--preview-max-area-ratio {float(args.preview_max_area_ratio)} "
        f"--preview-max-aspect-ratio {float(args.preview_max_aspect_ratio)} "
        f"--preview-border-margin-ratio {float(args.preview_border_margin_ratio)} "
        f"--preview-top-k {int(args.preview_top_k)} "
        "--class-agnostic-nms "
        f"--save-every {int(args.save_every)} "
        "--preview-layout source_only "
        "--preview-only-artifacts "
        "--preview-jpeg-quality 75 "
        "--preview-scale 1 "
        "--max-frames 0 "
        f"--output-dir {remote_output_dir} "
        "--no-save-video"
    )

    channel = client.get_transport().open_session()
    channel.get_pty()
    channel.exec_command(cmd)
    stdout = channel.makefile("r")
    stderr = channel.makefile_stderr("r")
    stop_flag = {"stop": False}
    stdout_thread = threading.Thread(target=_drain_stream, args=(stdout, "[board] ", stop_flag), daemon=True)
    stderr_thread = threading.Thread(target=_drain_stream, args=(stderr, "[board-err] ", stop_flag), daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    sftp = client.open_sftp()
    last_preview_mtime: int | None = None
    last_frame_index = -1

    try:
        while True:
            if channel.exit_status_ready() and last_frame_index >= 0:
                break

            preview_stat = None
            try:
                preview_stat = sftp.stat(remote_preview_path)
            except OSError:
                preview_stat = None

            metadata = _load_remote_metadata(sftp, remote_metadata_path)
            if metadata is not None:
                last_frame_index = max(last_frame_index, int(metadata.get("frame_index", -1)))

            if preview_stat is None:
                status = _format_header(metadata)
                if args.ui_mode == "dashboard":
                    ui_frame = _compose_dashboard(None, metadata, board_host=args.board_host, status_text=status)
                else:
                    blank = np.zeros((240, 640, 3), dtype=np.uint8)
                    ui_frame = _draw_header(blank, status)
                if abs(float(args.preview_scale) - 1.0) > 1e-6:
                    width = max(1, int(round(ui_frame.shape[1] * float(args.preview_scale))))
                    height = max(1, int(round(ui_frame.shape[0] * float(args.preview_scale))))
                    ui_frame = cv2.resize(ui_frame, (width, height), interpolation=cv2.INTER_LINEAR)
                cv2.imshow(args.window_name, ui_frame)
                key = cv2.waitKey(100) & 0xFF
                if key in (27, ord("q")):
                    break
                time.sleep(float(args.poll_interval))
                continue

            current_mtime = int(preview_stat.st_mtime)
            if last_preview_mtime is not None and current_mtime == last_preview_mtime:
                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord("q")):
                    break
                time.sleep(float(args.poll_interval))
                continue

            last_preview_mtime = current_mtime
            preview_bytes = _read_remote_bytes(sftp, remote_preview_path)
            preview_bgr = _decode_image(preview_bytes)
            if preview_bgr is None:
                time.sleep(float(args.poll_interval))
                continue

            _write_local(args.local_output_dir / "camera_preview_latest.jpg", preview_bytes)
            metadata_bytes = _read_remote_bytes(sftp, remote_metadata_path)
            _write_local(args.local_output_dir / "camera_latest_metadata.json", metadata_bytes)
            result_bytes = _read_remote_bytes(sftp, remote_result_path)
            _write_local(args.local_output_dir / "camera_run_result.json", result_bytes)

            header = _format_header(metadata)
            if args.ui_mode == "dashboard":
                preview_bgr = _compose_dashboard(
                    preview_bgr,
                    metadata,
                    board_host=args.board_host,
                    status_text=header,
                )
            else:
                preview_bgr = _draw_header(preview_bgr, header)
            if abs(float(args.preview_scale) - 1.0) > 1e-6:
                width = max(1, int(round(preview_bgr.shape[1] * float(args.preview_scale))))
                height = max(1, int(round(preview_bgr.shape[0] * float(args.preview_scale))))
                preview_bgr = cv2.resize(preview_bgr, (width, height), interpolation=cv2.INTER_LINEAR)

            cv2.imshow(args.window_name, preview_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            time.sleep(float(args.poll_interval))
    except KeyboardInterrupt:
        pass
    finally:
        stop_flag["stop"] = True
        try:
            channel.send("\x03")
        except Exception:
            pass
        time.sleep(0.3)
        try:
            channel.close()
        except Exception:
            pass
        try:
            sftp.close()
        except Exception:
            pass
        client.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

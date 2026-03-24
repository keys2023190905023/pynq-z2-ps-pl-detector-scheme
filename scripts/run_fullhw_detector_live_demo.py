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
                blank = np.zeros((240, 640, 3), dtype=np.uint8)
                blank = _draw_header(blank, _format_header(metadata))
                cv2.imshow(args.window_name, blank)
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

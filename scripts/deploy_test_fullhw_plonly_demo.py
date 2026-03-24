from __future__ import annotations

import argparse
import json
import os
import posixpath
import sys
from pathlib import Path

import paramiko


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy the PL-only fullhw demo overlay to PYNQ-Z2 and run board tests.")
    parser.add_argument("--host", default="192.168.2.99")
    parser.add_argument("--username", default="xilinx")
    parser.add_argument("--password", default="xilinx")
    parser.add_argument("--remote-root", default="/home/xilinx/jupyter_notebooks/ps_pl_partitioned_detector_scheme")
    parser.add_argument(
        "--overlay-basename",
        default="yolo_pynq_z2_fullhw_plonly_demo_cam32",
    )
    parser.add_argument(
        "--local-output-dir",
        type=Path,
        default=REPO_ROOT / "demo_output" / "fullhw_plonly_demo_board",
    )
    parser.add_argument("--run-camera", action="store_true")
    parser.add_argument("--camera-frames", type=int, default=2)
    return parser.parse_args()


def remote_join(*parts: str) -> str:
    return posixpath.join(*parts)


def ensure_remote_dirs(ssh: paramiko.SSHClient, remote_root: str) -> None:
    cmd = (
        f"mkdir -p {remote_join(remote_root, 'hardware', 'overlay')} "
        f"{remote_join(remote_root, 'python', 'yolov8_pynq_z2')} "
        f"{remote_join(remote_root, 'scripts')} "
        f"{remote_join(remote_root, 'demo_output')}"
    )
    run_remote(ssh, cmd)


def run_remote(ssh: paramiko.SSHClient, command: str, timeout: int = 600) -> tuple[int, str, str]:
    stdin, stdout, stderr = ssh.exec_command(command, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return exit_code, out, err


def sftp_put(sftp: paramiko.SFTPClient, local_path: Path, remote_path: str) -> None:
    sftp.put(str(local_path), remote_path)


def sftp_get_if_exists(sftp: paramiko.SFTPClient, remote_path: str, local_path: Path) -> bool:
    try:
        sftp.stat(remote_path)
    except OSError:
        return False
    local_path.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(remote_path, str(local_path))
    return True


def main() -> int:
    args = parse_args()
    args.local_output_dir.mkdir(parents=True, exist_ok=True)

    overlay_dir = REPO_ROOT / "hardware" / "overlay"
    files_to_upload = [
        (
            overlay_dir / f"{args.overlay_basename}.bit",
            remote_join(args.remote_root, "hardware", "overlay", f"{args.overlay_basename}.bit"),
        ),
        (
            overlay_dir / f"{args.overlay_basename}.hwh",
            remote_join(args.remote_root, "hardware", "overlay", f"{args.overlay_basename}.hwh"),
        ),
        (
            REPO_ROOT / "python" / "yolov8_pynq_z2" / "fullhw_detector.py",
            remote_join(args.remote_root, "python", "yolov8_pynq_z2", "fullhw_detector.py"),
        ),
        (
            REPO_ROOT / "python" / "yolov8_pynq_z2" / "detections.py",
            remote_join(args.remote_root, "python", "yolov8_pynq_z2", "detections.py"),
        ),
        (
            REPO_ROOT / "scripts" / "board_smoketest_fullhw_detector.py",
            remote_join(args.remote_root, "scripts", "board_smoketest_fullhw_detector.py"),
        ),
        (
            REPO_ROOT / "scripts" / "run_fullhw_detector_camera.py",
            remote_join(args.remote_root, "scripts", "run_fullhw_detector_camera.py"),
        ),
    ]

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        args.host,
        username=args.username,
        password=args.password,
        timeout=15,
        banner_timeout=15,
        auth_timeout=15,
    )
    sftp = ssh.open_sftp()

    try:
        ensure_remote_dirs(ssh, args.remote_root)
        for local_path, remote_path in files_to_upload:
            sftp_put(sftp, local_path, remote_path)

        smoke_remote = remote_join(args.remote_root, "demo_output", "board_smoketest_fullhw_plonly_demo_cam32.json")
        smoke_cmd = (
            f"cd {args.remote_root} && "
            "echo xilinx | sudo -S env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python "
            f"scripts/board_smoketest_fullhw_detector.py "
            f"--bitfile hardware/overlay/{args.overlay_basename}.bit "
            f"--output {smoke_remote} "
            "--reference-mode auto"
        )
        smoke_code, smoke_out, smoke_err = run_remote(ssh, smoke_cmd, timeout=1200)

        smoke_local = args.local_output_dir / "board_smoketest_fullhw_plonly_demo_cam32.json"
        sftp_get_if_exists(sftp, smoke_remote, smoke_local)

        result: dict[str, object] = {
            "smoke_exit_code": smoke_code,
            "smoke_stdout": smoke_out,
            "smoke_stderr": smoke_err,
            "smoke_local": str(smoke_local),
        }

        if args.run_camera:
            camera_remote_dir = remote_join(args.remote_root, "demo_output", "camera_fullhw_plonly_demo_cam32")
            camera_cmd = (
                f"cd {args.remote_root} && "
                "echo xilinx | sudo -S env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python "
                "scripts/run_fullhw_detector_camera.py "
                f"--bitfile hardware/overlay/{args.overlay_basename}.bit "
                "--download "
                "--camera-index 0 "
                "--capture-width 640 "
                "--capture-height 480 "
                f"--max-frames {args.camera_frames} "
                "--warmup-frames 1 "
                "--save-every 1 "
                "--score-threshold 0.60 "
                "--iou-threshold 0.20 "
                "--max-detections 6 "
                "--class-agnostic-nms "
                "--preview-only-artifacts "
                f"--output-dir {camera_remote_dir}"
            )
            cam_code, cam_out, cam_err = run_remote(ssh, camera_cmd, timeout=1800)
            result.update(
                {
                    "camera_exit_code": cam_code,
                    "camera_stdout": cam_out,
                    "camera_stderr": cam_err,
                }
            )
            for name in (
                "camera_run_result.json",
                "camera_latest_metadata.json",
                "camera_preview_latest.jpg",
                "camera_input_latest.png",
                "camera_output_latest.png",
            ):
                sftp_get_if_exists(sftp, remote_join(camera_remote_dir, name), args.local_output_dir / name)

        print(json.dumps(result, indent=2, ensure_ascii=True))
        return 0 if smoke_code == 0 else smoke_code
    finally:
        try:
            sftp.close()
        finally:
            ssh.close()


if __name__ == "__main__":
    raise SystemExit(main())

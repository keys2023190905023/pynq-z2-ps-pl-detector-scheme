from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from yolov8_pynq_z2.fullhw_detector import FullHwDetectorOverlay, build_fullhw_reference_packet, make_synthetic_input
from yolov8_pynq_z2.fullhw_detector import build_pl_only_demo_reference_packet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Board smoke test for the fullhw detector overlay.")
    parser.add_argument(
        "--bitfile",
        type=Path,
        default=REPO_ROOT / "hardware" / "overlay" / "yolo_pynq_z2_fullhw_detector.bit",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "demo_output" / "board_smoketest_fullhw_detector.json",
    )
    parser.add_argument("--download", dest="download", action="store_true")
    parser.add_argument("--no-download", dest="download", action="store_false")
    parser.add_argument(
        "--reference-mode",
        choices=("auto", "direct_head", "pl_only_demo"),
        default="auto",
    )
    parser.set_defaults(download=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    overlay = FullHwDetectorOverlay(args.bitfile, download=args.download)
    src = make_synthetic_input(overlay.spec)
    if args.reference_mode == "auto":
        reference_mode = "pl_only_demo" if overlay.spec.in_channels == 3 else "direct_head"
    else:
        reference_mode = args.reference_mode

    if reference_mode == "pl_only_demo":
        expected = build_pl_only_demo_reference_packet(src)
    else:
        expected = build_fullhw_reference_packet(
            src,
            num_anchors=overlay.spec.num_anchors,
            box_params=overlay.spec.box_params,
            num_classes=overlay.spec.num_classes,
        )
    actual_packet = overlay.run_tensor(src).packet
    match = bool(np.array_equal(actual_packet, expected))
    diff = actual_packet.astype(np.int16) - expected.astype(np.int16)
    counters = overlay.read_counters()

    payload = {
        "bitfile": str(args.bitfile),
        "bitfile_md5": hashlib.md5(args.bitfile.read_bytes()).hexdigest(),
        "download": bool(args.download),
        "img_width": overlay.spec.img_width,
        "img_height": overlay.spec.img_height,
        "in_channels": overlay.spec.in_channels,
        "record_bytes": overlay.spec.record_bytes,
        "input_size": int(src.size),
        "output_size": int(expected.size),
        "reference_mode": reference_mode,
        "match": match,
        "max_abs_diff": int(np.abs(diff).max()),
        "sum_abs_diff": int(np.abs(diff).sum()),
        "status_reg": counters["status_reg"],
        "in_count_reg": counters["in_count_reg"],
        "out_count_reg": counters["out_count_reg"],
        "expected": expected.tolist(),
        "actual": actual_packet.tolist(),
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0 if match else 1


if __name__ == "__main__":
    raise SystemExit(main())

# PS/PL Partitioned Detector Scheme (PYNQ-Z2)

English | [中文](README.zh-CN.md)

## 1. Project Scope

This repository is a packaged and reproducible PYNQ-Z2 delivery focused on an end-to-end on-board detection loop:

- `PL`: feature extraction, detector-head forward pass, and `DET1` packet output
- `PS`: camera capture, preprocessing, `DET1` decoding, NMS, visualization, and artifact saving
- `Host`: remote deployment, live preview, and result archiving

This folder is intentionally curated as a standalone deliverable, not a mixed historical workspace.

## 2. Repository Layout

- `hardware/overlay`: ready-to-deploy `bit/hwh`
- `hardware/vivado_src`: Vivado sources, RTL, testbenches, and Tcl rebuild scripts
- `python/yolov8_pynq_z2`: runtime modules (preprocess, DET1 decode, NMS, etc.)
- `scripts`: deployment, board smoke test, live preview, and publishing scripts
- `demo_output`: sample outputs and historical run artifacts
- `reports`: synthesis/implementation logs and timing reports

## 3. Current Delivery Status

- `cam32` overlay has been re-synthesized and implemented
- timing is closed:
  - `WNS = 3.313 ns`
  - `TNS = 0.000 ns`
- board smoke-test and camera sample outputs are included in `demo_output`

Overlay fingerprints:

- `yolo_pynq_z2_fullhw_plonly_demo_cam32.bit` MD5: `691CBDC2E3640C39A3871FCB443F13CB`
- `yolo_pynq_z2_fullhw_plonly_demo_cam32.hwh` MD5: `0B3F37054803B443F7A2C8496D02BB75`

## 4. Quick Start

Default assumptions:

- Board IP: `192.168.2.99`
- User: `xilinx`
- Password: `xilinx`

### 4.1 Deploy from host and start board-side run

```powershell
python .\scripts\deploy_test_fullhw_plonly_demo.py --run-camera
```

### 4.2 Run camera pipeline manually on board

```bash
cd /home/xilinx/jupyter_notebooks/ps_pl_partitioned_detector_scheme
echo xilinx | sudo -S env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python scripts/run_fullhw_detector_camera.py
```

### 4.3 Open live preview on host

```powershell
python .\scripts\run_fullhw_detector_live_demo.py --ui-mode dashboard
```

Optional:

- Minimal mode: `--ui-mode plain`
- Window scaling: `--preview-scale 1.2`

## 5. Advanced UI Notes

The default UI mode of `run_fullhw_detector_live_demo.py` is now an upgraded dashboard view with:

- top status strip (frame, inference latency, detections, timestamp)
- left live-video panel
- right metric cards (Inference/FPS/Preview Dets/Raw Dets)
- detection rows (class, score, bbox)
- DMA counters and board link status

This is a display-layer improvement only. It does not change the existing board-side algorithm chain.

## 6. Key Documents

- `ARCHITECTURE_PS_PL.md`: architecture rationale for PS/PL partitioning
- `RUN_ON_NEW_PC.md`: shortest 0-to-run path on a new PC
- `RESULTS_AND_TIMING.md`: run summary and timing evidence
- `CONTRIBUTING.md`: collaboration guidelines

## 7. Boundary Statement

Current `cam32` route demonstrates the feasibility of a “hardware forward path + software post-processing” architecture:

- PS/PL loop is stable and reproducible
- structured DET1 dataflow is validated
- it is not yet a fully hardwareized, fully trained person/helmet network

For continued high-hardware-ratio YOLOv8 work, this repo is the best baseline.

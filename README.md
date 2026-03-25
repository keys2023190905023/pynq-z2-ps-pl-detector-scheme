# PS/PL Detector Scheme

语言切换 / Language:

- 中文说明: [README.zh-CN.md](README.zh-CN.md)
- English Docs: [README.en.md](README.en.md)

This repository provides a delivered PS/PL partitioned detector scheme for PYNQ-Z2, including:

- PL-side forward path overlay and RTL/Vivado sources
- PS-side camera pipeline, DET1 decode, NMS, and runtime scripts
- Host-side deployment, live preview, and result collection tools

Quick start (host):

```powershell
python .\scripts\deploy_test_fullhw_plonly_demo.py --run-camera
python .\scripts\run_fullhw_detector_live_demo.py --ui-mode dashboard
```

The live demo script now supports two UI modes:

- `dashboard` (default): advanced monitoring-style interface
- `plain`: minimal OpenCV header overlay

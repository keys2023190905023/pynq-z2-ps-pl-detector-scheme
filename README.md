# PS/PL 分工检测方案交付包

这个目录是从主工程中单独整理出来的一套可交付方案，目标很明确：

- `PL` 负责特征提取、检测头前向和 `DET1` 检测包输出
- `PS` 负责摄像头采集、预处理、`DET1` 解码、NMS、显示与保存
- `Host` 负责部署、远程查看实时窗口和结果留档

这不是整个 `F:\yolo` 的历史混合目录，而是一份已经收口过、可以单独拷走的方案包。

## 目录结构

- `hardware/overlay`
  - 可直接上板的 overlay
  - 当前正式文件：
    - `yolo_pynq_z2_fullhw_plonly_demo_cam32.bit`
    - `yolo_pynq_z2_fullhw_plonly_demo_cam32.hwh`
- `hardware/vivado_src`
  - 这套方案需要的 Vivado 源码、RTL、testbench、BD Tcl、重建脚本
  - 已包含 `rtl_fullhw`、`rtl_fullhw_axi`、`rtl_pl_ps`
- `python/yolov8_pynq_z2`
  - 板端运行时、`DET1` 解码、NMS、相机流程
- `scripts`
  - 常用入口脚本
  - `deploy_test_fullhw_plonly_demo.py`
  - `run_fullhw_detector_camera.py`
  - `run_fullhw_detector_live_demo.py`
  - `board_smoketest_fullhw_detector.py`
  - `publish_to_github.ps1`
- `demo_output`
  - 已有板测结果、相机结果和吞吐参考
- `reports`
  - 这次重新综合后的 timing report 和 Vivado 构建日志

## 当前交付状态

- 这套包对应的 `cam32` overlay 已重新用 `Vivado 2019.1` 综合/实现
- 新 timing 已闭合：
  - `WNS = 3.313 ns`
  - `TNS = 0.000 ns`
- 板端 smoke test 样例已包含在：
  - `demo_output/fullhw_plonly_demo_board/board_smoketest_fullhw_plonly_demo_cam32.json`
- 短相机样例已包含在：
  - `demo_output/fullhw_plonly_demo_board/camera_run_result.json`
  - `demo_output/fullhw_plonly_demo_board/camera_latest_metadata.json`
  - `demo_output/fullhw_plonly_demo_board/camera_preview_latest.jpg`

## overlay 信息

- `bit` MD5: `691CBDC2E3640C39A3871FCB443F13CB`
- `hwh` MD5: `0B3F37054803B443F7A2C8496D02BB75`

## 快速使用

### 1. 部署到板子

默认假设：

- 板子 IP：`192.168.2.99`
- 用户名：`xilinx`
- 密码：`xilinx`

在主机上运行：

```powershell
python F:\yolo\ps_pl_partitioned_detector_scheme\scripts\deploy_test_fullhw_plonly_demo.py --run-camera
```

### 2. 板端直接跑相机

在板子上运行：

```powershell
cd /home/xilinx/jupyter_notebooks/ps_pl_partitioned_detector_scheme
echo xilinx | sudo -S env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python scripts/run_fullhw_detector_camera.py
```

### 3. 主机查看实时窗口

在主机上运行：

```powershell
python F:\yolo\ps_pl_partitioned_detector_scheme\scripts\run_fullhw_detector_live_demo.py
```

## 文档入口

- `ARCHITECTURE_PS_PL.md`
  - 解释为什么 `PL` 做前向、`PS` 做预处理和 NMS
- `RUN_ON_NEW_PC.md`
  - 新电脑从 0 到复现的最短步骤
- `RESULTS_AND_TIMING.md`
  - 板测、相机和 timing 结果摘要

## 开源协作文件

- `LICENSE`
  - 当前采用 `MIT License`
- `.gitignore`
  - 忽略 Vivado 构建缓存、Python 缓存和本地压缩产物
- `CONTRIBUTING.md`
  - 提交 issue、修改 RTL、补板测结果时的建议流程
- `scripts/publish_to_github.ps1`
  - 使用临时 `PAT` 创建 GitHub 仓库并推送，不在仓库内保存凭据

## 当前边界

这套包强调的是：

- `PL/PS` 分工清晰
- 纯 Verilog 前向链路已综合并上板
- `DET1` 结构化数据流和 `PS` 侧后处理已打通

需要如实说明：

- 这版 `cam32` 是 `demo detector head` 路线
- 它证明的是“硬件前向 + 软件后处理”的架构可行
- 它不是训练好的真实 `person/helmet` 模型全硬件版

如果后面继续推进真实 YOLOv8 的高硬件占比实现，这个目录就是最合适的独立起点。

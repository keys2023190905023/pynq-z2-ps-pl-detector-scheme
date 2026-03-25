# PS/PL 分工检测方案（PYNQ-Z2）

[English](README.en.md) | 中文

## 1. 项目定位

这是一个可交付、可复现的 PYNQ-Z2 检测方案包，核心目标是把板端流程做成闭环：

- `PL`：特征提取、检测头前向、`DET1` 检测包输出
- `PS`：摄像头采集、预处理、`DET1` 解码、NMS、可视化与结果保存
- `Host`：远程部署、实时预览、结果归档

这份目录不是历史混合工程，而是已经整理过的独立方案包，适合直接拷走复现。

## 2. 目录结构

- `hardware/overlay`：可直接上板的 `bit/hwh`
- `hardware/vivado_src`：Vivado 工程源码、RTL、testbench、Tcl 重建脚本
- `python/yolov8_pynq_z2`：板端运行时模块（预处理、DET1 解析、NMS 等）
- `scripts`：部署、板测、实时预览、发布脚本
- `demo_output`：样例输出与历史测试结果
- `reports`：综合/实现日志与 timing 报告

## 3. 当前交付状态

- `cam32` overlay 已重新综合实现
- timing 已闭合
  - `WNS = 3.313 ns`
  - `TNS = 0.000 ns`
- 板测与相机样例结果已随仓库提供（`demo_output`）

overlay 指纹：

- `yolo_pynq_z2_fullhw_plonly_demo_cam32.bit` MD5: `691CBDC2E3640C39A3871FCB443F13CB`
- `yolo_pynq_z2_fullhw_plonly_demo_cam32.hwh` MD5: `0B3F37054803B443F7A2C8496D02BB75`

## 4. 快速运行

默认参数：

- 板 IP：`192.168.2.99`
- 用户：`xilinx`
- 密码：`xilinx`

### 4.1 主机部署并触发板端运行

```powershell
python .\scripts\deploy_test_fullhw_plonly_demo.py --run-camera
```

### 4.2 板端手动运行相机链路

```bash
cd /home/xilinx/jupyter_notebooks/ps_pl_partitioned_detector_scheme
echo xilinx | sudo -S env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python scripts/run_fullhw_detector_camera.py
```

### 4.3 主机查看实时检测

```powershell
python .\scripts\run_fullhw_detector_live_demo.py --ui-mode dashboard
```

可选：

- 朴素模式：`--ui-mode plain`
- 缩放窗口：`--preview-scale 1.2`

## 5. 高级 UI 说明

`run_fullhw_detector_live_demo.py` 的默认界面已升级为 dashboard 风格，包含：

- 顶部状态条（帧号、推理耗时、检测数量、时间戳）
- 左侧实时视频区
- 右侧指标卡（Inference/FPS/Preview Dets/Raw Dets）
- 检测条目列表（class、score、bbox）
- DMA 计数与板端连接信息

该 UI 仅增强显示层，不改变你现有的板端算法链路和数据路径。

## 6. 关键文档

- `ARCHITECTURE_PS_PL.md`：PS/PL 划分设计说明
- `RUN_ON_NEW_PC.md`：新电脑从 0 到跑通
- `RESULTS_AND_TIMING.md`：结果摘要与 timing 数据
- `CONTRIBUTING.md`：协作与提交建议

## 7. 边界说明

当前 `cam32` 路线强调的是“硬件前向 + 软件后处理”的架构可行性：

- 已证明 PL/PS 分工闭环可稳定运行
- 已具备结构化 DET1 数据通路
- 不是训练好的真实 person/helmet 全网络纯硬件版本

如果后续继续推进高硬件占比 YOLOv8，本仓库就是最合适的工程起点。

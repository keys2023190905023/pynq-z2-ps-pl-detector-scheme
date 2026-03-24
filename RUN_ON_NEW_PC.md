# 新电脑复现步骤

这份说明假设你已经把整个目录复制到新电脑：

- `F:\yolo\ps_pl_partitioned_detector_scheme`

默认假设板子参数不变：

- IP：`192.168.2.99`
- 用户名：`xilinx`
- 密码：`xilinx`

## 1. 安装主机依赖

```powershell
cd F:\yolo\ps_pl_partitioned_detector_scheme
python -m pip install -r requirements-host.txt
```

## 2. 如果要重新综合 overlay

前提：

- 已安装 `Vivado 2019.1`

运行：

```powershell
& 'D:\Vivado\2019.1\bin\vivado.bat' -mode batch -source 'F:\yolo\ps_pl_partitioned_detector_scheme\hardware\vivado_src\rebuild_fullhw_pl_only_demo_overlay_clean.tcl'
```

如果只是想直接运行，可以跳过这一步，直接使用当前包内的：

- `hardware/overlay/yolo_pynq_z2_fullhw_plonly_demo_cam32.bit`
- `hardware/overlay/yolo_pynq_z2_fullhw_plonly_demo_cam32.hwh`

## 3. 部署到板子

```powershell
python F:\yolo\ps_pl_partitioned_detector_scheme\scripts\deploy_test_fullhw_plonly_demo.py
```

如果想部署后顺便跑一轮短相机：

```powershell
python F:\yolo\ps_pl_partitioned_detector_scheme\scripts\deploy_test_fullhw_plonly_demo.py --run-camera
```

## 4. 板端直接运行相机流程

```powershell
ssh xilinx@192.168.2.99
cd /home/xilinx/jupyter_notebooks/ps_pl_partitioned_detector_scheme
echo xilinx | sudo -S env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python scripts/run_fullhw_detector_camera.py
```

## 5. 主机查看实时窗口

```powershell
python F:\yolo\ps_pl_partitioned_detector_scheme\scripts\run_fullhw_detector_live_demo.py
```

## 6. 查看结果

板端短跑结果默认会落在：

- `demo_output/fullhw_plonly_demo_board`

重点文件：

- `board_smoketest_fullhw_plonly_demo_cam32.json`
- `camera_run_result.json`
- `camera_latest_metadata.json`
- `camera_preview_latest.jpg`

## 7. 如果只想检查 timing

直接看：

- `reports/fullhw_plonly_demo_timing_summary_routed.rpt`

关键结果：

- `WNS = 3.313 ns`
- `TNS = 0.000 ns`

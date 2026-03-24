# 结果与 Timing 摘要

## 板端功能结果

当前包内已经包含一轮成功样例：

- `demo_output/fullhw_plonly_demo_board/board_smoketest_fullhw_plonly_demo_cam32.json`

关键字段：

- `download = true`
- `match = true`
- `max_abs_diff = 0`
- `sum_abs_diff = 0`

说明：

- overlay 成功下载
- DMA 和寄存器链路工作正常
- `DET1` 输出与参考值一致

## 相机短跑样例

样例文件：

- `demo_output/fullhw_plonly_demo_board/camera_run_result.json`
- `demo_output/fullhw_plonly_demo_board/camera_latest_metadata.json`
- `demo_output/fullhw_plonly_demo_board/camera_preview_latest.jpg`

当前样例摘要：

- `frames_processed = 2`
- `capture_failures = 0`
- `average_fps ≈ 1.38`
- `capture_size = 640x480`
- `process_size = 32x32`

## 重新综合后的 Timing

本次重新综合/实现使用：

- `Vivado 2019.1`
- `hardware/vivado_src/rebuild_fullhw_pl_only_demo_overlay_clean.tcl`

报告文件：

- `reports/fullhw_plonly_demo_timing_summary_routed.rpt`

关键结果：

- `WNS = 3.313 ns`
- `TNS = 0.000 ns`
- `WHS = 0.043 ns`
- `THS = 0.000 ns`

结论：

- 当前 `50 MHz` 目标下 timing 已闭合
- 这套独立包里的 `cam32` overlay 已经是“源码、bit/hwh、timing 报告”一致的版本

## Overlay 哈希

- `yolo_pynq_z2_fullhw_plonly_demo_cam32.bit`
  - `691CBDC2E3640C39A3871FCB443F13CB`
- `yolo_pynq_z2_fullhw_plonly_demo_cam32.hwh`
  - `0B3F37054803B443F7A2C8496D02BB75`

## 还需要如实说明的边界

- 这条线是 `PL 前向 + PS 后处理` 的 demo detector 方案
- 它的价值在于：
  - 架构清晰
  - overlay 已可综合
  - 板端已可运行
  - timing 已闭合
- 它不是训练好的真实 `person/helmet` 模型全硬件版

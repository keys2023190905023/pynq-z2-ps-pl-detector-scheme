# PS/PL 架构说明

## 目标

这套方案的核心分工是：

- `PL`：特征提取、检测头前向、`DET1` 检测包输出
- `PS`：摄像头采集、预处理、`DET1` 解码、阈值筛选、NMS、显示与保存
- `Host`：部署、远程查看实时窗口、结果留档

这是一套典型的“规则前向在 FPGA，不规则后处理在 ARM”架构。

## 为什么预处理放在 PS

当前预处理主要是：

- `BGR -> GRAY`
- `resize`

这类逻辑的特点是：

- 算法强度不高
- 和摄像头格式、分辨率、实验参数强相关
- 调整频繁
- 在 `PS` 上实现成本低、灵活性高

如果把这些步骤硬塞进 `PL`，收益通常不大，但会增加：

- 视频输入接口适配复杂度
- RTL 验证成本
- 参数变化时的维护成本

所以预处理放在 `PS` 是合理的工程选择。

## 为什么 NMS 放在 PS

NMS 的本质是：

- 按分数排序
- 计算 IoU
- 做候选框抑制
- 输出动态长度结果

这类逻辑的特点不是“乘加密集”，而是“控制流重、分支多、输出动态”。

它不太适合优先做成纯 Verilog，原因包括：

- 排序和 IoU 比较很容易引入复杂状态机
- 类别相关抑制策略经常要调
- 输出框数不是固定的
- 放进 `PL` 对 timing 和验证都不友好

因此，`NMS` 保留在 `PS`，而把最重的卷积前向放进 `PL`，是更稳妥的划分。

## 为什么特征提取和检测头前向放在 PL

这一部分具备 FPGA 最擅长的特征：

- 卷积乘加密集
- 通道累加规则
- 数据流固定
- 易于做定点化和流水化

本方案中，前向主链由以下 RTL 组成：

- `hardware/vivado_src/rtl_fullhw/TinyFullHwPlOnlyDemoDetectorTop.v`
- `hardware/vivado_src/rtl_fullhw/TinyFullHwMultiChannelFeatureStage.v`
- `hardware/vivado_src/rtl_fullhw/TinyFullHwMultiChannelHead.v`
- `hardware/vivado_src/rtl_fullhw/DetectionHeadAxisPacketizer.v`

共享卷积算子和量化/流水路径在：

- `hardware/vivado_src/rtl_pl_ps/PlPsConvOperatorTop.v`
- `hardware/vivado_src/rtl_pl_ps/Conv3x3TileArray.v`
- `hardware/vivado_src/rtl_pl_ps/Conv3x3OutputPE.v`

## DET1 为什么是合理接口

当前 `PL` 并不把整块中间 feature map 原样回传，而是直接打成结构化 `DET1` 包输出。

`DET1` 头里包含：

- 魔数 `DET1`
- `grid_w`
- `grid_h`
- `anchors`
- `box_params`
- `num_classes`
- `record_bytes`

这样做的好处是：

- `PL/PS` 边界清晰
- `PS` 解码简单
- 后续替换前向网络时，上层软件接口可以保持稳定

## AXI/DMA 吞吐分析

### 接口配置

当前 block design 中：

- AXI DMA stream width = `8 bit`
- `clk_fpga_0 = 50 MHz`

理论单向 AXI-Stream 吞吐上限：

```text
50 MHz × 1 Byte = 50 MB/s
```

### 实测参考

吞吐参考样例在：

- `demo_output/throughput_reference/camera_latest_metadata.json`

其中一帧的关键数据是：

- 输入：`2048 B`
- 输出：`7216 B`
- 总传输：`9264 B`
- 硬件推理：`21.51 ms`

对应有效吞吐约为：

```text
9264 / 0.02151 ≈ 0.41 MB/s
```

这只占理论单向带宽的大约：

```text
0.41 / 50 ≈ 0.82%
```

结论很明确：

- 当前瓶颈不在 DMA/AXI 带宽
- 没有必要为了减少一点回传字节，就把 `NMS` 也强行塞进 `PL`

## 为什么这种划分有利于 timing

本次重新综合后的 `cam32` overlay timing 结果是：

- `WNS = 3.313 ns`
- `TNS = 0.000 ns`

说明当前 50 MHz 下已经 timing closure。

同时，保留 `NMS` 在 `PS` 还有一个很重要的好处：

- 避免把排序、IoU、动态候选框输出等控制密集逻辑放进 `PL`
- 减少额外的组合路径和状态机复杂度
- 给卷积前向保留更好的时序空间

## 这套方案的准确定位

这不是“纯 FPGA 全部完成真实目标检测”的方案。

它更准确的定位是：

- `PL` 承担规则前向计算
- `PS` 承担灵活的后处理与系统组织
- `Host` 承担部署和远程查看

这是在资源受限 FPGA 平台上非常实用、也非常适合继续演进到“真实 YOLOv8 分层硬件化”的中间架构。

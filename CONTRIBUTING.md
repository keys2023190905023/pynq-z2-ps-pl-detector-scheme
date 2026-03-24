# Contributing

感谢你愿意改进这个项目。

## 适合提交的内容

- 修复 `PL/PS` 接口或板端脚本问题
- 改进 Vivado 重建脚本
- 补充更完整的板测、相机样例和 timing 结果
- 将当前 `demo detector head` 进一步替换为真实模型前向中的一部分

## 建议流程

1. 先阅读 `README.md`、`ARCHITECTURE_PS_PL.md` 和 `RUN_ON_NEW_PC.md`
2. 修改前尽量先复现一轮当前结果
3. 对 RTL 修改至少补一项：
   - 仿真结果
   - timing 报告
   - 板测 JSON
4. 对 Python 修改至少补一项：
   - 命令行复现步骤
   - 示例输出

## 提交建议

- 提交信息尽量说明修改范围
- 如果改了 overlay，请同时说明：
  - 使用的 Vivado 版本
  - 新的 `WNS/TNS`
  - 新 bit/hwh 的哈希
- 如果改了 `DET1` 包格式，请同步更新文档

## 当前项目边界

请保持文档表述准确：

- 这条仓库主线是 `PL 前向 + PS 后处理`
- 当前公开版本是 `demo detector head`
- 不要把它描述成“训练好的真实 YOLOv8 全硬件检测器”，除非你已经补齐真实模型映射和板端验证

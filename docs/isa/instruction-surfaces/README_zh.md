# 指令集总览

本章描述 PTO ISA 的指令集（Instruction Set）——按功能角色组织的指令分类。不同指令集对应不同的执行路径与操作域。

## 本章内容

- [指令集总览](README_zh.md) — 四类指令集的整体说明、数据流图和操作数对照表
- [Tile 指令集](tile-instructions_zh.md) — `pto.t*` 逐 tile 操作指令集
- [向量指令集](vector-instructions_zh.md) — `pto.v*` 向量微指令集
- [标量与控制指令集](scalar-and-control-instructions_zh.md) — 标量、控制和配置操作指令集
- [其他指令集](other-instructions_zh.md) — 通信、调试和其他支持操作

## 阅读建议

建议按以下顺序阅读：

1. 先读 [指令集总览](README_zh.md)，理解 Tile / Vector / Scalar&Control / Other 四类指令集的整体结构和数据流关系
2. 再根据需要深入具体指令集页面，了解该指令集的操作数类型、约束和规范语言

## 章节定位

本章属于手册第 7 章（指令集）的一部分。指令集是介于编程模型与具体指令之间的中层抽象，用于按功能角色定位指令。

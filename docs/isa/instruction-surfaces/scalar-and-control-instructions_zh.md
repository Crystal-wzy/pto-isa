# 标量与控制指令集

`pto.*` 中的标量与控制指令集负责配置、同步、DMA 外壳、谓词和控制流支撑。它们围绕 tile 与 vector 有效载荷工作，而不是直接产生 tile / vector 结果。

## 指令集概览

| 类别 | 说明 |
| --- | --- |
| 控制与配置 | 屏障、yield、控制外壳和基础配置 |
| 流水线同步 | 事件、barrier 与 producer-consumer 顺序 |
| DMA 拷贝 | GM 与 UB 之间的数据搬运配置和发起 |
| 谓词加载存储 | 谓词相关的内存访问 |
| 谓词生成与代数 | 谓词构造、比较和布尔组合 |
| 共享算术 / 共享 SCF | 支撑 PTO 区域的标量算术与结构化控制流 |

## 输入

标量与控制操作主要处理：

- 标量寄存器
- pipe 标识和 event 标识
- DMA loop size / stride
- UB / GM 指针
- 谓词与控制参数

## 输出

这类操作会产生：

- 事件或同步状态
- DMA 配置状态
- 谓词 mask
- 标量结果或控制状态变化

## 约束

- pipe / event 空间必须符合所选 profile。
- DMA 参数必须自洽。
- 谓词宽度和目标操作必须匹配。
- 同步建立与消费必须成对出现。

## 不允许的情形

- 等待从未建立的事件
- 使用目标 profile 不支持的 pipe 或 event 标识
- 在没有完成同步的情况下越过 DMA / vector / tile 的顺序边

## 相关页面

- [标量与控制指令族](../instruction-families/scalar-and-control-families_zh.md)
- [标量参考入口](../scalar/README_zh.md)

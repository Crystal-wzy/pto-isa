# Tile 指令集

`pto.t*` 指令集覆盖以 tile 为中心的计算、搬运、重排、归约和同步。tile 的 shape、layout、role 与 valid region 都属于架构可见状态。

## 指令集概览

| 类别 | 说明 |
| --- | --- |
| 同步与配置 | 资源绑定、同步边和模式控制 |
| 逐元素 Tile-Tile | 两个或多个 tile 之间的逐元素操作 |
| Tile-标量与立即数 | tile 与标量/立即数混合操作 |
| 归约与扩展 | 按行列做归约和广播式扩展 |
| 内存与数据搬运 | GM 与 tile 之间的传输，含 gather/scatter |
| 矩阵与矩阵-向量 | GEMV、matmul 及其变体 |
| 布局与重排 | reshape、transpose、extract、insert、img2col |
| 不规则与复杂 | sort、quant、print、partial 等 |

## 输入

Tile 指令集常见输入包括：

- 源 tile
- 目标 tile / tile buffer
- 标量修饰符和立即数
- GM 视图
- 可选事件链

## 输出

Tile 指令会产生：

- 目标 tile 数据
- valid region 或布局解释变化
- 同步边或资源状态更新

## 约束

- dtype、shape、layout、role 与 valid region 都可能参与合法性判断。
- 多输入 tile 操作必须明确说明 valid region 的组合关系。
- 某些 tile 形式仅对特定 profile 可用，例如部分 MX 或 FP8 路径。

## 不允许的情形

- 把域外数据当作稳定语义使用
- 假设 tile 指令自动继承 vector 寄存器语义
- 依赖未文档化的隐式广播、隐式 reshape 或隐式 valid-region 修复

## 相关页面

- [Tile 指令族](../instruction-families/tile-families_zh.md)
- [Tile 参考入口](../tile/README_zh.md)

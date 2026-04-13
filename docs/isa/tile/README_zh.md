# Tile ISA 参考

`pto.t*` 是 PTO ISA 中以 tile 为中心的主干指令集。该指令集覆盖 tile 载荷的装载、逐元素计算、归约、扩展、重排、矩阵运算和显式同步。

## 组织方式

Tile 参考按指令族组织，具体 per-op 页面位于 `tile/ops/` 下。

## 指令族

- 同步与配置
- 逐元素 Tile-Tile
- Tile-标量与立即数
- 归约与扩展
- 内存与数据搬运
- 矩阵与矩阵-向量
- 布局与重排
- 不规则与复杂

## 共享约束

- tile 的 dtype、shape、layout、role 和 valid region 都可能进入合法性判断
- 目标 tile 的 valid region 通常决定逐元素迭代域
- 某些高性能或特殊格式路径仅在特定 profile 上可用

## 相关页面

- [Tile 指令集](../instruction-surfaces/tile-instructions_zh.md)
- [Tile 指令族](../instruction-families/tile-families_zh.md)
- [指令描述格式](../reference/format-of-instruction-descriptions_zh.md)

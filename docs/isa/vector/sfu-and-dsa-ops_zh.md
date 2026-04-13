# SFU 与 DSA 指令集

这组指令覆盖特殊函数、融合形式和某些领域专用的向量操作。它们通常比普通算术更窄，也更容易受 profile 缩窄影响。

## 常见操作

- `pto.vprelu`
- `pto.vexpdiff`
- `pto.vaxpy`
- `pto.vaddrelu` / `pto.vsubrelu`
- `pto.vaddreluconv` / `pto.vmulconv`
- `pto.vmull` / `pto.vmula`
- `pto.vtranspose`
- `pto.vsort32`
- `pto.vmrgsort`
- `pto.vbitsort`

## 机制

这些操作通常属于以下几类：

- 特殊函数变体
- 融合算术 + 激活
- 领域专用重排或排序
- 向量层的扩宽乘加

## 约束

- 某些操作只支持特定元素类型
- 某些融合形式或排序形式只在 A5 profile 上稳定支持
- lowering 必须保留这些操作的专用语义，不能随意拆成“差不多等价”的普通算术

## 不允许的情形

- 把 profile 专属 SFU/DSA 形式写成所有目标的通用路径
- 省略融合形式带来的额外语义约束

## 相关页面

- [向量指令族](./vector-families_zh.md)
- [向量指令集](../instruction-surfaces/vector-instructions_zh.md)

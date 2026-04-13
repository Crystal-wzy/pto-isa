# 一元向量指令集

一元向量指令对单个向量输入做 lane 级变换，常见于绝对值、取负、指数、对数、平方根和倒数等操作。

## 常见操作

- `pto.vabs`
- `pto.vneg`
- `pto.vexp`
- `pto.vln`
- `pto.vsqrt`
- `pto.vrsqrt`
- `pto.vrec`
- `pto.vrelu`
- `pto.vnot`
- `pto.vbcnt`
- `pto.vcls`
- `pto.vmov`

## 操作数模型

- `%input`：源向量寄存器
- `%mask`：控制激活 lane
- `%result`：目标向量寄存器

## 机制

对每个激活 lane：

$$ \mathrm{dst}_i = f(\mathrm{input}_i) $$

不同指令在浮点异常、整数边界值和 masked-lane 处理上会有各自细节。

## 约束

- 源和目标类型通常保持一致，除非具体指令另有说明
- 指数、对数、平方根等只对浮点类型合法
- profile 会缩窄某些低精度形式

## 不允许的情形

- 在未文档化类型上使用一元数学函数
- 依赖 masked-lane 的未说明结果

## 相关页面

- [向量指令族](./vector-families_zh.md)
- [向量指令集](../instruction-surfaces/vector-instructions_zh.md)

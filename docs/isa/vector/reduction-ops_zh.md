# 归约指令集

向量归约指令在 lane 之间做归约。与 tile 归约不同，它们直接作用于向量寄存器结构，并受 VLane 架构影响。

## 常见操作

- `pto.vcadd`
- `pto.vcmax`
- `pto.vcmin`
- `pto.vcgadd`
- `pto.vcgmax`
- `pto.vcgmin`
- `pto.vcpadd`

## 机制

### 普通归约

在单个向量寄存器内部沿 lane 做归约。

### Group Reduction

在 A5 上，`vcg*` 类操作按 VLane 分组归约，每个 VLane 产生一个结果，而不是整个寄存器只产出单值。

## 约束

- 归约算子的元素类型必须与操作匹配
- group reduction 语义必须保留 VLane 粒度
- 后续消费者必须理解结果是在每个 VLane 上产生的

## 不允许的情形

- 把 group reduction 写成“整寄存器单值归约”
- 混淆普通归约和按 VLane 分组归约

## 相关页面

- [执行代理与目标 Profile](../machine-model/execution-agents_zh.md)
- [向量指令族](./vector-families_zh.md)

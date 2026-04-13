# 向量-标量指令集

向量-标量指令把一个向量寄存器与一个标量操作数结合。标量会逻辑广播到每个参与的 lane。

## 常见操作

- `pto.vadds` / `vsubs` / `vmuls`
- `pto.vmaxs` / `vmins`
- `pto.vands` / `vors` / `vxors`
- `pto.vshls` / `vshrs`
- `pto.vlrelu`
- `pto.vaddcs` / `vsubcs`

## 操作数模型

- `%input`：向量输入
- `%scalar`：标量输入
- `%mask`：激活 lane 谓词
- `%result`：目标向量

## 机制

```text
dst[i] = f(src[i], scalar)
```

标量广播本身是该指令集语义的一部分，而不是 lowering 时临时补出来的行为。

## 约束

- 标量类型必须与目标向量类型兼容
- 对 32-bit scalar 形式，backend 可能对标量来源施加额外限制
- 位移类要求 shift amount 与元素位宽兼容
- carry 变体同时会产生结果与 carry-out 谓词

## 不允许的情形

- 假设存在隐式类型提升
- 使用超出元素位宽的 shift count
- 把 carry-out 谓词丢失在 lowering 中

## 相关页面

- [向量指令族](./vector-families_zh.md)
- [二元向量指令集](./binary-vector-ops_zh.md)

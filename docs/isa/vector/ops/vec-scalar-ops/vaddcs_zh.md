# pto.vaddcs

执行向量-标量带进位加法。

## 语法

```mlir
%result, %carry = pto.vaddcs %lhs, %rhs, %carry_in, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask, !pto.mask -> !pto.vreg<NxT>, !pto.mask
```

## 关键约束

- 标量类型必须与目标向量类型兼容。
- 位移和 carry 变体有额外限制。

## 相关页面

- [向量-标量指令集](../../vec-scalar-ops_zh.md)

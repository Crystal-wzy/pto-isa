# pto.vaddc

执行带扩展或进位语义的向量加法。

## 语法

```mlir
%result, %carry = pto.vaddc %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>, !pto.mask
```

## 关键约束

- 输入向量与结果向量的宽度和元素类型必须兼容。
- 位运算、移位和 carry 变体还会受具体元素类型限制。

## 相关页面

- [二元向量指令集](../../binary-vector-ops_zh.md)

# pto.vmuls

执行向量-标量乘法。

## 语法

```mlir
%result = pto.vmuls %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>
```

## 关键约束

- 标量类型必须与目标向量类型兼容。
- 位移和 carry 变体有额外限制。

## 相关页面

- [向量-标量指令集](../../vec-scalar-ops_zh.md)

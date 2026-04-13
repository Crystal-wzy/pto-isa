# pto.vmax

按激活 lane 取最大值。

## 语法

```mlir
%result = pto.vmax %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
```

A5 已文档化的支持形式：`i8` 到 `i32`，以及 `f16`、`bf16`、`f32`。

## 关键约束

- 输入与结果类型必须一致。


## 相关页面

- [二元向量指令集](../../binary-vector-ops_zh.md)

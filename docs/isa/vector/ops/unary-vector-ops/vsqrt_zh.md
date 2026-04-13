# pto.vsqrt

逐 lane 取平方根。

## 语法

```mlir
%result = pto.vsqrt %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
```

A5 已文档化的支持形式：`f16` 与 `f32`。

## 关键约束

- 仅支持浮点元素类型。


## 相关页面

- [一元向量指令集](../../unary-vector-ops_zh.md)

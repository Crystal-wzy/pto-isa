# pto.vneg

逐 lane 取相反数。

## 语法

```mlir
%result = pto.vneg %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
```

A5 已文档化的支持形式：`i8-i32, f16, f32`。

## 关键约束

- 源与结果类型必须一致。


## 相关页面

- [一元向量指令集](../../unary-vector-ops_zh.md)

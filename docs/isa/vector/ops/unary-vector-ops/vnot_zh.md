# pto.vnot

逐 lane 取按位非。

## 语法

```mlir
%result = pto.vnot %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
```

A5 已文档化的支持形式：全部整数类型。

## 关键约束

- 仅支持整数元素类型。


## 相关页面

- [一元向量指令集](../../unary-vector-ops_zh.md)

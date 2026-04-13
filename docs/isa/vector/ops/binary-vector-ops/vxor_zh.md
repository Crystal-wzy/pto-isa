# pto.vxor

按激活 lane 做按位异或。

## 语法

```mlir
%result = pto.vxor %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
```

A5 已文档化的支持形式：全部整数类型。

## 关键约束

- 仅支持整数元素类型。


## 相关页面

- [二元向量指令集](../../binary-vector-ops_zh.md)

# pto.vors

执行向量-标量按位或。

## 语法

```mlir
%result = pto.vors %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>
```

## 关键约束

- 仅支持整数元素类型。


## 相关页面

- [向量-标量指令集](../../vec-scalar-ops_zh.md)

# pto.vmins

执行向量-标量最小值。

## 语法

```mlir
%result = pto.vmins %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>
```

## 关键约束

- 输入与结果类型必须一致。


## 相关页面

- [向量-标量指令集](../../vec-scalar-ops_zh.md)

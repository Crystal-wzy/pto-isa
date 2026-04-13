# pto.vcvt

在不同数值类型之间做向量转换。

## 语法

```mlir
%result = pto.vcvt %input {round_mode = "ROUND_MODE", sat = "SAT_MODE", part = "PART_MODE"} : !pto.vreg<NxT0> -> !pto.vreg<MxT1>
```

## 关键约束

- 当前 target profile 可能对该形式施加额外限制。


## 相关页面

- [转换操作指令集](../../conversion-ops_zh.md)

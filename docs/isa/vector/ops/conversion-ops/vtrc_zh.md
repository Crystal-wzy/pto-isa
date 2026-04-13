# pto.vtrc

把浮点值舍入为整数值的浮点表示。

## 语法

```mlir
%result = pto.vtrc %input, "ROUND_MODE" : !pto.vreg<NxT> -> !pto.vreg<NxT>
```

## 关键约束

- 当前 target profile 可能对该形式施加额外限制。


## 相关页面

- [转换操作指令集](../../conversion-ops_zh.md)

# pto.vadd

按激活 lane 逐项相加两个向量。

## 语法

### PTO 汇编形式

```text
vadd %dst, %lhs, %rhs, %mask : !pto.vreg<NxT>
```

### AS Level 1（SSA）

```mlir
%result = pto.vadd %lhs, %rhs, %mask : (!pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask) -> !pto.vreg<NxT>
```

### AS Level 2（DPS）

```mlir
pto.vadd ins(%lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask)
          outs(%result : !pto.vreg<NxT>)
```

## 关键约束

- 输入与结果类型必须一致。
- 谓词宽度必须与目标向量宽度一致。


## 相关页面

- [二元向量指令集](../../binary-vector-ops_zh.md)

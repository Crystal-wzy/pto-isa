# pto.vexp

逐 lane 取指数。

## 语法

### PTO 汇编形式

```text
%result = vexp %input, %mask : !pto.vreg<NxT>, !pto.mask
```

### AS Level 1（SSA）

```mlir
%result = pto.vexp %input, %mask : (!pto.vreg<NxT>, !pto.mask) -> !pto.vreg<NxT>
```

### AS Level 2（DPS）

```mlir
pto.vexp ins(%input, %mask : !pto.vreg<NxT>, !pto.mask)
          outs(%result : !pto.vreg<NxT>)
```

## 关键约束

- 输入与结果类型必须一致。
- 谓词宽度必须与目标向量宽度一致。
- 当前 target profile 可能对该形式施加额外限制。


## 相关页面

- [一元向量指令集](../../unary-vector-ops_zh.md)

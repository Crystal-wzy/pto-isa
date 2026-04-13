# pto.vci

根据标量索引或种子生成索引向量。

## 语法

### PTO 汇编形式

```asm
vci %index, %mask {order = "ORDER"} : !pto.vreg<Nxi32> -> !pto.vreg<Nxi32>
```

### AS Level 1（SSA）

```mlir
%indices = pto.vci %index {order = "ASC"} : i32 -> !pto.vreg<64xi32>
```

### AS Level 2（DPS）

```mlir
pto.vci ins(%index : i32) outs(%indices : !pto.vreg<64xi32>) {order = "ASC"}
```

## 关键约束

- 输入与结果类型必须一致。


## 相关页面

- [转换操作指令集](../../conversion-ops_zh.md)

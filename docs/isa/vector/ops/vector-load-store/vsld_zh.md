# pto.vsld

把向量寄存器回写到 UB。

## 语法

```mlir
%result = pto.vsld %source[%offset], "STRIDE" : !pto.ptr<T, ub> -> !pto.vreg<NxT>
```

## 关键约束

- UB 地址空间与对齐要求必须满足。
- DMA 与向量计算之间的顺序边必须显式建立。

## 相关页面

- [向量加载存储指令集](../../vector-load-store_zh.md)

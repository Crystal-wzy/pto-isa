# pto.vstx2

执行双路存储或去交错存储。

## 语法

```mlir
pto.vstx2 %low, %high, %dest[%offset], "DIST", %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.ptr<T, ub>, index, !pto.mask
```

## 关键约束

- UB 地址空间与对齐要求必须满足。
- DMA 与向量计算之间的顺序边必须显式建立。

## 相关页面

- [向量加载存储指令集](../../vector-load-store_zh.md)

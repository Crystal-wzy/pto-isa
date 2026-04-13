# pto.vsts

把向量寄存器结果写回 UB。

## 语法

```mlir
pto.vsts %value, %dest[%offset], %mask {dist = "DIST"} : !pto.vreg<NxT>, !pto.ptr<T, ub>, !pto.mask
```

## 关键约束

- UB 地址空间与对齐要求必须满足。
- DMA 与向量计算之间的顺序边必须显式建立。

## 相关页面

- [向量加载存储指令集](../../vector-load-store_zh.md)

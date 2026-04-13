# pto.vlds

从 UB 加载向量寄存器。

## 语法

```mlir
%result = pto.vlds %source[%offset] {dist = "DIST"} : !pto.ptr<T, ub> -> !pto.vreg<NxT>
```

## 关键约束

- UB 地址空间与对齐要求必须满足。
- DMA 与向量计算之间的顺序边必须显式建立。

## 相关页面

- [向量加载存储指令集](../../vector-load-store_zh.md)

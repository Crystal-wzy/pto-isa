# pto.vstas

初始化流式非对齐存储状态。

## 语法

```mlir
pto.vstas %value, %dest, %offset : !pto.align, !pto.ptr<T, ub>, i32
```

## 关键约束

- UB 地址空间与对齐要求必须满足。
- DMA 与向量计算之间的顺序边必须显式建立。

## 相关页面

- [向量加载存储指令集](../../vector-load-store_zh.md)

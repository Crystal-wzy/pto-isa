# pto.vldas

初始化非对齐加载所需的对齐状态。

## 语法

```mlir
%result = pto.vldas %source : !pto.ptr<T, ub> -> !pto.align
```

## 关键约束

- UB 地址空间与对齐要求必须满足。
- DMA 与向量计算之间的顺序边必须显式建立。

## 相关页面

- [向量加载存储指令集](../../vector-load-store_zh.md)

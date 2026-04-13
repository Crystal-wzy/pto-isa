# pto.vsst

执行流式向量存储。

## 语法

```mlir
pto.vsst %value, %dest[%offset], "STRIDE" : !pto.vreg<NxT>, !pto.ptr<T, ub>
```

## 关键约束

- UB 地址空间与对齐要求必须满足。
- DMA 与向量计算之间的顺序边必须显式建立。

## 相关页面

- [向量加载存储指令集](../../vector-load-store_zh.md)

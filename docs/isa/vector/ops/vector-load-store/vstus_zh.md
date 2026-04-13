# pto.vstus

执行流式非对齐向量存储。

## 语法

```mlir
%align_out, %base_out = pto.vstus %align_in, %offset, %value, %base, "MODE" : !pto.align, i32, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align, !pto.ptr<T, ub>
```

## 关键约束

- UB 地址空间与对齐要求必须满足。
- DMA 与向量计算之间的顺序边必须显式建立。

## 相关页面

- [向量加载存储指令集](../../vector-load-store_zh.md)

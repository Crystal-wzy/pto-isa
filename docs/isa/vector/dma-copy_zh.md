# 向量 DMA 路径

这组文档说明向量路径中与 DMA 相关的可见约束。它们描述的是 GM、UB 与向量寄存器之间的搬运关系，而不是独立的向量算术指令。

## 作用

在 `pto.v*` 路径中：

```text
GM -> copy_gm_to_ubuf -> UB -> vlds -> vreg
vreg -> vsts -> UB -> copy_ubuf_to_gm -> GM
```

因此向量路径的 DMA 不是“可选细节”，而是向量计算可见数据路径的一部分。

## 关键点

- GM 与 UB 之间通过 `copy_gm_to_ubuf` / `copy_ubuf_to_gm`
- 向量寄存器与 UB 之间通过 `vlds` / `vsts`
- DMA 与向量计算之间必须通过 `set_flag` / `wait_flag` 建立顺序

## 不允许的情形

- 把 DMA 完成假设为向量计算的隐式前提
- 把 UB 可见性误写成 GM 可见性

## 相关页面

- [向量加载存储](./vector-load-store_zh.md)
- [流水线同步](./pipeline-sync_zh.md)
- [标量 DMA 拷贝](../scalar/dma-copy_zh.md)

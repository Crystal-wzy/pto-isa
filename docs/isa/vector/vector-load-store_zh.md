# 向量加载存储指令集

向量加载存储指令集定义 UB 与向量寄存器之间的数据搬运形式。`pto.v*` 计算只在向量寄存器上执行，因此这组指令构成 DMA 与向量计算之间的桥。

## 执行模型

```text
GM -> MTE2 -> UB -> VLDS -> vreg -> PTO.v* compute -> VSTS -> UB -> MTE3 -> GM
```

典型顺序链：

- `copy_gm_to_ubuf`
- `set_flag(PIPE_MTE2, PIPE_V, id)`
- `wait_flag(...)`
- `vlds`
- 向量计算
- `vsts`
- `set_flag(PIPE_V, PIPE_MTE3, id)`
- `wait_flag(...)`
- `copy_ubuf_to_gm`

## 常见操作

- `pto.vlds`
- `pto.vldas`
- `pto.vldus`
- `pto.vldx2`
- `pto.vsld`
- `pto.vsldb`
- `pto.vsst`
- `pto.vsstb`
- `pto.vsta`
- `pto.vstar`
- `pto.vstas`
- `pto.vstu`
- `pto.vstus`
- `pto.vstur`
- `pto.vscatter`
- `pto.vgather2`
- `pto.vgather2_bc`
- `pto.vgatherb`

## 操作数模型

- `%source` / `%dest`：UB 地址基址
- `%offset`：位移
- `%mask`：predicated memory op 的谓词
- `%result`：目标向量寄存器
- `!pto.align`：非对齐 load/store 使用的对齐状态

## 分布模式

常见 `dist` 模式包括：

- `NORM`
- `BRC_B8/B16/B32`
- `US_B8/B16`
- `DS_B8/B16`
- `UNPK_B8/B16/B32`
- `DINTLV_B32`
- `SPLT2CHN_B8/B16`
- `SPLT4CHN_B8`

## 约束

- 基址必须是 UB 指针
- 有效地址必须满足对应分布模式的对齐要求
- 非对齐流必须先通过 `vldas` 等操作初始化对齐状态
- predicated memory op 不应在 inactive lane 上发起未文档化的内存请求

## 不允许的情形

- 在没有建立对齐状态时直接使用非对齐流式 load/store
- 在 DMA 尚未完成时直接消费 UB 数据
- 把 UB 地址空间与 GM 地址空间混用

## 相关页面

- [向量流水同步](./pipeline-sync_zh.md)
- [向量指令集](../instruction-surfaces/vector-instructions_zh.md)
- [标量 DMA 拷贝](../scalar/dma-copy_zh.md)

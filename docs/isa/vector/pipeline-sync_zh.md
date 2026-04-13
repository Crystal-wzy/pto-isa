# 向量流水同步

向量路径通过显式事件在 DMA 与 `PIPE_V` 之间建立顺序。该同步机制决定数据何时能从 UB 被加载到向量寄存器，又何时能从向量寄存器写回 UB / GM。

## 典型顺序链

```text
copy_gm_to_ubuf
  -> set_flag(PIPE_MTE2, PIPE_V, id)
  -> wait_flag(PIPE_MTE2, PIPE_V, id)
  -> vlds
  -> vector compute
  -> vsts
  -> set_flag(PIPE_V, PIPE_MTE3, id)
  -> wait_flag(PIPE_V, PIPE_MTE3, id)
  -> copy_ubuf_to_gm
```

## 关键原语

- `set_flag`
- `wait_flag`
- `get_buf`
- `rls_buf`
- `mem_bar`

## 约束

- DMA 完成前，向量加载不能消费对应 UB 数据
- `get_buf / rls_buf` 负责 buffer-token 级别的依赖管理，但不替代事件顺序本身
- `mem_bar` 只处理内存可见性，不单独建立跨流水顺序

## 不允许的情形

- 在 `copy_gm_to_ubuf` 完成前直接 `vlds`
- 在 `vsts` 完成前直接 `copy_ubuf_to_gm`
- 把 buffer-token 协议误写成“无需事件”

## 相关页面

- [向量加载存储](./vector-load-store_zh.md)
- [顺序与同步](../machine-model/ordering-and-synchronization_zh.md)

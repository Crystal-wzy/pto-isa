# 内存与数据搬运指令集

该指令集覆盖 GM 与 tile 之间的数据搬运，以及索引式 `mgather` / `mscatter`。它们决定 tile 数据何时进入本地缓冲，又何时离开本地缓冲回到 GM。

## 操作

| 操作 | 说明 |
| --- | --- |
| `pto.tload` | 从 GlobalTensor 加载到 tile |
| `pto.tprefetch` | 预取数据到局部缓冲路径 |
| `pto.tstore` | 从 tile 写回 GlobalTensor |
| `pto.tstore_fp` | 带浮点特殊处理的 store |
| `pto.mgather` | 索引 gather |
| `pto.mscatter` | 索引 scatter |

## 机制

### TLOAD

```text
dst[i, j] = src[r0 + i, c0 + j]
```

搬运大小由目标 tile 的 `GetValidRow()` / `GetValidCol()` 决定。

### TSTORE

```text
dst[r0 + i, c0 + j] = src[i, j]
```

写回大小由源 tile 的 valid region 决定。

### MGATHER / MSCATTER

索引类搬运允许以非连续位置访问数据：

```text
gather:  dst[i] = src[index[i]]
scatter: dst[index[i]] = src[i]
```

## 数据路径

```text
GM -> UB -> Tile Buffer
Tile Buffer -> UB -> GM
```

Tile 路径与向量路径不同；向量路径需要先显式经过 UB 和 `vlds/vsts`。

## 约束

- 源与目标 dtype 大小必须兼容
- GlobalTensor 布局必须与目标 tile 布局兼容
- gather/scatter 索引必须落在合法范围内
- atomic store 仅在支持的 profile 上提供

## 不允许的情形

- 把 tile 搬运和 vector 搬运写成同一套契约
- 在没有显式同步的情况下跨越 DMA / compute 顺序边
- scatter 到合法范围之外

## 相关页面

- [内存模型](../memory-model/consistency-baseline_zh.md)
- [生产者-消费者排序](../memory-model/producer-consumer-ordering_zh.md)
- [Tile 指令族](../instruction-families/tile-families_zh.md)

# TPREFETCH

## 指令示意图

![TPREFETCH tile operation](../../../../figures/isa/TPREFETCH.svg)

## 简介

`TPREFETCH` 把 `GlobalTensor` 的一段数据提前搬进 tile 本地缓冲，用作后续访问前的预取或预热。它的目的不是做 layout 变换，而是把“稍后会用到的 GM 数据”尽早拉到 tile 可见的本地空间里。

名字里虽然叫 prefetch，但在当前仓库实现里它并不是纯粹的“可完全忽略的提示位”，而是会实际把一段数据写入 `dst`。真正应该把它当成“提示”的部分，是缓存或局部性收益本身，而不是 `dst` 是否被填充。

## 数学语义

若把 `src` 当前视图看成一个二维切片，`TPREFETCH` 的可见结果接近于：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{r_0 + i,\; c_0 + j} $$

其中具体会搬多少行列，取决于 `dst` 的尺寸以及 `src` 当前 `GlobalTensor` 视图的 shape / stride。和 `TLOAD` 相比，这条指令不承诺复杂的 layout 转换；它更像“把接下来要访问的 GM 切片装进一个临时 tile 缓冲”。

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../../../../assembly/PTO-AS_zh.md)。

同步形式：

```text
%dst = tprefetch %src : !pto.global<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tprefetch %src : !pto.global<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tprefetch ins(%src : !pto.global<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileData, typename GlobalData>
PTO_INST RecordEvent TPREFETCH(TileData &dst, GlobalData &src);
```

和大多数 PTO C++ 指令不同，`TPREFETCH` 这个接口没有 `WaitEvents...`，封装层也不会在调用前自动插入 `TSYNC(events...)`。

## 约束

### 通用约束

- 这条指令面向“本地预取缓冲”场景，不应把它当成 `TLOAD` 的语义等价替代品。
- 当前仓库实现没有像 `TLOAD` 那样为它建立一套完整的 layout / dtype 合法性检查路径。
- 可移植代码应只把它用于“提前把后续要访问的数据搬进临时 tile”的场景，而不要依赖某个 backend 的特殊缓存收益。

### CPU 模拟器

- CPU 直接按 `dst.GetValidRow()` / `dst.GetValidCol()` 做逐元素拷贝。

### A2/A3 与 A5 实现

- 两条 NPU 实现都会把 `GlobalTensor` 当前 5D 视图拆成切片后分块搬入 `dst`。
- 如果单个切片能放进 `dst`，则一次性搬运；否则按 `dst` 容量分块预取。
- 这意味着 `TPREFETCH` 的可见结果依赖 `dst` 大小和 `src` 切片大小，而不是单独由“预取提示”决定。

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T, typename GT>
void example(Tile<TileType::Vec, T, 16, 16>& tileBuf, GT& globalView) {
  TPREFETCH(tileBuf, globalView);
}
```

## 相关页面

- [TLOAD](./tload_zh.md)
- [内存与数据搬运指令集](../../memory-and-data-movement_zh.md)
- [GlobalTensor 与数据搬运](../../../programming-model/globaltensor-and-data-movement_zh.md)

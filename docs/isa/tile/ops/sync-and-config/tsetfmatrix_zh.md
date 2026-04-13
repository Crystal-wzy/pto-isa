# TSETFMATRIX

## 指令示意图

![TSETFMATRIX tile operation](../../../../figures/isa/TSETFMATRIX.svg)

## 简介

`TSETFMATRIX` 把 `Img2colTileConfig` 里的输入特征图尺寸和 padding 信息写入 FMATRIX 配置寄存器，供后续 `TIMG2COL` 一类操作读取。

这条指令本身不产生张量算术结果。它的作用是“准备好后面数据重排要依赖的几何参数”。

## 机制

从当前实现看，`TSETFMATRIX` 主要写入三类信息：

- `fmapW`
- `fmapH`
- 四个 padding 字节

在 A2/A3 和 A5 上，这些字段会被打包进一个 64 位配置字：

- 低 16 位：`fmapW`
- 接下来的 16 位：`fmapH`
- 高 32 位：4 个 `padList` 字节

`SetFmatrixMode` 用来决定把这份配置写到 A 侧还是 B 侧寄存器。

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../../../../assembly/PTO-AS_zh.md)。

### AS Level 1（SSA）

```text
pto.tsetfmatrix %cfg : !pto.fmatrix_config -> ()
```

### AS Level 2（DPS）

```text
pto.tsetfmatrix ins(%cfg : !pto.fmatrix_config) outs()
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL, typename... WaitEvents>
PTO_INST RecordEvent TSETFMATRIX(ConvTileData &src, WaitEvents&... events);
```

`SetFmatrixMode` 的可选值包括：

- `FMATRIX_A_AUTO`
- `FMATRIX_B_AUTO`
- `FMATRIX_A_MANUAL`
- `FMATRIX_B_MANUAL`

对这条手动配置指令来说，真正有意义的是两个 `*_MANUAL` 变体。

## 约束

### 通用约束

- `src` 应是有效的 `Img2colTileConfig` 一类配置 Tile。
- 这条指令只负责写配置，不会替你执行 `TIMG2COL`。
- 在同一执行流里，通常应先配置，再发出依赖该配置的 IMG2COL 类指令。

### CPU 模拟器

- CPU 只做轻量检查：`src.GetFmapH() > 0` 且 `src.GetFmapW() > 0`。

### A2/A3 与 A5 实现

- 当 `FmatrixMode` 为 `FMATRIX_A_MANUAL` 或 `FMATRIX_B_MANUAL` 时，backend 才会真正写寄存器。
- `FMATRIX_A_MANUAL` 写 A 侧 FMATRIX。
- `FMATRIX_B_MANUAL` 写 B 侧 FMATRIX。
- 若传的是 `*_AUTO` 变体，这条手动 setter 路径当前不会主动写寄存器；自动模式通常由消费方指令自己完成配置。

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example(Img2colTileConfig<uint16_t>& cfg) {
  TSETFMATRIX<Img2colTileConfig<uint16_t>, SetFmatrixMode::FMATRIX_A_MANUAL>(cfg);
}
```

## 相关页面

- [TSET_IMG2COL_RPT](./tset-img2col-rpt_zh.md)
- [TSET_IMG2COL_PADDING](./tset-img2col-padding_zh.md)
- [TIMG2COL](../layout-and-rearrangement/timg2col_zh.md)

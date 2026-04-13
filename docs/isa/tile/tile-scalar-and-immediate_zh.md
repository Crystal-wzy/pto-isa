# Tile-标量与立即数指令集

Tile-标量类操作把一个 tile 与一个标量或立即数结合。标量会在语义上广播到整个 tile 形状。比较类变体产生谓词 tile。

## 操作

| 操作 | 说明 |
| --- | --- |
| `pto.tadds` / `tsubs` / `tmuls` / `tdivs` | 与标量做逐元素算术 |
| `pto.tfmods` / `trems` | 与标量做模 / 余数 |
| `pto.tmins` / `tmaxs` | 与标量做 min / max |
| `pto.tands` / `tors` / `txors` | 与标量做按位逻辑 |
| `pto.tshls` / `tshrs` | 用标量做位移 |
| `pto.tlrelu` | 标量 slope 的 Leaky ReLU |
| `pto.taddsc` / `tsubsc` | 饱和加减 |
| `pto.texpands` / `tcmps` | 与标量比较，产生谓词结果 |
| `pto.tsels` | 依谓词条件选择 |

## 机制

对目标 valid region 中的每个 lane：

$$ \mathrm{dst}_{r,c} = f(\mathrm{src}_{r,c}, \mathrm{scalar}) $$

标量逻辑上广播到所有 lane。比较类变体会产生谓词 tile。

## 标量操作数

标量可以是：

- 标量寄存器值
- 编译期立即数
- 运行时参数

标量类型必须与 tile 元素类型兼容，不存在隐式类型提升。

## 饱和变体

`TADDSC` 与 `TSUBSC` 在溢出/下溢时截断到类型上下界，而不是使用 wrapping 语义。

## 目标 Profile 支持

| 元素类型 | CPU | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| f32 / f16 / bf16 | Yes | Yes | Yes |
| i8/u8, i16/u16, i32/u32, i64/u64 | Yes | Yes | Yes |

## 约束

- 标量类型必须与 tile 元素类型兼容
- `TSHLS` / `TSHRS` 把标量解释为无符号 shift count
- 比较类产生谓词 tile，而不是数值 tile

## 不允许的情形

- 使用与 tile 元素类型不兼容的标量
- 使用超出元素位宽的 shift count
- 依赖隐式类型提升

## 相关页面

- [Tile 指令族](../instruction-families/tile-families_zh.md)
- [Tile 指令集](../instruction-surfaces/tile-instructions_zh.md)

# 逐元素 Tile-Tile 指令集

逐元素 Tile-Tile 操作按 lane 对两个或一个 tile 做逐元素运算。它们是 PTO 中最常用的 tile 计算类指令。

## 操作

| 操作 | 说明 | 类别 |
| --- | --- | --- |
| `pto.tadd` | 逐元素加法 | Binary |
| `pto.tabs` | 逐元素绝对值 | Unary |
| `pto.tand` | 逐元素按位与 | Binary |
| `pto.tor` | 逐元素按位或 | Binary |
| `pto.tsub` | 逐元素减法 | Binary |
| `pto.tmul` | 逐元素乘法 | Binary |
| `pto.tmin` | 逐元素最小值 | Binary |
| `pto.tmax` | 逐元素最大值 | Binary |
| `pto.tcmp` | 逐元素比较 | Binary |
| `pto.tdiv` | 逐元素除法 | Binary |
| `pto.tshl` / `pto.tshr` | 逐元素移位 | Binary |
| `pto.txor` | 逐元素按位异或 | Binary |
| `pto.tlog` / `pto.trecip` / `pto.texp` / `pto.tsqrt` / `pto.trsqrt` | 一元数学运算 | Unary |
| `pto.tprelu` / `pto.trelu` / `pto.tneg` / `pto.tnot` | 激活或一元变体 | Unary/Binary |
| `pto.taddc` / `pto.tsubc` | 饱和加减 | Binary |
| `pto.tcvt` | 逐元素类型转换 | Unary |
| `pto.tsel` | 条件选择 | Ternary |
| `pto.trem` / `pto.tfmod` | 余数 / 浮点模 | Binary |

## 机制

逐元素操作在目标 tile 的 valid region 上迭代：

$$ \mathrm{dst}_{r,c} = f(\mathrm{src0}_{r,c}, \mathrm{src1}_{r,c}) $$

对 `TSEL`：

$$ \mathrm{dst}_{r,c} = (\mathrm{cmp}_{r,c} \neq 0) ? \mathrm{src0}_{r,c} : \mathrm{src1}_{r,c} $$

## Valid Region 兼容性

所有逐元素 Tile-Tile 操作都以 **目标 tile 的 valid region** 为迭代域。

- 会按相同 `(r, c)` 坐标读取源 tile
- 若源 tile 在该坐标超出自身 valid region，则读取值属于 implementation-defined
- 在没有单独文档化的情况下，不应依赖这些域外值

## 饱和变体

带 `_c` 后缀的变体使用饱和算术，而不是 wrap 算术：

| 基础操作 | 饱和变体 | 溢出行为 |
| --- | --- | --- |
| `TADD` | `TADDC` | 截断到类型上下界 |
| `TSUB` | `TSUBC` | 截断到类型上下界 |

## 目标 Profile 支持

| 元素类型 | CPU | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| f32 / f16 / bf16 | Yes | Yes | Yes |
| i8/u8, i16/u16, i32/u32, i64/u64 | Yes | Yes | Yes |
| f8e4m3 / f8e5m2 | No | No | Yes |

## 约束

- layout、shape 和 valid-region 状态都会影响合法性
- 源与目标 tile 的物理 shape 必须兼容
- `TCMP` 产生谓词 tile，算术类产生数值 tile
- `TCVT` 允许源和目标 dtype 不同，但必须落在文档化转换组内
- 移位类要求第二操作数在元素位宽范围内

## 不允许的情形

- 假设存在隐式广播、隐式 reshape 或 valid-region 修复
- 依赖源 tile 域外 lane 的确定值
- 假设 `TADDC` / `TSUBC` 与非饱和版本对所有输入都位级一致
- 使用超出元素位宽的 shift count

## 相关页面

- [Tile 指令族](../instruction-families/tile-families_zh.md)
- [Tile 指令集](../instruction-surfaces/tile-instructions_zh.md)

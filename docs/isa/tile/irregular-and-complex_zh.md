# 不规则与复杂指令集

不规则操作覆盖无法归入标准逐元素、归约或内存模型的 tile 计算，包括调试、排序、量化、索引型搬运、三角矩阵操作和 partial reduction。

## 操作

| 操作 | 说明 | 类别 | Profile |
| --- | --- | --- | :---: |
| `pto.tprint` | 打印 tile 数据 | Debug | All |
| `pto.tmrgsort` / `pto.tsort32` | 排序 | Sort | All |
| `pto.tgather` / `pto.tgatherb` / `pto.tscatter` | 基于索引的 gather/scatter | Gather/Scatter | All |
| `pto.tci` | 复杂索引操作 | Index | All |
| `pto.ttri` | 三角矩阵抽取或相关操作 | Matrix | All |
| `pto.tpartadd` / `tpartmul` / `tpartmax` / `tpartmin` | partial reduction | Reduce | All |
| `pto.tquant` / `pto.tdequant` | 量化 / 反量化 | Quantize | A2/A3, A5 |
| `pto.tpack` / `pto.trandom` / `pto.thistogram` | 打包、随机数、直方图 | A5-only | A5 |

## 机制

### 排序

`TSORT32` 对 32-bit 值排序，`TMRGSORT` 对 tile 行做 merge sort。

### Gather / Scatter

这些操作按索引 tile 在非连续位置读写数据：

$$ \mathrm{dst}_i = \mathrm{src}_{\mathrm{index}_i} \quad \text{(gather)} $$

$$ \mathrm{dst}_{\mathrm{index}_i} = \mathrm{src}_i \quad \text{(scatter)} $$

### Partial Reduction

与完整按行/按列归约不同，partial reduction 会产生沿某个轴缩减但仍然不是单列或单行的中间 tile。

### 量化

`TQUANT` / `TDEQUANT` 在浮点与量化表示之间转换，需要 scale 和 zero-point 等参数。

## 目标 Profile 支持

| 元素类型 | CPU | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| f32 / f16 / bf16 | Yes | Yes | Yes |
| i8/u8, i16/u16, i32/u32, i64/u64 | Yes | Yes | Yes |
| INT4 / FP4 / NF4 等量化格式 | No | Yes | Yes |

## 约束

- 排序要求元素类型与具体排序变体兼容
- 量化要求 scale 非零，zero-point 在合法范围内
- Scatter 要求索引非负且落在目标范围内
- partial reduction 的具体行为可能随 profile 缩窄
- A5-only 操作不能在 CPU 或 A2/A3 上使用

## 不允许的情形

- 用非法 scale 或越界 zero-point 做量化
- scatter 到目标 tile 形状之外
- 在 CPU / A2 / A3 上使用 `TPACK`、`TRANDOM`、`THISTOGRAM`

## 相关页面

- [Tile 指令族](../instruction-families/tile-families_zh.md)
- [Tile 指令集](../instruction-surfaces/tile-instructions_zh.md)

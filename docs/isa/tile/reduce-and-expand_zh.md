# 归约与扩展指令集

归约操作沿一个轴把二维 tile 折叠成一维结果；扩展操作则把一维 tile 沿一个轴广播成二维 tile。

## 操作

### 按行归约

- `pto.trowsum`
- `pto.trowprod`
- `pto.trowmax`
- `pto.trowmin`
- `pto.trowargmax`
- `pto.trowargmin`

### 按列归约

- `pto.tcolsum`
- `pto.tcolprod`
- `pto.tcolmax`
- `pto.tcolmin`
- `pto.tcolargmax`
- `pto.tcolargmin`

### 按行扩展

- `pto.trowexpand`
- `pto.trowexpandadd`
- `pto.trowexpandsub`
- `pto.trowexpandmul`
- `pto.trowexpanddiv`
- `pto.trowexpandmax`
- `pto.trowexpandmin`
- `pto.trowexpandexpdif`

### 按列扩展

- `pto.tcolexpand`
- `pto.tcolexpandadd`
- `pto.tcolexpandsub`
- `pto.tcolexpandmul`
- `pto.tcolexpanddiv`
- `pto.tcolexpandmax`
- `pto.tcolexpandmin`
- `pto.tcolexpandexpdif`

## 机制

### 归约

按行归约：

$$ \mathrm{dst}_r = \bigoplus_{c=0}^{C-1} \mathrm{src}_{r,c} $$

按列归约：

$$ \mathrm{dst}_c = \bigoplus_{r=0}^{R-1} \mathrm{src}_{r,c} $$

### 扩展

按行扩展：

$$ \mathrm{dst}_{r,c} = \mathrm{src}_r $$

按列扩展：

$$ \mathrm{dst}_{r,c} = \mathrm{src}_c $$

扩展变体会把广播值与第二输入 tile 做逐元素组合。

## 输出形状

| 操作 | 输入形状 | 输出形状 |
| --- | --- | --- |
| 行归约 | `(R, C)` | `(R, 1)` |
| 列归约 | `(R, C)` | `(1, C)` |
| 行扩展 | `(R, 1)` | `(R, C)` |
| 列扩展 | `(1, C)` | `(R, C)` |

## 约束

- 源 tile 的 valid region 决定归约域
- `argmax/argmin` 变体输出索引 tile，而不是数值 tile
- 归约目标 tile 在被归约轴上的 extent 必须为 `1`
- 扩展变体要求第二输入在被扩展轴长度上匹配
- `expdif` 变体依赖指数差的特殊语义

## 不允许的情形

- 在长度为 0 的轴上归约
- 对不支持的元素类型使用 arg 变体
- 用不匹配的扩展轴长度做 expand 变体

## 相关页面

- [Tile 指令族](../instruction-families/tile-families_zh.md)
- [Tile 指令集](../instruction-surfaces/tile-instructions_zh.md)

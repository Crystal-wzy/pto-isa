# 布局与重排指令集

布局与重排操作改变 tile 数据在缓冲区中的组织方式。这些操作是**纯数据移动/重解释**操作，不改变元素值本身。

## 操作

| 操作 | 说明 | 类别 |
| --- | --- | --- |
| `pto.tmov` / `pto.tmov_fp` | 搬运或填充式搬运 | Copy |
| `pto.treshape` | 改变 tile 形状解释 | Transform |
| `pto.ttrans` | 转置 | Transform |
| `pto.textract` / `pto.textract_fp` | 抽取子 tile | Extract |
| `pto.tinsert` / `pto.tinsert_fp` | 插入子 tile | Insert |
| `pto.tfillpad` / `tfillpad_inplace` / `tfillpad_expand` | 填充 padding | Fill |
| `pto.timg2col` | 图像到列的重排 | Transform |

## 机制

### Copy

```text
dst[i, j] = src[i, j]
```

`*_fp` 变体会额外处理 padding 或 fill value。

### Transform

```text
dst[i, j] = src[index(i, j)]
```

- `TTRANS`：交换行列
- `TRESHAPE`：在保持元素总数不变的前提下改变形状解释
- `TIMG2COL`：把卷积 patch 重排成列

### Extract / Insert

```text
TEXTRACT: dst = src[row_offset : ..., col_offset : ...]
TINSERT:  dst[row_offset : ..., col_offset : ...] = src
```

### Fill

对 valid region 之外的 padding 区域填入指定值。`INPLACE` 在原 tile 上操作，`EXPAND` 还会扩大 valid region。

## 目标 Profile 支持

| 元素类型 | CPU | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| f32 / f16 / bf16 | Yes | Yes | Yes |
| i8/u8, i16/u16, i32/u32, i64/u64 | Yes | Yes | Yes |
| f8e4m3 / f8e5m2 | No | No | Yes |

## 约束

- `TRESHAPE` 要求总元素数不变
- `TEXTRACT` 的 offset 与子形状必须落在源 tile 声明范围内
- `TINSERT` 插入后的区域必须落在目标 tile 范围内
- `*_fp` 变体要求 fill value 与 tile 元素类型兼容
- `TIMG2COL` 依赖 kernel/padding/stride 配置，且受 profile 缩窄

## 不允许的情形

- reshape 到不同总元素数的形状
- 用越界 offset 做 extract / insert
- 在不支持的 profile 上把 FP8 与 `TIMG2COL` 组合使用

## 相关页面

- [Tile 指令族](../instruction-families/tile-families_zh.md)
- [Tile 指令集](../instruction-surfaces/tile-instructions_zh.md)
- [布局参考](../state-and-types/layout_zh.md)

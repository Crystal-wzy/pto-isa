# 同步与配置指令集

同步与配置操作管理 tile 可见状态：资源绑定、事件等待、模式配置以及逻辑子视图。它们本身不产生算术有效载荷，但会改变后续 tile 指令所消费的状态。

## 操作

| 操作 | 说明 |
| --- | --- |
| `pto.tassign` | 把 tile 绑定到 UB 地址 |
| `pto.tsync` | 等待事件或插入 tile 级屏障 |
| `pto.tsethf32mode` | 设置 HF32 模式 |
| `pto.tsettf32mode` | 设置 TF32 模式 |
| `pto.tsetfmatrix` | 设置 FMatrix 配置 |
| `pto.tset_img2col_rpt` | 设置 img2col repetition |
| `pto.tset_img2col_padding` | 设置 img2col padding |
| `pto.tsubview` | 创建 tile 子视图 |
| `pto.tget_scale_addr` | 获取量化 matmul 的 scale 地址 |

## 机制

- **`TASSIGN`**：把物理 UB 地址绑定到 tile
- **`TSYNC`**：等待事件或为特定操作类建立 barrier
- **`TSET*`**：设置后续操作会读取的模式寄存器
- **`TSUBVIEW`**：在共享底层存储的前提下创建逻辑子视图
- **`TGET_SCALE_ADDR`**：查询量化路径中使用的 scale 地址

## 同步模型

`TSYNC` 有两种常见形式：

1. **事件等待形式**
   `TSYNC(%e0, %e1)`，等待事件完成
2. **屏障形式**
   `TSYNC<Op>()`，为某类操作建立 pipeline barrier

## 约束

- `TASSIGN` 绑定的是地址；没有别名关系的两个 tile 不能同时安全复用同一地址
- 空参数 `TSYNC()` 是 no-op
- `TSET*` 的配置会影响后续同类依赖操作，直到下一次同类设置覆盖
- `TSUBVIEW` 共享底层存储，超出子视图范围的访问无定义

## 不允许的情形

- 在没有别名语义的情况下让两个 tile 共享同一物理 tile register 地址
- 等待从未由先前操作产生的事件
- 在依赖操作仍在 flight 时修改相关模式寄存器

## 相关页面

- [Tile 指令族](../instruction-families/tile-families_zh.md)
- [顺序与同步](../machine-model/ordering-and-synchronization_zh.md)

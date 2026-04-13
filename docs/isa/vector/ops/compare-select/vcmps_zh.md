# pto.vcmps

比较向量与标量并生成谓词结果。

## 语法

```mlir
%result = pto.vcmps %src, %scalar, %seed, "CMP_MODE" : !pto.vreg<NxT>, T, !pto.mask -> !pto.mask
```

## 关键约束

- 比较模式、谓词宽度和输入类型必须匹配。
- `vselr` / `vselrv2` 的具体选择语义必须由 lowering 精确保留。

## 相关页面

- [比较与选择指令集](../../compare-select_zh.md)

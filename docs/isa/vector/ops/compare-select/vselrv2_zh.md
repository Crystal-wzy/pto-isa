# pto.vselrv2

按 v2 选择变体在两个向量结果之间做选择。

## 语法

```mlir
%result = pto.vselrv2 %src0, %src1 : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>
```

## 关键约束

- 比较模式、谓词宽度和输入类型必须匹配。
- `vselr` / `vselrv2` 的具体选择语义必须由 lowering 精确保留。

## 相关页面

- [比较与选择指令集](../../compare-select_zh.md)

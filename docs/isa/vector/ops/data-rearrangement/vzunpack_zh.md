# pto.vzunpack

按零扩展规则解包向量。

## 语法

```mlir
%result = pto.vzunpack %src, %part : !pto.vreg<NxT_narrow>, index -> !pto.vreg<N/2xT_wide>
```

## 关键约束

- 成对输出的顺序必须保留。
- 不同重排形式只在文档化的元素宽度和模式下合法。

## 相关页面

- [数据重排指令集](../../data-rearrangement_zh.md)

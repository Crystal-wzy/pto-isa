# pto.vintlv

对两个向量做交错组合。

## 语法

```mlir
%low, %high = pto.vintlv %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>, !pto.vreg<NxT>
```

## 关键约束

- 成对输出的顺序必须保留。
- 不同重排形式只在文档化的元素宽度和模式下合法。

## 相关页面

- [数据重排指令集](../../data-rearrangement_zh.md)

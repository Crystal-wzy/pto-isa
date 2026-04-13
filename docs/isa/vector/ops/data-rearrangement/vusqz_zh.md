# pto.vusqz

按无符号规则压缩有效 lane。

## 语法

```mlir
%result = pto.vusqz %mask : !pto.mask -> !pto.vreg<NxT>
```

## 关键约束

- 成对输出的顺序必须保留。
- 不同重排形式只在文档化的元素宽度和模式下合法。

## 相关页面

- [数据重排指令集](../../data-rearrangement_zh.md)

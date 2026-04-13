# pto.vbr

把单值广播为完整向量。

## 语法

```mlir
%result = pto.vbr %value : T -> !pto.vreg<NxT>
```

## 关键约束

- 输出向量宽度必须与目标元素类型匹配。
- 不能把这类操作误写成普通内存加载。

## 相关页面

- [谓词与物化指令集](../../predicate-and-materialization_zh.md)

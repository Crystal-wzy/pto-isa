# pto.vdup

复制值或模式到完整向量。

## 语法

```mlir
%result = pto.vdup %input {position = "POSITION"} : T|!pto.vreg<NxT> -> !pto.vreg<NxT>
```

## 关键约束

- 输出向量宽度必须与目标元素类型匹配。
- 不能把这类操作误写成普通内存加载。

## 相关页面

- [谓词与物化指令集](../../predicate-and-materialization_zh.md)

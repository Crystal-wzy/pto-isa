# pto.vldx2

执行双路加载或去交错加载。

## 语法

```mlir
%low, %high = pto.vldx2 %source[%offset], "DIST" : !pto.ptr<T, ub>, index -> !pto.vreg<NxT>, !pto.vreg<NxT>
```

## 关键约束

- 该形式只对交错/去交错分布模式合法。
- 两个输出构成有序结果对，顺序必须保留。


## 相关页面

- [向量加载存储指令集](../../vector-load-store_zh.md)

# pto.vmulconv

执行乘法与转换融合操作。

## 语法

```mlir
%result = pto.vmulconv %lhs, %rhs : !pto.vreg<NxT0>, !pto.vreg<NxT0> -> !pto.vreg<MxT1>
```

## 关键约束

- 某些操作只在特定 profile 或元素类型上受支持。
- lowering 不能把专用语义退化为未文档化的普通算术组合。

## 相关页面

- [SFU 与 DSA 指令集](../../sfu-and-dsa-ops_zh.md)

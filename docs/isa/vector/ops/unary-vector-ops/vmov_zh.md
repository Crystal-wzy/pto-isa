# pto.vmov

复制或移动向量寄存器内容。

## 语法

```mlir
%result = pto.vmov %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
```

## 关键约束

- 指数、对数、平方根等只对文档化浮点类型合法。
- masked lane 的结果不能依赖未说明行为。

## 相关页面

- [一元向量指令集](../../unary-vector-ops_zh.md)

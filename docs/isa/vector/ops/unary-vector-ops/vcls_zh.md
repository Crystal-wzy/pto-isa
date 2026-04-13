# pto.vcls

统计前导符号位或前导模式。

## 语法

```mlir
%result = pto.vcls %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
```

A5 已文档化的支持形式：全部整数类型。

## 关键约束

- 指数、对数、平方根等只对文档化浮点类型合法。
- masked lane 的结果不能依赖未说明行为。

## 相关页面

- [一元向量指令集](../../unary-vector-ops_zh.md)

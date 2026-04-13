# pto.vshr

按激活 lane 右移。

## 语法

```mlir
%result = pto.vshr %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
```

A5 已文档化的支持形式：全部整数类型。

## 关键约束

- 输入向量与结果向量的宽度和元素类型必须兼容。
- 位运算、移位和 carry 变体还会受具体元素类型限制。

## 相关页面

- [二元向量指令集](../../binary-vector-ops_zh.md)

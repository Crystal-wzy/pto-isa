# pto.vexpdiff

执行指数差相关变换。

## 语法

```mlir
%result = pto.vexpdiff %input, %max : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>
```

A5 已文档化的支持形式：`f16` 与 `f32`。

## 关键约束

- 仅支持浮点元素类型。


## 相关页面

- [SFU 与 DSA 指令集](../../sfu-and-dsa-ops_zh.md)

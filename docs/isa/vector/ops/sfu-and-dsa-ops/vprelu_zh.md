# pto.vprelu

执行参数化 ReLU。

## 语法

```mlir
%result = pto.vprelu %input, %alpha : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>
```

A5 已文档化的支持形式：`f16` 与 `f32`。

## 关键约束

- 某些操作只在特定 profile 或元素类型上受支持。
- lowering 不能把专用语义退化为未文档化的普通算术组合。

## 相关页面

- [SFU 与 DSA 指令集](../../sfu-and-dsa-ops_zh.md)

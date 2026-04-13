# pto.vmrgsort

执行 merge sort 变体。

## 语法

### PTO 汇编形式

```asm
vmrgsort4 %dest, %src0, %src1, %src2, %src3, %count, %config : !pto.ptr<T, ub>, !pto.ptr<T, ub>, !pto.ptr<T, ub>, !pto.ptr<T, ub>, !pto.ptr<T, ub>, i64, i64
```

### AS Level 1（SSA）

```mlir
pto.vmrgsort4 %dest, %src0, %src1, %src2, %src3, %count, %config
    : !pto.ptr<f32, ub>, !pto.ptr<f32, ub>, !pto.ptr<f32, ub>, !pto.ptr<f32, ub>, !pto.ptr<f32, ub>, i64, i64
```

## 关键约束

- 某些操作只在特定 profile 或元素类型上受支持。
- lowering 不能把专用语义退化为未文档化的普通算术组合。

## 相关页面

- [SFU 与 DSA 指令集](../../sfu-and-dsa-ops_zh.md)

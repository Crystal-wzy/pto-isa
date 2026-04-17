# pto.vcgmax

`pto.vcgmax` is part of the [Reduction Instructions](../../reduction-ops.md) instruction set.

## Summary

Per-VLane-group maximum reduction.

## Mechanism

The instruction reduces each hardware 32-byte VLane group independently. Within each group, it finds the maximum of the active lanes and writes the result to the low slot of that group; the remaining lanes in each group are zero-filled.

## Syntax

### PTO Assembly Form

```text
vcgmax %dst, %src, %mask : !pto.vreg<NxT>
```

### AS Level 1 (SSA)

```mlir
%result = pto.vcgmax %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
```

## Inputs

| Operand | Type | Description |
| --- | --- | --- |
| %input | `!pto.vreg<NxT>` | Source vector register to reduce per VLane group |
| %mask | `!pto.mask` | Predicate mask; inactive lanes do not participate |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| %result | `!pto.vreg<NxT>` | One maximum per 32-byte VLane group, written to the low lane of each group |

## Side Effects

This operation has no architectural side effect beyond producing its destination values. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

- Grouping is by the hardware 32-byte VLane, not by an arbitrary software subvector.
- The mask width MUST match `N`.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- Documented A5 coverage: `i16-i32`, `f16`, `f32`.
- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.

## Performance

### Timing Disclosure

The current public VPTO timing material for PTO micro instructions remains limited.
For `pto.vcgmax`, those public sources describe the instruction semantics, operand legality, and pipeline placement, but they do **not** publish a numeric latency or steady-state throughput.

| Metric | Status | Source Basis |
|--------|--------|--------------|
| A5 latency | Not publicly published | Current public VPTO timing material |
| Steady-state throughput | Not publicly published | Current public VPTO timing material |

If software scheduling or performance modeling depends on the exact cost of `pto.vcgmax`, treat that cost as target-profile-specific and measure it on the concrete backend rather than inferring a manual constant.

## Examples

```c
for (int g = 0; g < GROUPS; g++) {
    T mx = -INF;
    for (int i = 0; i < LANES_PER_GROUP; i++)
        if (mask[g*LANES_PER_GROUP + i] && src[g*LANES_PER_GROUP + i] > mx) mx = src[g*LANES_PER_GROUP + i];
    dst[g*LANES_PER_GROUP] = mx;
}
```

```mlir
%result = pto.vcgmax %input, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Reduction Instructions](../../reduction-ops.md)
- Previous op in instruction set: [pto.vcmin](./vcmin.md)
- Next op in instruction set: [pto.vcgmin](./vcgmin.md)

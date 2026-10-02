"""Read-only diagnostics from gradients already produced by the critic update."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch as th


def collect_parameter_gradient_diagnostics(
    module: th.nn.Module, groups: Mapping[str, Sequence[str]]
) -> dict[str, float | None]:
    """Summarize named parameter groups without modifying gradients or RNG state.

    Prefixes match a parameter name or module subtree; ``""`` matches all.
    Groups may overlap, so their norms must not be added together. Missing
    gradients have a null norm; allocated all-zero gradients have norm zero.
    Only existing parameter gradients are read: no forward/backward is run.
    """
    named = tuple(module.named_parameters())
    values: dict[str, float | None] = {}
    scalars: list[tuple[str, th.Tensor]] = []
    ratios: list[str] = []

    with th.no_grad():
        for label, prefixes in groups.items():
            prefixes = (prefixes,) if isinstance(prefixes, str) else tuple(prefixes)
            selected = [p for name, p in named if any(
                prefix == "" or name == prefix or name.startswith(prefix + ".")
                for prefix in prefixes
            )]
            key = "encoder_grad/" + str(label).strip("/") + "/"
            present = [p for p in selected if p.grad is not None]
            values[key + "parameter_tensors"] = float(len(selected))
            values[key + "parameter_elements"] = float(sum(p.numel() for p in selected))
            values[key + "requires_grad_elements"] = float(sum(p.numel() for p in selected if p.requires_grad))
            values[key + "gradient_tensors"] = float(len(present))
            values[key + "gradient_parameter_elements"] = float(sum(p.numel() for p in present))
            values[key + "missing_gradient_elements"] = float(sum(p.numel() for p in selected if p.grad is None))
            values[key + "gradient_parameter_fraction"] = (
                sum(p.numel() for p in present) / sum(p.numel() for p in selected)
                if selected and sum(p.numel() for p in selected) else None
            )
            if not selected:
                values[key + "parameter_l2"] = None
                values[key + "gradient_l2"] = None
                values[key + "gradient_to_parameter_l2"] = None
                continue

            device = selected[0].device

            def total(items):
                return th.stack([item.to(device=device, dtype=th.float64) for item in items]).sum()

            scalars.append((key + "parameter_l2", total([
                p.detach().to(dtype=th.float64).square().sum() for p in selected
            ]).sqrt()))
            scalars.append((key + "parameter_nonfinite_elements", total([
                (~th.isfinite(p.detach())).sum() for p in selected
            ])))
            if not present:
                values[key + "gradient_l2"] = None
                values[key + "gradient_max_abs"] = None
                values[key + "gradient_to_parameter_l2"] = None
                values[key + "gradient_elements_examined"] = 0.0
                values[key + "gradient_nonzero_elements"] = 0.0
                values[key + "gradient_nonfinite_elements"] = 0.0
                continue

            gradients = [p.grad.detach().coalesce().values() if p.grad.is_sparse
                         else p.grad.detach() for p in present]
            values[key + "gradient_elements_examined"] = float(sum(g.numel() for g in gradients))
            scalars.append((key + "gradient_l2", total([
                g.to(dtype=th.float64).square().sum() for g in gradients
            ]).sqrt()))
            scalars.append((key + "gradient_max_abs", th.stack([
                g.abs().max().to(device=device, dtype=th.float64)
                if g.numel() else th.zeros((), device=device, dtype=th.float64)
                for g in gradients
            ]).max()))
            scalars.append((key + "gradient_nonzero_elements", total([
                (g != 0).sum() for g in gradients
            ])))
            scalars.append((key + "gradient_nonfinite_elements", total([
                (~th.isfinite(g)).sum() for g in gradients
            ])))
            ratios.append(key)

        # One host transfer for all scalar diagnostics in the usual single-device case.
        if scalars:
            device = scalars[0][1].device
            numbers = th.stack([tensor.to(device) for _, tensor in scalars]).cpu().tolist()
            for (name, _), number in zip(scalars, numbers):
                values[name] = float(number) if math.isfinite(number) else None
        for key in ratios:
            numerator, denominator = values[key + "gradient_l2"], values[key + "parameter_l2"]
            values[key + "gradient_to_parameter_l2"] = (
                numerator / denominator if numerator is not None and denominator is not None
                and denominator > 0 else None
            )
    return values


def snapshot_parameter_groups(
    module: th.nn.Module,
    groups: Mapping[str, Sequence[str]],
    optimizer: th.optim.Optimizer,
) -> dict[str, dict[str, Any]]:
    """Clone optimizer-owned parameter values for an immediately following step.

    Call only on explicitly sampled updates. The returned tensors are detached
    clones, and neither gradients nor optimizer/RNG state are read or changed.
    Parameters not owned by ``optimizer`` remain visible in the ownership
    counts but are deliberately excluded from the delta snapshots.
    """
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group.get("params", ())
    }
    named = tuple(module.named_parameters())
    snapshots: dict[str, dict[str, Any]] = {}
    with th.no_grad():
        for label, prefixes in groups.items():
            prefixes = (prefixes,) if isinstance(prefixes, str) else tuple(prefixes)
            selected = [
                (name, parameter)
                for name, parameter in named
                if any(
                    prefix == ""
                    or name == prefix
                    or name.startswith(prefix + ".")
                    for prefix in prefixes
                )
            ]
            owned = [
                (name, parameter)
                for name, parameter in selected
                if id(parameter) in optimizer_ids
            ]
            snapshots[str(label)] = {
                "parameter_tensors": len(selected),
                "parameter_elements": sum(parameter.numel() for _, parameter in selected),
                "trainable_elements": sum(
                    parameter.numel()
                    for _, parameter in selected
                    if parameter.requires_grad
                ),
                "optimizer_owned_elements": sum(
                    parameter.numel() for _, parameter in owned
                ),
                "optimizer_nonowned_elements": sum(
                    parameter.numel() for _, parameter in selected
                    if id(parameter) not in optimizer_ids
                ),
                "owned_parameters": tuple(
                    (name, parameter, parameter.detach().clone())
                    for name, parameter in owned
                ),
            }
    return snapshots


def collect_parameter_update_diagnostics(
    snapshots: Mapping[str, Mapping[str, Any]],
) -> dict[str, float | None]:
    """Measure parameter deltas after the optimizer step represented by a snapshot.

    Metrics are per group and report optimizer ownership explicitly. A group
    with no owned parameters has null deltas (no applicable optimizer update),
    rather than a misleading observed zero delta.
    """
    values: dict[str, float | None] = {}
    scalar_tensors: list[tuple[str, th.Tensor]] = []
    ratios: list[str] = []
    with th.no_grad():
        for label, row in snapshots.items():
            prefix = "encoder_update/" + str(label).strip("/") + "/"
            values[prefix + "parameter_tensors"] = float(row["parameter_tensors"])
            values[prefix + "parameter_elements"] = float(row["parameter_elements"])
            values[prefix + "trainable_elements"] = float(row["trainable_elements"])
            values[prefix + "optimizer_owned_elements"] = float(row["optimizer_owned_elements"])
            values[prefix + "optimizer_nonowned_elements"] = float(row["optimizer_nonowned_elements"])
            parameters = tuple(row["owned_parameters"])
            if not parameters:
                for metric in (
                    "delta_l2",
                    "parameter_l2_before",
                    "parameter_l2_after",
                    "relative_delta_l2",
                    "delta_max_abs",
                    "changed_elements",
                    "delta_nonfinite_elements",
                ):
                    values[prefix + metric] = None
                continue

            deltas = [parameter.detach() - before for _, parameter, before in parameters]
            before_values = [before for _, _, before in parameters]
            after_values = [parameter.detach() for _, parameter, _ in parameters]
            device = deltas[0].device

            def total(items):
                return th.stack([
                    item.to(device=device, dtype=th.float64).sum()
                    for item in items
                ]).sum()

            scalar_tensors.extend([
                (prefix + "delta_l2", total([delta.to(th.float64).square() for delta in deltas]).sqrt()),
                (prefix + "parameter_l2_before", total([value.to(th.float64).square() for value in before_values]).sqrt()),
                (prefix + "parameter_l2_after", total([value.to(th.float64).square() for value in after_values]).sqrt()),
                (prefix + "delta_max_abs", th.stack([
                    delta.abs().max().to(device=device, dtype=th.float64)
                    if delta.numel() else th.zeros((), device=device, dtype=th.float64)
                    for delta in deltas
                ]).max()),
                (prefix + "changed_elements", total([(delta != 0).sum() for delta in deltas])),
                (prefix + "delta_nonfinite_elements", total([(~th.isfinite(delta)).sum() for delta in deltas])),
            ])
            ratios.append(prefix)

        if scalar_tensors:
            device = scalar_tensors[0][1].device
            numbers = th.stack([value.to(device) for _, value in scalar_tensors]).cpu().tolist()
            for (name, _), number in zip(scalar_tensors, numbers):
                values[name] = float(number) if math.isfinite(number) else None
        for prefix in ratios:
            delta = values[prefix + "delta_l2"]
            before = values[prefix + "parameter_l2_before"]
            values[prefix + "relative_delta_l2"] = (
                delta / before
                if delta is not None and before is not None and before > 0
                else None
            )
    return values

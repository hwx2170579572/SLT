"""Read-only diagnostics from gradients already produced by the critic update."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

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

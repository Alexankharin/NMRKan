import math
from typing import Callable, Dict, List, Optional

import numpy as np
import sympy as sp
import torch
from torch import nn

from .activations import default_activation_bank

Tensor = torch.Tensor

__all__ = [
    "KharKAN",
    "DenseKanLayer",
    "_clean_expr",
]


def _clean_expr(expr: sp.Expr, eps: float = 1e-3) -> sp.Expr:
    """Zero-out numerical atoms with absolute value below ``eps``."""
    nums: Dict[sp.Number, float] = {}
    for n in expr.atoms(sp.Number):
        try:
            val = float(n)
        except (TypeError, ValueError):
            continue
        if math.isnan(val):
            continue
        nums[n] = val if abs(val) >= eps else 0
    return expr.xreplace(nums)


def _round_expr(expr: sp.Expr, digits: int = 5) -> sp.Expr:
    nums = {n: round(float(n), digits) for n in expr.atoms(sp.Number)}
    return expr.xreplace(nums)


def _fix_powers(expr: sp.Expr) -> sp.Expr:
    if not isinstance(expr, sp.Expr):
        return expr
    return expr.replace(
        lambda e: isinstance(e, sp.Pow) and isinstance(e.exp, sp.Float) and float(e.exp).is_integer(),
        lambda e: sp.Pow(e.base, int(e.exp)),
    )


class DenseKanLayer(nn.Module):
    """A single KAN layer with learnable activation mixtures."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activations: Optional[List[Callable[[Tensor], Tensor]]] = None,
        activation_reprs: Optional[List[str]] = None,
        bias: bool = True,
        include_abs: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if activations is None:
            activations, default_reprs = default_activation_bank(include_abs=include_abs)
            self.activations = activations
            if activation_reprs is None:
                activation_reprs = default_reprs
        else:
            self.activations = activations
            if activation_reprs is None:
                raise ValueError(
                    "activation_reprs must be provided when custom activations are specified"
                )

        self.activation_reprs = activation_reprs
        self.num_act = len(self.activations)

        if len(self.activation_reprs) != self.num_act:
            raise ValueError("activation_reprs must match length of activations list")

        self.act_logits = nn.Parameter(torch.zeros(input_dim, self.num_act))
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        act_outs = [act(x) for act in self.activations]
        y = torch.stack(act_outs, dim=2)
        a = torch.softmax(self.act_logits, dim=1)
        v = torch.einsum("bik,ik->bi", y, a)
        # v = (y * a.unsqueeze(0)).sum(dim=2)
        return self.linear(v)

    def L05_reg(self) -> Tensor:
        # if only 1 weight is non-zero, the L05 norm is 0
        return (
            torch.sum(torch.softmax(self.act_logits, dim=1) ** 0.5) - self.input_dim
        )  # 0.5-norm of the softmax probabilities

    def anneal(self, rate: float = 0.1) -> None:
        with torch.no_grad():
            self.act_logits.add_(rate * torch.randn_like(self.act_logits))
            self.linear.weight.add_(rate * torch.randn_like(self.linear.weight))
            if self.linear.bias is not None:
                self.linear.bias.add_(rate * torch.randn_like(self.linear.bias))

    def symbolic_formula(self, clean: bool = True, round_digits: int = 5) -> Dict[str, sp.Expr]:
        a = torch.softmax(self.act_logits, dim=1).detach().cpu().numpy()
        W = self.linear.weight.detach().cpu().numpy()
        b = self.linear.bias.detach().cpu().numpy() if self.linear.bias is not None else np.zeros(self.output_dim)

        xs = [sp.Symbol(f"x_{j}") for j in range(self.input_dim)]
        z_exprs: Dict[str, sp.Expr] = {}
        for i in range(self.output_dim):
            expr_i = 0
            for j in range(self.input_dim):
                for k in range(self.num_act):
                    coeff = W[i, j] * a[j, k]
                    if abs(coeff) < 10 ** (-round_digits):
                        continue
                    rep = self.activation_reprs[k]
                    base = xs[j]
                    if isinstance(rep, str) and rep.strip():
                        rep_str = rep.strip()
                        if rep_str.lower() in {"abs", "abs()"}:
                            base = sp.Abs(xs[j])
                        elif "{x}" in rep_str:
                            expr_str = rep_str.format(x=f"(x_{j})")
                            base = sp.sympify(expr_str)
                        else:
                            base = sp.sympify(f"(x_{j}){rep_str}")
                    expr_i += coeff * base
            if abs(b[i]) > 10 ** (-round_digits):
                expr_i += b[i]
            expr_i = sp.expand(expr_i)
            expr_i = _fix_powers(expr_i)
            if clean:
                expr_i = _clean_expr(expr_i, eps=10 ** (-round_digits))
                expr_i = _round_expr(expr_i, digits=round_digits)
            z_exprs[f"z_{i}"] = expr_i
        return z_exprs


class KharKAN(nn.Module):
    """Simple multi-layer KAN model."""

    def __init__(
        self,
        layers: Optional[tuple] = None,
        activations: Optional[List[Callable[[Tensor], Tensor]]] = None,
        activation_reprs: Optional[List[str]] = None,
        linear_bias: bool = True,
        include_abs: bool = True,
    ) -> None:
        super().__init__()
        if layers is None:
            layers = (2, 8, 3, 1)

        act_bank = activations
        repr_bank = activation_reprs
        if act_bank is None or repr_bank is None:
            default_acts, default_reprs = default_activation_bank(include_abs=include_abs)
            if act_bank is None:
                act_bank = default_acts
            if repr_bank is None:
                if act_bank is default_acts:
                    repr_bank = default_reprs
                else:
                    raise ValueError(
                        "activation_reprs must be provided when custom activations are specified"
                    )

        self.layers = nn.ModuleList(
            [
                DenseKanLayer(
                    layers[i],
                    layers[i + 1],
                    act_bank,
                    repr_bank,
                    bias=linear_bias,
                    include_abs=include_abs,
                )
                for i in range(len(layers) - 1)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        for layer in self.layers:
            x = layer(x)
        return x

    def L05_loss(self) -> Tensor:
        return sum(layer.L05_reg() for layer in self.layers)

    def anneal(self, rate: float = 0.1) -> None:
        for layer in self.layers:
            layer.anneal(rate)

    def symbolic_formula(self, clean: bool = True, round_digits: int = 5) -> Dict[str, sp.Expr]:
        formulas = [layer.symbolic_formula(clean=False, round_digits=round_digits) for layer in self.layers]
        subs_maps = []
        for fm in formulas[:-1]:
            subs_maps.append({sp.Symbol(f"x_{i}"): sp.expand(fm[f"z_{i}"]) for i in range(len(fm))})
        final = {k: sp.expand(v) for k, v in formulas[-1].items()}
        for subs in reversed(subs_maps):
            for k in final:
                final[k] = sp.expand(final[k].xreplace(subs))
        for k in final:
            if clean:
                final[k] = _clean_expr(final[k], eps=10 ** (-round_digits))
                final[k] = _round_expr(final[k], digits=round_digits)
            final[k] = _fix_powers(final[k])
        return final

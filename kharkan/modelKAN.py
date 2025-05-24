import torch
import torch.nn as nn
import sympy as sp
import numpy as np
from typing import List, Optional, Callable, Dict
import math
Tensor = torch.Tensor


def _clean_expr(expr: sp.Expr, eps: float = 1e-3) -> sp.Expr:
    """
    Zero-out any numeric atom whose absolute value < eps,
    and skip NaNs to avoid invalid Sympy comparisons.
    """
    nums = {}
    for n in expr.atoms(sp.Number):
        try:
            val = float(n)
        except (TypeError, ValueError):
            # not a plain float-convertible number → leave it alone
            continue
        if math.isnan(val):
            # skip NaNs
            continue
        nums[n] = val if abs(val) >= eps else 0
    return expr.xreplace(nums)


def _round_expr(expr: sp.Expr, digits: int = 5) -> sp.Expr:
    """
    Round all numerical atoms in the expression to the specified number of digits.
    """
    nums = {n: round(float(n), digits) for n in expr.atoms(sp.Number)}
    return expr.xreplace(nums)


def _fix_powers(expr: sp.Expr) -> sp.Expr:
    """
    Convert float exponents that are integer values (e.g., 2.0) into actual integer exponents.
    """
    # Replace Pow(base, Float) where the float is integral
    if not isinstance(expr, sp.Expr):
        return expr
    return expr.replace(
        lambda e: isinstance(e, sp.Pow) and isinstance(e.exp, sp.Float) and float(e.exp).is_integer(),
        lambda e: sp.Pow(e.base, int(e.exp))
    )


class DenseKanLayer(nn.Module):
    """
    DenseKanLayer applies a learnable convex combination of multiple activations to each input feature,
    then feeds the resulting vector through a linear transformation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activations: Optional[List[Callable[[Tensor], Tensor]]] = None,
        activation_reprs: Optional[List[str]] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Default activations: identity, square, zero
        if activations is None:
            self.activations = [lambda x: x, lambda x: x**2, lambda x: torch.zeros_like(x)]
        else:
            self.activations = activations
        self.num_act = len(self.activations)

        # Symbolic representations for each activation
        if activation_reprs is None:
            self.activation_reprs = ["", "**2", "*0"]
        else:
            self.activation_reprs = activation_reprs
        if len(self.activation_reprs) != self.num_act:
            raise ValueError("activation_reprs must match length of activations list")

        # Mixing logits for activations (will be softmaxed)
        self.act_logits = nn.Parameter(torch.zeros(input_dim, self.num_act))

        # Linear layer on mixed inputs
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        act_outs = [act(x) for act in self.activations]
        y = torch.stack(act_outs, dim=2)
        a = torch.softmax(self.act_logits, dim=1)
        v = (y * a.unsqueeze(0)).sum(dim=2)
        return self.linear(v)

    def L1_reg(self) -> Tensor:
        return torch.sum(torch.abs(self.act_logits))

    def anneal(self, rate: float = 0.1) -> None:
        with torch.no_grad():
            self.act_logits.add_(rate * torch.randn_like(self.act_logits))
            self.linear.weight.add_(rate * torch.randn_like(self.linear.weight))
            if self.linear.bias is not None:
                self.linear.bias.add_(rate * torch.randn_like(self.linear.bias))

    def symbolic_formula(
        self, clean: bool = True, round_digits: int = 5
    ) -> Dict[str, sp.Expr]:
        a = torch.softmax(self.act_logits, dim=1).detach().cpu().numpy()
        W = self.linear.weight.detach().cpu().numpy()
        b = (
            self.linear.bias.detach().cpu().numpy()
            if self.linear.bias is not None
            else np.zeros(self.output_dim)
        )
        xs = [sp.Symbol(f"x_{j}") for j in range(self.input_dim)]
        z_exprs: Dict[str, sp.Expr] = {}

        for i in range(self.output_dim):
            expr_i = 0
            for j in range(self.input_dim):
                for k in range(self.num_act):
                    coeff = W[i, j] * a[j, k]
                    if abs(coeff) < 10**(-round_digits):
                        continue
                    rep = self.activation_reprs[k]
                    if rep:
                        base = sp.sympify(f"(x_{j}){rep}")
                    else:
                        base = xs[j]
                    expr_i += coeff * base
            if abs(b[i]) > 10**(-round_digits):
                expr_i += b[i]
            expr_i = sp.expand(expr_i)
            # fix float exponents
            expr_i = _fix_powers(expr_i)
            if clean:
                expr_i = _clean_expr(expr_i, eps=10**(-round_digits))
                expr_i = _round_expr(expr_i, digits=round_digits)
            z_exprs[f"z_{i}"] = expr_i

        return z_exprs


class KharKAN(nn.Module):
    """
    KharKAN composes multiple DenseKanLayer instances for deep symbolic regression.
    """

    def __init__(
        self,
        layers: Optional[tuple] = None,
        activations: Optional[List[Callable[[Tensor], Tensor]]] = None,
        activation_reprs: Optional[List[str]] = None,
        linear_bias: bool = True,
    ) -> None:
        super().__init__()
        if layers is None:
            layers = (2, 8, 3, 1)
        self.layers = nn.ModuleList([
            DenseKanLayer(
                layers[i], layers[i + 1], activations, activation_reprs, bias=linear_bias
            ) for i in range(len(layers) - 1)
        ])

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def L1_loss(self) -> Tensor:
        return sum(layer.L1_reg() for layer in self.layers)

    def anneal(self, rate: float = 0.1) -> None:
        for layer in self.layers:
            layer.anneal(rate)

    def symbolic_formula(
        self, clean: bool = True, round_digits: int = 5
    ) -> Dict[str, sp.Expr]:
        formulas = [lay.symbolic_formula(clean=False, round_digits=round_digits) for lay in self.layers]
        subs_maps = []
        for fm in formulas[:-1]:
            subs_maps.append({sp.Symbol(f"x_{i}"): sp.expand(fm[f"z_{i}"]) for i in range(len(fm))})
        final = {k: sp.expand(v) for k, v in formulas[-1].items()}
        for subs in reversed(subs_maps):
            for k in final:
                final[k] = sp.expand(final[k].xreplace(subs))
        # fix float exponents in final
        for k in final:
            if clean:
                final[k] = _clean_expr(final[k], eps=10**(-round_digits))
                final[k] = _round_expr(final[k], digits=round_digits)
        for k in final:
            final[k] = _fix_powers(final[k])
        return final

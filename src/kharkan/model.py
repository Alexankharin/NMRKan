import torch
import sympy as sp
import numpy as np
from typing import List, Dict, Optional, Callable, Tuple

Tensor = torch.Tensor


def _clean_expr(expr: sp.Expr, eps: float = 1e-3) -> sp.Expr:
    """
    Replace numerical atoms in the expression that are smaller than eps with zero.
    """
    def replacer(val):
        return val if abs(val) >= eps else 0

    nums = {n: repl for n in expr.atoms(sp.Number) for repl in [replacer(n)]}
    return expr.xreplace(nums)


def _round_expr(expr: sp.Expr, digits: int = 5) -> sp.Expr:
    """
    Round all numerical atoms in the expression to the specified number of digits.
    """
    nums = {n: round(float(n), digits) for n in expr.atoms(sp.Number)}
    return expr.xreplace(nums)


class DenseKanLayer(torch.nn.Module):
    """
    DenseKanLayer implements a single layer of the KAN architecture:
      - A linear transformation producing multiple activation channels
      - Application of a fixed set of symbolic activations
      - A separable convolution (regression) that learns to combine these activations
    """

    def __init__(
        self,
        N_input: int,
        N_output: int,
        activations: Optional[List[Callable[[Tensor], Tensor]]] = None,
        activations_reprs: Optional[Dict[int, str]] = None,
        normalize: bool = True,
        linear_bias: bool = True,
    ) -> None:
        super().__init__()
        self.N_input = N_input
        self.N_output = N_output
        self.normalize = normalize

        if activations is None:
            activations = [torch.nn.Identity(), torch.square, lambda x: x * 0.0]
        self.activations = activations
        num_act = len(self.activations)

        if activations_reprs is None:
            activations_reprs = {0: "", 1: "**2", 2: "*0"}
        self.activations_reprs = activations_reprs

        self.linear = torch.nn.Linear(N_input, N_output * num_act, bias=linear_bias)
        self.regression_layer = torch.nn.Conv1d(
            in_channels=N_output,
            out_channels=N_output,
            kernel_size=num_act,
            groups=N_output,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        x_lin = self.linear(x)
        num_act = len(self.activations)
        x_chan = x_lin.view(-1, self.N_output, num_act)
        activated = torch.stack([
            act(x_chan[:, :, idx]) for idx, act in enumerate(self.activations)
        ], dim=2)

        with torch.no_grad():
            w = self.regression_layer.weight.data.abs()
            if self.normalize:
                sum_w = w.sum(dim=2, keepdim=True)
                sum_w = torch.where(sum_w > 0, sum_w, torch.ones_like(sum_w))
                w = w / sum_w
            self.regression_layer.weight.data.copy_(w)

        return self.regression_layer(activated).squeeze(-1)

    def L1_reg(self) -> Tensor:
        return torch.sum(self.regression_layer.weight.abs())

    def simplify(self, threshold: float = 1e-2) -> None:
        mask = torch.abs(self.regression_layer.weight.data) >= threshold
        self.regression_layer.weight.data.mul_(mask)
        with torch.no_grad():
            num_act = len(self.activations)
            reg_flat = mask.view(self.N_output * num_act, 1)
            self.linear.weight.data.mul_(reg_flat)

    def symbolic_formula(self, clean: bool = True, round_digits: int = 5) -> Dict[str, sp.Expr]:
        """
        Return a dict of final outputs z_i in terms of original x_j.
        Always expands parentheses, then optionally cleans and rounds.
        """
        # Build intermediate y expressions
        y_exprs: Dict[int, sp.Expr] = {}
        for idx in range(self.linear.out_features):
            coeffs = self.linear.weight[idx]
            expr = sum(
                coeffs[j].item() * sp.Symbol(f"x_{j}")
                for j in range(self.linear.in_features)
                if abs(coeffs[j].item()) > 0
            )
            if self.linear.bias is not None and abs(self.linear.bias[idx].item()) > 0:
                expr += self.linear.bias[idx].item()
            y_exprs[idx] = expr

        # Build final z expressions
        z_exprs: Dict[str, sp.Expr] = {}
        num_act = len(self.activations)
        for out_idx in range(self.N_output):
            terms = []
            for act_idx in range(num_act):
                w = self.regression_layer.weight[out_idx, 0, act_idx].item()
                if abs(w) > 0:
                    y_idx = out_idx * num_act + act_idx
                    base = y_exprs[y_idx]
                    repr_str = self.activations_reprs.get(act_idx, "")
                    if repr_str:
                        term = w * sp.sympify(f"({base}){repr_str}")
                    else:
                        term = w * base
                    terms.append(term)
            z_exprs[f"z_{out_idx}"] = sum(terms)

        # Expand all parentheses
        z_exprs = {k: sp.expand(v) for k, v in z_exprs.items()}

        # Optional cleaning and rounding
        if clean:
            for k, v in z_exprs.items():
                v_clean = _clean_expr(v, eps=10**(-round_digits))
                z_exprs[k] = _round_expr(v_clean, digits=round_digits)

        return z_exprs

    def print_symbolic_formula(self, **kwargs) -> None:
        z_exprs = self.symbolic_formula(**kwargs)
        for name, expr in z_exprs.items():
            print(f"{name} =")
            print(sp.pretty(expr, use_unicode=False))
            print()


class KharKAN(torch.nn.Module):
    """
    KharKAN composes multiple DenseKanLayer instances for deep symbolic regression.
    """

    def __init__(
        self,
        layers: Optional[Tuple[int, ...]] = None,
        activations: Optional[List[Callable[[Tensor], Tensor]]] = None,
        activations_reprs: Optional[Dict[int, str]] = None,
        normalize: bool = True,
        linear_bias: bool = True,
    ) -> None:
        super().__init__()
        if layers is None:
            layers = (2, 8, 3, 1)
        self.layers = torch.nn.ModuleList([
            DenseKanLayer(
                layers[i], layers[i+1], activations, activations_reprs, normalize, linear_bias
            ) for i in range(len(layers)-1)
        ])

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def simplify(self, threshold: float = 1e-2) -> None:
        for layer in self.layers:
            layer.simplify(threshold)

    def symbolic_formula(
        self, clean: bool = True, round_digits: int = 5
    ) -> Dict[str, sp.Expr]:
        # Obtain per-layer formulas without cleaning
        formulas = [lay.symbolic_formula(clean=False) for lay in self.layers]
        # Build substitution maps for all but final layer
        subs_maps: List[Dict[sp.Symbol, sp.Expr]] = []
        for fm in formulas[:-1]:
            subs_maps.append({sp.Symbol(f"x_{i}"): sp.expand(fm[f"z_{i}"])
                              for i in range(len(fm))})
        # Apply maps to final layer
        final = {k: sp.expand(v) for k, v in formulas[-1].items()}
        for subs in reversed(subs_maps):
            for k, expr in final.items():
                final[k] = sp.expand(expr.xreplace(subs))
        # Clean and round
        if clean:
            for k in final:
                v = final[k]
                v = _clean_expr(v, eps=10**(-round_digits))
                final[k] = _round_expr(v, digits=round_digits)
        return final

    def print_symbolic_formula(self, **kwargs) -> None:
        final = self.symbolic_formula(**kwargs)
        print("Symbolic formulas for each output:")
        for name, expr in final.items():
            print(f"{name} =")
            print(sp.pretty(expr, use_unicode=False))
            print()

    def symbolic_formula_latex(self, **kwargs) -> Dict[str, str]:
        return {k: sp.latex(v) for k, v in self.symbolic_formula(**kwargs).items()}

    def L1_loss(self) -> Tensor:
        return sum(layer.L1_reg() for layer in self.layers)

    def anneal(self, anneal_rate: float = 1e-1) -> None:
        for layer in self.layers:
            layer.anneal(anneal_rate)

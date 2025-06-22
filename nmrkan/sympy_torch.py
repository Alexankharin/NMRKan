import sympy as sp
import torch
import torch.nn as nn
from typing import Dict, List, Callable
from functools import reduce
import operator


def torchModelFromFormula(expr: sp.Expr) -> torch.nn.Module:
    """
    Creates a torch.nn.Module from a SymPy expression without using eval:

    - Numeric literals (except exponents) become nn.Parameters.
    - 'forward' accepts a dict mapping symbol names (e.g. 'x_0') to batched tensors.
    - Provides `show_signature()` to list expected input keys.
    - Provides `to_sympy()` to reconstruct the SymPy expression with current parameter values.

    Args:
        expr (sp.Expr): SymPy expression with symbols x_0, x_1, ..., numeric constants,
                        and operations (+, -, *, /, **).
    Returns:
        nn.Module: with parameters for constants and methods:
            - forward(inputs: Dict[str, Tensor]) -> Tensor
            - show_signature() -> List[str]
            - to_sympy() -> sp.Expr
    """
    # 1. Collect numeric constants (excluding exponents)
    nums: List[sp.Number] = []
    def collect(node, parent=None):
        if isinstance(node, sp.Pow):
            collect(node.base, node)
        elif node.is_Mul or node.is_Add:
            for arg in node.args:
                collect(arg, node)
        elif node.is_Number:
            if not (isinstance(parent, sp.Pow) and node == parent.exp):
                nums.append(node)
        else:
            for arg in node.args:
                collect(arg, node)
    collect(expr)
    unique_nums: List[sp.Number] = []
    for n in nums:
        if n not in unique_nums:
            unique_nums.append(n)

    # Map each constant to a parameter name
    param_names = [f"p{i}" for i in range(len(unique_nums))]

    # 2. Input symbol names sorted
    syms = sorted(expr.free_symbols, key=lambda s: s.name)
    input_names: List[str] = [s.name for s in syms]

    class FormulaModule(nn.Module):
        def __init__(self):
            super().__init__()
            # register parameters
            for idx, num in enumerate(unique_nums):
                val = float(num)
                param = nn.Parameter(torch.tensor(val, dtype=torch.float32))
                setattr(self, param_names[idx], param)
            self.input_names: List[str] = input_names
            # compile the expression into a callable
            self._compute = self._build_func(expr)

        def _build_func(self, node) -> Callable[[Dict[str, torch.Tensor]], torch.Tensor]:
            if node.is_Number:
                # constant parameter or fixed literal
                try:
                    idx = unique_nums.index(node)
                    pname = param_names[idx]
                    return lambda inputs: getattr(self, pname)
                except ValueError:
                    val = float(node)
                    return lambda inputs: torch.tensor(val, dtype=torch.float32)
            if node.is_Symbol:
                return lambda inputs: inputs[node.name]
            if node.is_Add:
                funcs = [self._build_func(arg) for arg in node.args]
                return lambda inputs: sum(f(inputs) for f in funcs)
            if node.is_Mul:
                funcs = [self._build_func(arg) for arg in node.args]
                return lambda inputs: reduce(operator.mul, (f(inputs) for f in funcs))
            if isinstance(node, sp.Pow) or node.is_Pow:
                base_func = self._build_func(node.base)
                exp = node.exp
                if exp.is_Number and exp not in unique_nums:
                    e = float(exp)
                    return lambda inputs: torch.pow(base_func(inputs), e)
                else:
                    exp_func = self._build_func(exp)
                    return lambda inputs: torch.pow(base_func(inputs), exp_func(inputs))
            raise NotImplementedError(f"Unsupported node type: {node}")

        def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
            missing = set(self.input_names) - set(inputs.keys())
            if missing:
                raise KeyError(f"Missing inputs: {missing}")
            for name in self.input_names:
                tensor = inputs[name]
                if tensor.dim() < 2:
                    raise ValueError(f"Input '{name}' must have batch dimension, got {tensor.shape}")
            return self._compute(inputs)

        def show_signature(self) -> List[str]:
            return list(self.input_names)

        def to_sympy(self) -> sp.Expr:
            subs_map = {num: float(getattr(self, pname).item())
                       for num, pname in zip(unique_nums, param_names)}
            return expr.subs(subs_map)

        def __repr__(self):
            sig = ", ".join(self.input_names)
            return f"<FormulaModule inputs=[{sig}] params={param_names}>"

    return FormulaModule()
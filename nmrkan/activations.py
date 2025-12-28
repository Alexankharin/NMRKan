"""Utility functions for KAN activation banks."""

from typing import Callable, List, Tuple

import torch

ActivationFn = Callable[[torch.Tensor], torch.Tensor]


def default_activation_bank(include_abs: bool = True) -> Tuple[List[ActivationFn], List[str]]:
    """Build the default activation functions and their symbolic representations.

    Args:
        include_abs: Whether to include the absolute value basis function.

    Returns:
        A tuple of (activation_functions, activation_representations).
    """

    def identity(x: torch.Tensor) -> torch.Tensor:
        return x

    def quadratic(x: torch.Tensor) -> torch.Tensor:
        return x ** 2

    def absolute(x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x)

    def zero(x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    activations: List[ActivationFn] = [identity, quadratic]
    activation_reprs: List[str] = ["", "**2"]

    if include_abs:
        activations.append(absolute)
        activation_reprs.append("Abs")

    activations.append(zero)
    activation_reprs.append("*0")

    return activations, activation_reprs

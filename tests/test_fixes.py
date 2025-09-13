#!/usr/bin/env python3
"""
Test script to verify the MSE formatting fix and variable name replacements.
"""

import numpy as np
import sympy as sp
from report_utils import safe_format_float

def test_safe_format_float():
    """Test the safe_format_float function with various inputs."""
    print("Testing safe_format_float function:")
    
    # Test cases
    test_cases = [
        (1.234e-5, "1.23e-05"),        # Normal float
        ("1.234e-5", "1.23e-05"),      # String float
        (None, "N/A"),                 # None value
        (float('inf'), "N/A"),         # Infinity
        (float('nan'), "N/A"),         # NaN
        ("invalid", "N/A"),            # Invalid string
        (0.0, "0.00e+00"),            # Zero
        (1234.5678, "1.23e+03"),      # Large number
    ]
    
    for value, expected in test_cases:
        result = safe_format_float(value)
        status = "✓" if result == expected else "✗"
        print(f"  {status} safe_format_float({value!r}) = {result!r} (expected: {expected!r})")

def test_variable_substitution():
    """Test that variable substitution works for symbolic expressions."""
    print("\nTesting variable substitution:")
    
    # Create test expressions
    x0, x1, x2, x3 = sp.symbols('x_0 x_1 x_2 x_3')
    j, dJ = sp.symbols('j dJ')
    
    # Test 2D substitution: x0=deltaJ, x1=deltaJ/Jintra
    expr_2d = x0**2 + x1
    expected_2d = dJ**2 + dJ/j
    result_2d = expr_2d.subs({x0: dJ, x1: dJ/j})
    print(f"  2D: {expr_2d} -> {result_2d} (expected: {expected_2d})")
    
    # Test 3D substitution: x0=deltaJ, x1=deltaJ/Jintra, x2=Jintra
    expr_3d = x0 * x1 + x2
    expected_3d = dJ * (dJ/j) + j  # Simplifies to dJ²/j + j
    result_3d = expr_3d.subs({x0: dJ, x1: dJ/j, x2: j})
    print(f"  3D: {expr_3d} -> {result_3d} (expected: {expected_3d})")
    
    # Test 4D substitution: x0=deltaJ, x1=deltaJ/Jintra, x2=Jintra, x3=Jintra/deltaJ
    expr_4d = x0 + x1 + x2 + x3
    expected_4d = dJ + dJ/j + j + j/dJ
    result_4d = expr_4d.subs({x0: dJ, x1: dJ/j, x2: j, x3: j/dJ})
    print(f"  4D: {expr_4d} -> {result_4d} (expected: {expected_4d})")

def test_dataset_generation():
    """Test that the dataset generation works correctly."""
    print("\nTesting dataset generation:")

    from run_many_experiments import make_dataset_from_function
    from data_generation import get_frequences_ordered, get_perturbation
    
    # Test 2D dataset
    print("  Testing 2D dataset:")
    data_2d = make_dataset_from_function(
        100, get_frequences_ordered,
        min_Jintra=-2, max_Jintra=-1, min_deltaJ=-0.7, max_deltaJ=-0.1,
        ratio_threshold=100, input_mode="2d"
    )
    print(f"    Input shape: {data_2d['train_input'].shape} (expected: (100, 2))")
    print(f"    Label shape: {data_2d['train_label'].shape} (expected: (100, 3))")
    
    # Test 3D dataset
    print("  Testing 3D dataset:")
    data_3d = make_dataset_from_function(
        100, get_frequences_ordered,
        min_Jintra=-2, max_Jintra=-1, min_deltaJ=-0.7, max_deltaJ=-0.1,
        ratio_threshold=100, input_mode="3d"
    )
    print(f"    Input shape: {data_3d['train_input'].shape} (expected: (100, 3))")
    print(f"    Label shape: {data_3d['train_label'].shape} (expected: (100, 3))")
    
    # Test perturbation calculation for 3D (should use x2=Jintra, x0=deltaJ)
    jintra_vals = data_3d['train_input'][:5, 2:3].numpy()  # x2 = Jintra
    deltaj_vals = data_3d['train_input'][:5, 0:1].numpy()  # x0 = deltaJ
    pert_result = get_perturbation(jintra_vals, deltaj_vals)
    print(f"    Perturbation result shape: {pert_result.shape} (expected: (5, 3))")
    
    # Test 4D dataset
    print("  Testing 4D dataset:")
    data_4d = make_dataset_from_function(
        100, get_frequences_ordered,
        min_Jintra=-2, max_Jintra=-1, min_deltaJ=-0.7, max_deltaJ=-0.1,
        ratio_threshold=100, input_mode="4d"
    )
    print(f"    Input shape: {data_4d['train_input'].shape} (expected: (100, 4))")
    print(f"    Label shape: {data_4d['train_label'].shape} (expected: (100, 3))")

if __name__ == "__main__":
    test_safe_format_float()
    test_variable_substitution()
    test_dataset_generation()
    print("\nAll tests completed!")

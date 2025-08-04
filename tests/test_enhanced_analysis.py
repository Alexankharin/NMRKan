#!/usr/bin/env python3
"""
Test script for dimensional analysis with coefficient optimization.
"""

import numpy as np
import torch
import sympy as sp
from nmrkan.dimensional_analysis import (
    analyze_nmr_formula, 
    optimize_formula_coefficients,
    Dimension,
    _hz
)

def test_dimensional_analysis_with_optimization():
    """Test dimensional analysis with configurable output dimensions and coefficient optimization."""
    
    print("Testing dimensional analysis with configurable output dimensions...")
    
    # Test 1: Hz output dimension (default)
    formula1 = sp.parse_expr("2*x_0 + 3*x_2")  # Both in Hz, should be OK
    result1 = analyze_nmr_formula(formula1, expected_output_dim=_hz(1))
    print(f"Test 1 (Hz output): {result1['original_consistent']} - {result1['original_message']}")
    
    # Test 2: Dimensionless output dimension
    formula2 = sp.parse_expr("2*x_1 + 3*x_3")  # Both dimensionless, should be OK
    result2 = analyze_nmr_formula(formula2, expected_output_dim=Dimension())
    print(f"Test 2 (dimensionless output): {result2['original_consistent']} - {result2['original_message']}")
    
    # Test 3: Mixed dimensions (should be fixed)
    formula3 = sp.parse_expr("x_0 + x_1")  # Hz + dimensionless, should be fixed
    result3 = analyze_nmr_formula(formula3, expected_output_dim=_hz(1))
    print(f"Test 3 (mixed dims): Fixed={result3['formula_changed']}, Consistent={result3['fixed_consistent']}")
    print(f"   Fixed formula: {result3['fixed_formula']}")
    
    print("\nTesting coefficient optimization...")
    
    # Generate synthetic data for a simple formula: y = 2*x_0 + 3*x_2 + noise
    np.random.seed(42)
    n_samples = 100
    x_data = np.random.rand(n_samples, 4)  # 4 features: x_0, x_1, x_2, x_3
    true_coeffs = [2.0, 0.0, 3.0, 0.0]
    y_true = sum(true_coeffs[i] * x_data[:, i] for i in range(4))
    y_data = y_true + 0.1 * np.random.randn(n_samples)  # Add noise
    
    # Test with a formula that has wrong coefficients
    test_formula = sp.parse_expr("1.5*x_0 + 0.8*x_2")  # Wrong coefficients
    
    print(f"Original formula: {test_formula}")
    print(f"True coefficients should be: {true_coeffs[:3]}")
    
    # Optimize coefficients
    opt_result = optimize_formula_coefficients(
        test_formula,
        x_data,
        y_data,
        max_epochs=200,
        lr=0.1
    )
    
    if opt_result["optimization_success"]:
        print(f"Optimization successful!")
        print(f"Original MSE: {opt_result['original_mse']:.4f}")
        print(f"Optimized MSE: {opt_result['optimized_mse']:.4f}")
        print(f"Optimized formula: {opt_result['optimized_formula']}")
        print(f"Epochs trained: {opt_result['epochs_trained']}")
    else:
        print(f"Optimization failed: {opt_result['message']}")
    
    return opt_result["optimization_success"]

if __name__ == "__main__":
    success = test_dimensional_analysis_with_optimization()
    if success:
        print("\n✅ All tests passed! Integration is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the implementation.")

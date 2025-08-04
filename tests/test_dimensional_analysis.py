#!/usr/bin/env python3
"""
Test dimensional analysis functionality for NMR formulas.
"""

import sys
import os
sys.path.append('.')

import sympy as sp
from nmrkan.dimensional_analysis import (
    Dimension, get_variable_dimensions, analyze_nmr_formula,
    check_dimensional_consistency, apply_dimensional_analysis,
    get_nmr_output_dimension
)

def test_dimensional_analysis():
    """Test the dimensional analysis system."""
    print("=== Testing Dimensional Analysis for NMR Formulas ===\n")
    
    # Define symbols
    x0, x1, x2, x3 = sp.symbols('x_0 x_1 x_2 x_3')
    j, dJ = sp.symbols('j dJ')
    
    # Get variable dimensions
    var_dims = get_variable_dimensions()
    print("Variable dimensions:")
    for var, dim in var_dims.items():
        print(f"  {var}: {dim}")
    print()
    
    # Test cases
    test_formulas = [
        # Good formulas (dimensionally consistent)
        ("Simple frequency", dJ, True),
        ("Sum of frequencies", dJ + j, True),
        ("Frequency combination", dJ + j / 2, True),
        ("Perturbation theory", dJ / sp.sqrt(2) + dJ**2 / (8 * j), True),
        (
            "Complex mixed",
            x0 * x1 + x2 * x3,
            True,
        ),  # Hz*dimensionless + Hz*dimensionless
        (
            "polynomial frequency",
            -0.03 * dJ * j**2.0
            - 1.23 * dJ
            + 0.19 * dJ**2.0 / j**2.0
            + 0.02 * j
            + 0.01 * j**2.0
            + 0.05,
            False,
        ),
        # Bad formulas (dimensionally inconsistent)
        ("Mixed dimensions in sum", dJ + dJ / j, False),  # Hz + dimensionless
        ("Dimensional exponent", sp.exp(dJ), False),  # exp(Hz) is invalid
        ("More mixed dimensions", x0 + x1, False),  # Hz + dimensionless
        # Formulas that can be fixed
        ("Fixable mixed sum", dJ + j + dJ / j, True),  # Should keep frequency terms
        (
            "Multiple exp terms",
            sp.exp(dJ / j) + sp.exp(dJ),
            True,
        ),  # Should zero out exp(dJ)
    ]
    
    print("Testing formulas:")
    print("-" * 80)
    
    for name, formula, should_be_consistent in test_formulas:
        print(f"\nFormula: {name}")
        print(f"Expression: {formula}")
        
        # Analyze the formula
        analysis = analyze_nmr_formula(formula, var_dims)
        
        print(f"Original consistent: {analysis['original_consistent']}")
        print(f"Original message: {analysis['original_message']}")
        
        if analysis['formula_changed']:
            print(f"Fixed formula: {analysis['fixed_formula']}")
            print(f"Fixed consistent: {analysis['fixed_consistent']}")
            print(f"Fixed message: {analysis['fixed_message']}")
        else:
            print("No changes needed")
        
        # Check if result matches expectation
        final_consistent = analysis['fixed_consistent'] if analysis['formula_changed'] else analysis['original_consistent']
        status = "✓" if final_consistent == should_be_consistent else "✗"
        print(f"Result: {status} {'PASS' if final_consistent == should_be_consistent else 'FAIL'}")
        print("-" * 40)

def test_dimension_operations():
    """Test dimension arithmetic operations."""
    print("\n=== Testing Dimension Operations ===\n")
    
    freq_dim = Dimension(frequency=1)
    dimensionless_dim = Dimension(dimensionless=1)
    
    print("Basic dimensions:")
    print(f"  Frequency: {freq_dim}")
    print(f"  Dimensionless: {dimensionless_dim}")
    
    print("\nDimension operations:")
    
    # Multiplication
    product = freq_dim * dimensionless_dim
    print(f"  Hz * dimensionless = {product}")
    
    # Division  
    quotient = freq_dim / freq_dim
    print(f"  Hz / Hz = {quotient}")
    print(f"  Is dimensionless? {quotient.is_dimensionless()}")
    
    # Powers
    squared = freq_dim ** 2
    print(f"  Hz^2 = {squared}")
    
    # Try to add incompatible dimensions (should raise error)
    try:
        result = freq_dim + dimensionless_dim
        print(f"  ERROR: Should not be able to add Hz + dimensionless")
    except ValueError as e:
        print(f"  ✓ Correctly caught error: {e}")

def test_real_nmr_formulas():
    """Test with realistic NMR formulas."""
    print("\n=== Testing Real NMR Formulas ===\n")
    
    # Define symbols 
    j, dJ = sp.symbols('j dJ')
    
    # Real perturbation theory formulas
    formulas = {
        'f0': dJ * sp.sqrt(2),
        'f1': dJ / sp.sqrt(2) + dJ**2 / (8*j),
        'f2': dJ / sp.sqrt(2) - dJ**2 / (8*j),
    }
    
    print("Perturbation theory formulas:")
    for name, formula in formulas.items():
        analysis = analyze_nmr_formula(formula)
        print(f"\n{name}: {formula}")
        print(f"  Consistent: {analysis['original_consistent']}")
        print(f"  Message: {analysis['original_message']}")
        if analysis['formula_changed']:
            print(f"  Fixed: {analysis['fixed_formula']}")

if __name__ == "__main__":
    test_dimension_operations()
    test_dimensional_analysis()
    test_real_nmr_formulas()
    print("\n=== Dimensional Analysis Tests Complete ===")

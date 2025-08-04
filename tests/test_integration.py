#!/usr/bin/env python3
"""Test the dimensional analysis integration with run_many_experiments."""

import sys
sys.path.append('.')

import sympy as sp
from run_many_experiments import apply_dimensional_fixing

def test_dimensional_fixing():
    """Test the dimensional fixing functionality."""
    print("Testing dimensional analysis integration...")
    
    # Create test formulas with dimensional issues
    x0, x1, x2, x3 = sp.symbols('x_0 x_1 x_2 x_3')
    
    test_formulas = {
        'z_0': x0 + x1,  # Hz + dimensionless (should be fixed)
        'z_1': x0 * x1 + x2,  # Hz*dimensionless + Hz = Hz (should be OK)
        'z_2': None  # No formula
    }
    
    print("Original formulas:")
    for name, formula in test_formulas.items():
        print(f"  {name}: {formula}")
    
    # Apply dimensional fixing
    fixed_formulas, consistency_flags, messages = apply_dimensional_fixing(test_formulas, 3)
    
    print("\nFixed formulas:")
    for name in test_formulas.keys():
        print(f"  {name}: {fixed_formulas.get(name)} (consistent: {consistency_flags.get(name)})")
        print(f"    Message: {messages.get(name)}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_dimensional_fixing()

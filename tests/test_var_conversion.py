#!/usr/bin/env python3
"""
Test variable name conversion for coefficient optimization.
"""

import sympy as sp
import numpy as np

def test_variable_conversion():
    """Test that physics variable names are correctly converted to/from x_i format."""
    
    # Define symbols
    j, dJ = sp.symbols("j dJ")
    x0, x1, x2, x3 = sp.symbols("x_0 x_1 x_2 x_3")
    
    # Test formulas with physics names (as they come from extract_symbolic_formulas)
    formula_physics = sp.parse_expr("2*dJ + 3*j")  # Example formula
    
    print("Testing variable name conversion...")
    print(f"Original physics formula: {formula_physics}")
    
    # Test conversion for different input dimensions
    for n_features in [2, 3, 4]:
        print(f"\nTesting {n_features}D conversion:")
        
        # Convert to torch format (physics -> x_i)
        if n_features == 2:
            torch_formula = formula_physics.subs({dJ: x0, dJ/j: x1})
        elif n_features == 3:
            torch_formula = formula_physics.subs({dJ: x0, dJ/j: x1, j: x2})
        elif n_features == 4:
            torch_formula = formula_physics.subs({dJ: x0, dJ/j: x1, j: x2, j/dJ: x3})
        
        print(f"  Torch format: {torch_formula}")
        
        # Convert back to physics format (x_i -> physics)
        if n_features == 2:
            back_to_physics = torch_formula.subs({x0: dJ, x1: dJ/j})
        elif n_features == 3:
            back_to_physics = torch_formula.subs({x0: dJ, x1: dJ/j, x2: j})
        elif n_features == 4:
            back_to_physics = torch_formula.subs({x0: dJ, x1: dJ/j, x2: j, x3: j/dJ})
        
        print(f"  Back to physics: {back_to_physics}")
        
        # Check if we get back the original (for cases where the mapping makes sense)
        if n_features >= 3:  # Only for 3D and 4D where we have both dJ and j
            print(f"  Round-trip successful: {back_to_physics.equals(formula_physics)}")
    
    # Test complex formula with ratios
    complex_formula = sp.parse_expr("dJ + dJ/j*2 + j*0.5")
    print(f"\nComplex formula test: {complex_formula}")
    
    # 3D conversion
    torch_complex = complex_formula.subs({dJ: x0, dJ/j: x1, j: x2})
    print(f"Torch format: {torch_complex}")
    
    back_complex = torch_complex.subs({x0: dJ, x1: dJ/j, x2: j})
    print(f"Back to physics: {back_complex}")
    print(f"Round-trip successful: {back_complex.equals(complex_formula)}")
    
    print("\n✅ Variable conversion test completed!")

if __name__ == "__main__":
    test_variable_conversion()

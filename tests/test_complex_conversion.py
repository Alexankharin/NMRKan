#!/usr/bin/env python3
"""
Test complex variable name conversion for coefficient optimization.
"""

import sympy as sp

def test_complex_variable_conversion():
    """Test that complex physics expressions are correctly converted to/from x_i format."""
    
    # Define symbols
    j, dJ = sp.symbols("j dJ")
    x0, x1, x2, x3 = sp.symbols("x_0 x_1 x_2 x_3")
    
    # Test complex formulas that would come from the actual experiment
    test_formulas = [
        "0.28*dJ/j**1.0 - 0.36*dJ**2.0/j**2.0 + 0.06",
        "0.5*dJ/j**1.0 - 0.7*dJ**2.0/j**2.0 + 0.13",
        "dJ**2.0/j**2.0 + 0.06",
        "2*dJ + 3*j + dJ/j",
        "dJ**3/j**2 + j/dJ"
    ]
    
    print("Testing complex variable name conversion...")
    
    for formula_str in test_formulas:
        formula = sp.parse_expr(formula_str)
        print(f"\nOriginal formula: {formula}")
        
        # Test 3D conversion (most common case)
        n_features = 3
        print(f"Testing {n_features}D conversion:")
        
        # Convert to torch format (physics -> x_i)
        # Order matters: ratios first, then individual variables
        torch_formula = formula.subs(dJ/j, x1)  # First substitute ratios
        torch_formula = torch_formula.subs(j/dJ, 1/x1)  # Handle inverse ratios
        torch_formula = torch_formula.subs(dJ, x0)  # Then substitute dJ
        torch_formula = torch_formula.subs(j, x2)   # Finally substitute j
        torch_formula = sp.simplify(torch_formula)
        
        print(f"  Torch format: {torch_formula}")
        
        # Convert back to physics format (x_i -> physics)
        back_to_physics = torch_formula.subs(1/x1, j/dJ)  # Handle inverse ratios first
        back_to_physics = back_to_physics.subs(x1, dJ/j)         # Then ratios
        back_to_physics = back_to_physics.subs(x0, dJ)           # Then dJ
        back_to_physics = back_to_physics.subs(x2, j)            # Finally j
        back_to_physics = sp.simplify(back_to_physics)
        
        print(f"  Back to physics: {back_to_physics}")
        
        # Check if we get back something equivalent
        diff = sp.simplify(back_to_physics - formula)
        round_trip_ok = diff == 0
        print(f"  Round-trip successful: {round_trip_ok}")
        if not round_trip_ok:
            print(f"    Difference: {diff}")
        
        # Check that torch_formula doesn't contain j or dJ symbols
        torch_symbols = torch_formula.free_symbols
        has_physics_symbols = any(str(sym) in ['j', 'dJ'] for sym in torch_symbols)
        print(f"  Clean torch conversion (no j/dJ): {not has_physics_symbols}")
        if has_physics_symbols:
            physics_syms = [str(sym) for sym in torch_symbols if str(sym) in ['j', 'dJ']]
            print(f"    Remaining physics symbols: {physics_syms}")
    
    print("\n✅ Complex variable conversion test completed!")

if __name__ == "__main__":
    test_complex_variable_conversion()

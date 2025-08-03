#!/usr/bin/env python3
"""Ultra simple test."""

print("Starting test...")

try:
    import sys
    print("Sys import: OK")
    
    sys.path.append('.')
    print("Path append: OK")
    
    from nmrkan.dimensional_analysis import analyze_nmr_formula
    print("Module import: OK")
    
    import sympy as sp
    print("SymPy import: OK")
    
    j, dJ = sp.symbols('j dJ')
    formula = dJ + j
    print(f"Formula created: {formula}")
    
    result = analyze_nmr_formula(formula)
    print(f"Analysis result keys: {list(result.keys())}")
    print(f"Original consistent: {result['original_consistent']}")
    
    print("SUCCESS: All tests passed!")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

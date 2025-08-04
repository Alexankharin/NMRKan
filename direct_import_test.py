#!/usr/bin/env python3
"""Direct import test."""

print("Starting direct import test...")

try:
    import sys
    import os
    sys.path.append('.')
    sys.path.append('./nmrkan')
    
    print("Attempting direct import...")
    import dimensional_analysis
    print("Direct import successful!")
    
    import sympy as sp
    j, dJ = sp.symbols('j dJ')
    formula = dJ + j
    
    result = dimensional_analysis.analyze_nmr_formula(formula)
    print(f"Analysis successful: {result['original_consistent']}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

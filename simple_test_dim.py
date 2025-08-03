#!/usr/bin/env python3
"""Simple test of dimensional analysis."""

import sys
sys.path.append('.')

from nmrkan.dimensional_analysis import Dimension, analyze_nmr_formula
import sympy as sp

# Test basic functionality
print('Testing dimensional analysis...')
j, dJ = sp.symbols('j dJ')

# Test 1: Consistent formula
formula1 = dJ + j
result1 = analyze_nmr_formula(formula1)
print(f'Formula 1: {formula1}')
print(f'Consistent: {result1["original_consistent"]}')
print(f'Message: {result1["original_message"]}')

# Test 2: Inconsistent formula that should be fixed
formula2 = dJ + dJ/j  # Hz + dimensionless
result2 = analyze_nmr_formula(formula2)
print(f'\nFormula 2: {formula2}')
print(f'Original consistent: {result2["original_consistent"]}')
print(f'Fixed formula: {result2["fixed_formula"]}')
print(f'Fixed consistent: {result2["fixed_consistent"]}')

print('\nTest completed successfully!')

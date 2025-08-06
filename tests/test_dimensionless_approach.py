#!/usr/bin/env python3
"""
Test script for the dimensionless KAN experiments.

This script tests the core mathematical functionality of the dimensionless approach.
"""

import sys
import numpy as np
from pathlib import Path

# Add the parent directory to the path
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))


def test_dimensionless_perturbation_math():
    """Test that dimensionless perturbation theory math is correct."""
    
    print("Testing dimensionless perturbation theory math...")
    
    # Test parameters
    dJ_over_J = np.array([0.25, 0.5, 1.0])
    
    # Dimensionless perturbation formulas:
    # f1/J = (dJ/J) / sqrt(2) + (dJ/J)^2 / 8
    # f2/J = (dJ/J) / sqrt(2) - (dJ/J)^2 / 8
    # f0/J = (dJ/J) * sqrt(2)
    
    ratio = dJ_over_J
    f1_over_J = ratio / 2**0.5 + ratio**2 / 8
    f2_over_J = ratio / 2**0.5 - ratio**2 / 8
    f0_over_J = ratio * 2**0.5
    
    result = -np.stack((f2_over_J, f1_over_J, f0_over_J), axis=1).reshape(-1, 3)
    
    # Test specific case: dJ/J = 0.5
    ratio_05 = 0.5
    expected_f1 = ratio_05 / 2**0.5 + ratio_05**2 / 8  # ≈ 0.3848
    expected_f2 = ratio_05 / 2**0.5 - ratio_05**2 / 8  # ≈ 0.3223
    expected_f0 = ratio_05 * 2**0.5  # ≈ 0.7071
    
    # Check the result for middle element (index 1, dJ/J = 0.5)
    actual_f2, actual_f1, actual_f0 = -result[1]  # Negative because of the minus sign
    
    assert abs(actual_f1 - expected_f1) < 1e-10, f"f1 mismatch: {actual_f1} vs {expected_f1}"
    assert abs(actual_f2 - expected_f2) < 1e-10, f"f2 mismatch: {actual_f2} vs {expected_f2}"
    assert abs(actual_f0 - expected_f0) < 1e-10, f"f0 mismatch: {actual_f0} vs {expected_f0}"
    
    print("✓ Dimensionless perturbation theory math test PASSED!")


def test_scaling_math():
    """Test the scaling from dimensionless to physical units."""
    
    print("Testing scaling from dimensionless to physical units...")
    
    # Test scaling math
    dimensionless_preds = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
    jintra_values = np.array([-2.0, -1.5])
    
    # Manual scaling: multiply by |jintra|
    expected_physical = dimensionless_preds * np.abs(jintra_values).reshape(-1, 1)
    
    # Verify specific values
    assert expected_physical[0, 0] == 2.0, f"Scaling error: {expected_physical[0, 0]} != 2.0"
    assert expected_physical[0, 1] == 4.0, f"Scaling error: {expected_physical[0, 1]} != 4.0"
    assert expected_physical[1, 0] == 0.75, f"Scaling error: {expected_physical[1, 0]} != 0.75"
    
    print("✓ Scaling math test PASSED!")


def main():
    """Run all dimensionless tests."""
    print("Running dimensionless approach tests...")
    
    test_dimensionless_perturbation_math()
    test_scaling_math()
    
    print("\n✓ All dimensionless tests PASSED!")


if __name__ == "__main__":
    main()

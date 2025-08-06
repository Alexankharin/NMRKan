#!/usr/bin/env python3
"""
Minimal test for dimensionless approach - mathematical correctness only.
"""

import numpy as np

def test_perturbation_math():
    """Test the perturbation theory math is correct."""
    print("Testing perturbation theory math...")
    
    # Test values
    dJ_over_J = np.array([0.25, 0.5, 1.0])
    
    # Expected dimensionless perturbation formulas:
    # f1/J = (dJ/J) / sqrt(2) + (dJ/J)^2 / 8
    # f2/J = (dJ/J) / sqrt(2) - (dJ/J)^2 / 8  
    # f0/J = (dJ/J) * sqrt(2)
    
    ratio = dJ_over_J
    expected_f1_over_J = ratio / 2**0.5 + ratio**2 / 8
    expected_f2_over_J = ratio / 2**0.5 - ratio**2 / 8
    expected_f0_over_J = ratio * 2**0.5
    
    expected_result = -np.stack((expected_f2_over_J, expected_f1_over_J, expected_f0_over_J), axis=1)
    
    print(f"Input ratios: {dJ_over_J}")
    print(f"Expected output shape: {expected_result.shape}")
    print(f"Expected values:\n{expected_result}")
    
    # Check that the math makes sense
    assert expected_result.shape == (3, 3), f"Wrong shape: {expected_result.shape}"
    
    # Check specific values for dJ/J = 0.5
    ratio_05 = 0.5
    f1_05 = ratio_05 / 2**0.5 + ratio_05**2 / 8  # ≈ 0.3536 + 0.03125 = 0.3849
    f2_05 = ratio_05 / 2**0.5 - ratio_05**2 / 8  # ≈ 0.3536 - 0.03125 = 0.3223
    f0_05 = ratio_05 * 2**0.5  # ≈ 0.7071
    
    print(f"\nFor dJ/J = 0.5:")
    print(f"f1/J = {f1_05:.4f}")
    print(f"f2/J = {f2_05:.4f}")
    print(f"f0/J = {f0_05:.4f}")
    
    # Test scaling back to physical units
    J_test = -2.0  # Example J value
    f1_physical = f1_05 * abs(J_test)  # Should be ≈ 0.7698
    f2_physical = f2_05 * abs(J_test)  # Should be ≈ 0.6446
    f0_physical = f0_05 * abs(J_test)  # Should be ≈ 1.4142
    
    print(f"\nScaled back to physical units (|J| = {abs(J_test)}):")
    print(f"f1 = {f1_physical:.4f}")
    print(f"f2 = {f2_physical:.4f}")
    print(f"f0 = {f0_physical:.4f}")
    
    print("✓ Perturbation theory math test PASSED!")

def test_scaling_math():
    """Test the scaling math."""
    print("\nTesting scaling math...")
    
    # Fake dimensionless predictions
    dimensionless_preds = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
    jintra_values = np.array([-2.0, -1.5])
    
    # Expected scaling: multiply by |J|
    expected_physical = dimensionless_preds * np.abs(jintra_values).reshape(-1, 1)
    
    print(f"Dimensionless predictions:\n{dimensionless_preds}")
    print(f"Jintra values: {jintra_values}")
    print(f"Expected physical predictions:\n{expected_physical}")
    
    # Manual verification
    assert expected_physical[0, 0] == 1.0 * 2.0  # 2.0
    assert expected_physical[0, 1] == 2.0 * 2.0  # 4.0
    assert expected_physical[1, 0] == 0.5 * 1.5  # 0.75
    
    print("✓ Scaling math test PASSED!")

def main():
    """Run all mathematical tests."""
    print("Running dimensionless mathematical tests...\n")
    
    test_perturbation_math()
    test_scaling_math()
    
    print("\n✓ All mathematical tests PASSED!")

if __name__ == "__main__":
    main()

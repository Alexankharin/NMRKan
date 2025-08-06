#!/usr/bin/env python3
"""
Quick test to verify the dimensionless script works without device errors.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add the current directory to the path
if "." not in sys.path:
    sys.path.append(".")

from run_many_experiments_dimensionless import (
    scale_predictions_to_physical_units,
    get_perturbation_dimensionless,
)

def test_device_compatibility():
    """Test that device operations work correctly."""
    
    print("Testing device compatibility...")
    
    # Test the scaling function with different tensor locations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    
    # Create test data
    dimensionless_preds = torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
    jintra_values = torch.tensor([-2.0, -1.5])
    
    # Test GPU case if available
    if device.type == "cuda":
        print("Testing GPU->CPU scaling...")
        gpu_preds = dimensionless_preds.to(device)
        cpu_result = scale_predictions_to_physical_units(gpu_preds.cpu(), jintra_values)
        expected = dimensionless_preds * torch.abs(jintra_values).unsqueeze(1)
        
        error = torch.abs(cpu_result - expected).max()
        assert error < 1e-10, f"GPU->CPU scaling failed: {error}"
        print("✓ GPU->CPU scaling test PASSED!")
    
    # Test CPU case
    print("Testing CPU scaling...")
    cpu_result = scale_predictions_to_physical_units(dimensionless_preds, jintra_values)
    expected = dimensionless_preds * torch.abs(jintra_values).unsqueeze(1)
    
    error = torch.abs(cpu_result - expected).max()
    assert error < 1e-10, f"CPU scaling failed: {error}"
    print("✓ CPU scaling test PASSED!")
    
    # Test perturbation function
    print("Testing perturbation function...")
    dJ_over_J = np.array([0.25, 0.5])
    result = get_perturbation_dimensionless(dJ_over_J)
    assert result.shape == (2, 3), f"Wrong shape: {result.shape}"
    print("✓ Perturbation function test PASSED!")
    
    print("\n✓ All device compatibility tests PASSED!")

if __name__ == "__main__":
    test_device_compatibility()

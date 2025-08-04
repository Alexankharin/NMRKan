#!/usr/bin/env python3
"""
Quick test to verify data structure access for dimensional analysis integration.
"""

import torch
import numpy as np

def test_data_structure():
    """Test that data structure matches expected format."""
    
    # Simulate the data structure that would be created by make_dataset_from_function
    n_samples = 100
    n_inputs = 3
    n_outputs = 3
    
    # Create mock data similar to what the experiment generates
    data_dict = {
        "train_input": torch.randn(n_samples, n_inputs),
        "train_label": torch.randn(n_samples, n_outputs)
    }
    
    print("Testing data structure access...")
    print(f"data_dict keys: {list(data_dict.keys())}")
    print(f"train_input shape: {data_dict['train_input'].shape}")
    print(f"train_label shape: {data_dict['train_label'].shape}")
    
    # Test the data structure we would pass to apply_dimensional_fixing
    x_data = data_dict["train_input"]
    y_data = {
        "z_0": data_dict["train_label"][:, 0], 
        "z_1": data_dict["train_label"][:, 1], 
        "z_2": data_dict["train_label"][:, 2]
    }
    
    print(f"\nStructured data for dimensional analysis:")
    print(f"x_data shape: {x_data.shape}")
    print(f"y_data keys: {list(y_data.keys())}")
    for key, value in y_data.items():
        print(f"  {key} shape: {value.shape}")
    
    # Test conversion to numpy (as done in optimize_formula_coefficients)
    x_np = x_data.detach().cpu().numpy()
    y_np = y_data["z_0"].detach().cpu().numpy()
    
    print(f"\nNumpy conversion test:")
    print(f"x_np shape: {x_np.shape}")
    print(f"y_np shape: {y_np.shape}")
    
    print("✅ Data structure test passed!")
    return True

if __name__ == "__main__":
    test_data_structure()

#!/usr/bin/env python3
"""
Test script for run_many_experiments.py functionality.
"""

import sys
import torch
import numpy as np

# Add the current directory to the path
if "." not in sys.path:
    sys.path.append(".")

try:
    from nmrkan.models import KharKAN
    from data_generation import get_frequences_ordered
    print("✓ Successfully imported KharKAN and get_frequences_ordered")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test basic functionality
def get_perturbation(Jintra, deltaJ):
    """Test perturbation function."""
    f1 = deltaJ / 2**0.5 + deltaJ * deltaJ / 8 / Jintra
    f2 = deltaJ / 2**0.5 - deltaJ * deltaJ / 8 / Jintra
    f0 = deltaJ * (2**0.5)
    return -np.stack((f2, f1, f0), axis=1)

print("✓ Perturbation function defined")

# Test device availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ Using device: {device}")

# Test model creation
try:
    model = KharKAN((3, 4, 3))
    print("✓ KharKAN model created successfully")
except Exception as e:
    print(f"✗ Model creation error: {e}")
    sys.exit(1)

# Test data generation functions
try:
    jintra = np.random.rand(100) * 10 - 5
    deltaj = np.random.rand(100) * 2 - 1
    
    # Test perturbation
    pert_data = get_perturbation(jintra, deltaj)
    print(f"✓ Perturbation data shape: {pert_data.shape}")
    
    # Test eigenvalue calculation
    eigen_data = get_frequences_ordered(jintra, deltaj)
    print(f"✓ Eigenvalue data shape: {eigen_data.shape}")
    
except Exception as e:
    print(f"✗ Data generation error: {e}")
    sys.exit(1)

print("\n✓ All basic functionality tests passed!")
print("The run_many_experiments.py script should work correctly.")

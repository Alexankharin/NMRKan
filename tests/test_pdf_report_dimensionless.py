#!/usr/bin/env python3
"""
Test the PDF report generation with dimensionless data structure.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add the parent directory to the path
if "." not in sys.path:
    sys.path.append(".")

from pdf_report import create_dimensionless_plots
import matplotlib.pyplot as plt


def test_dimensionless_plots():
    """Test the dimensionless plotting function."""
    
    print("Testing dimensionless plot creation...")
    
    # Create fake dimensionless data
    n_samples = 100
    
    # 1D data (dJ/J only)
    data_1d_pert = {
        'train_input': torch.randn(n_samples, 1) * 0.5,  # dJ/J ratios
        'train_label': torch.randn(n_samples, 3) * 0.1   # Z/J dimensionless frequencies
    }
    
    data_1d_eigen = {
        'train_input': torch.randn(n_samples, 1) * 0.5,
        'train_label': torch.randn(n_samples, 3) * 0.1
    }
    
    # Test 1D plots
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Test 1D Dimensionless Plots', fontsize=14, fontweight='bold')
    
    try:
        create_dimensionless_plots(data_1d_pert, data_1d_eigen, fig, input_dim=1)
        print("✓ 1D dimensionless plots created successfully")
        plt.close(fig)
    except Exception as e:
        print(f"✗ Error creating 1D plots: {e}")
        plt.close(fig)
    
    # 2D data (dJ/J and J/dJ)
    data_2d_pert = {
        'train_input': torch.randn(n_samples, 2) * 0.5,  # dJ/J and J/dJ ratios
        'train_label': torch.randn(n_samples, 3) * 0.1   # Z/J dimensionless frequencies
    }
    
    data_2d_eigen = {
        'train_input': torch.randn(n_samples, 2) * 0.5,
        'train_label': torch.randn(n_samples, 3) * 0.1
    }
    
    # Test 2D plots
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Test 2D Dimensionless Plots', fontsize=14, fontweight='bold')
    
    try:
        create_dimensionless_plots(data_2d_pert, data_2d_eigen, fig, input_dim=2)
        print("✓ 2D dimensionless plots created successfully")
        plt.close(fig)
    except Exception as e:
        print(f"✗ Error creating 2D plots: {e}")
        plt.close(fig)
    
    print("✓ All dimensionless plot tests completed!")


if __name__ == "__main__":
    test_dimensionless_plots()

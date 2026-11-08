#!/usr/bin/env python3
"""
Simple test for dimensionless functions.
"""

import sys
import numpy as np
from pathlib import Path

# Add the parent directory to the path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from data_generation.datasets import get_perturbation_dimensionless


def test_dimensionless_frequency_sum():
    ratios = np.array([.25, .5])
    result = get_perturbation_dimensionless(ratios)
    assert result.shape == (2, 3)
    np.testing.assert_allclose(result[:, 0]+result[:, 1], result[:, 2])
    np.testing.assert_allclose(result[:, 2], -np.sqrt(2)*ratios)

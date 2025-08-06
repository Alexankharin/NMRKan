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

try:
    from create_many_experiments_dimensionless import get_perturbation_dimensionless
    print("✓ Import successful")
    
    # Simple test
    dJ_over_J = np.array([0.25, 0.5])
    result = get_perturbation_dimensionless(dJ_over_J)
    print(f"✓ Function call successful: {result.shape}")
    print("✓ All basic tests PASSED!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

#!/usr/bin/env python3
"""Debug safe_format_float function."""

import numpy as np

def safe_format_float(value, format_str=":.2e", default="N/A"):
    """Safely format a float value, returning default if not a number."""
    print(f"Debug - Input value: {value} (type: {type(value)})")
    
    try:
        # Handle None values
        if value is None:
            print("Debug - Value is None")
            return default

        # Convert to float first to handle string numbers
        if isinstance(value, str):
            print("Debug - Value is string, trying to convert")
            try:
                value = float(value)
                print(f"Debug - Converted to: {value}")
            except (ValueError, TypeError):
                print("Debug - String conversion failed")
                return default

        # Check if it's a valid number
        print(f"Debug - isinstance check: {isinstance(value, (int, float))}")
        print(f"Debug - isnan check: {np.isnan(value)}")
        print(f"Debug - isinf check: {np.isinf(value)}")
        
        if (
            isinstance(value, (int, float))
            and not np.isnan(value)
            and not np.isinf(value)
        ):
            result = f"{float(value):{format_str}}"
            print(f"Debug - Formatting successful: {result}")
            return result
        else:
            print("Debug - Failed validation checks")
            return default
    except (ValueError, TypeError, AttributeError) as e:
        print(f"Debug - Exception caught: {e}")
        return default

# Test with the known problematic value
test_value = 0.11685662716627121
print("Testing with valid float:")
result = safe_format_float(test_value)
print(f"Result: {result}")

# Test with inf
print("\nTesting with inf:")
result = safe_format_float(float('inf'))
print(f"Result: {result}")

# Test with nan
print("\nTesting with nan:")
result = safe_format_float(float('nan'))
print(f"Result: {result}")

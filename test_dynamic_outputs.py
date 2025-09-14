#!/usr/bin/env python3
"""
Test the dynamic output detection in report_utils.py
"""

import sys
sys.path.append('.')

from report_utils import extract_experiment_data_for_csv

# Test data with different numbers of outputs
test_results = [
    {
        'experiment_id': 'test_3outputs',
        'architecture': [1, 8, 8],
        'lr': 0.0001,
        'l05_penalty': 0.1,
        'perturbation': None,
        'eigenvalue': {
            'final_mse': 2.10,
            'formulas': {'z_0': 'x_0 + 0.5', 'z_1': 'x_0**2', 'z_2': 'x_0**3'},
            'refined_formulas': {'z_0': 'refined_z_0', 'z_1': 'refined_z_1'},
            'symbolic_mse_scores': {'z_0': 0.1, 'z_1': 0.2, 'z_2': 0.3}
        }
    },
    {
        'experiment_id': 'test_8outputs',
        'architecture': [1, 8, 8],
        'lr': 0.0001,
        'l05_penalty': 0.1,
        'perturbation': None,
        'eigenvalue': {
            'final_mse': 1.50,
            'formulas': {f'z_{i}': f'x_0**{i+1}' for i in range(8)},  # z_0 through z_7
            'refined_formulas': {f'z_{i}': f'refined_z_{i}' for i in range(4)},  # z_0 through z_3
            'symbolic_mse_scores': {f'z_{i}': 0.1 * (i+1) for i in range(8)}
        }
    }
]

print("Testing dynamic output detection...")

try:
    csv_data = extract_experiment_data_for_csv(test_results)
    print(f"✅ SUCCESS: CSV data extracted: {len(csv_data)} rows")
    
    # Count outputs for each experiment
    exp1_outputs = [row for row in csv_data if row['experiment_id'] == 'test_3outputs']
    exp2_outputs = [row for row in csv_data if row['experiment_id'] == 'test_8outputs']
    
    print(f"✅ Experiment 1 (3 outputs): {len(exp1_outputs)} rows generated")
    print(f"✅ Experiment 2 (8 outputs): {len(exp2_outputs)} rows generated")
    
    # Show sample output names
    if exp1_outputs:
        print(f"✅ Exp1 output names: {[row['output_frequency'] for row in exp1_outputs]}")
    if exp2_outputs:
        print(f"✅ Exp2 output names: {[row['output_frequency'] for row in exp2_outputs]}")

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("Test completed.")
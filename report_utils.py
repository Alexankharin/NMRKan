#!/usr/bin/env python3
"""
Report utilities for KAN NMR experiments.

This module contains functions for generating reports, tables, and CSV exports
from experiment results.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from typing import Dict, List, Any, Optional
from pathlib import Path


def safe_format_float(value, format_str=":.2e", default="N/A"):
    """Safely format a float value, returning default if not a number."""
    try:
        if isinstance(value, (int, float)) and not np.isnan(value) and value != float('inf'):
            return f"{float(value):{format_str}}"
        else:
            return default
    except (ValueError, TypeError):
        return default


def extract_experiment_data_for_csv(all_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract experiment data in a format suitable for CSV export."""
    csv_data = []
    
    for result in all_results:
        if 'error' in result:
            continue
            
        exp_id = result['experiment_id']
        params_name = exp_id.split('_')[0] if '_' in exp_id else 'unknown'
        architecture = result['architecture']
        input_dim = architecture[0] if architecture else 0
        lr = result.get('lr', 'N/A')
        l05_penalty = result.get('l05_penalty', 'N/A')
        
        for data_type in ['perturbation', 'eigenvalue']:
            if data_type in result and 'error' not in result[data_type]:
                data = result[data_type]
                
                # Extract basic experiment info
                base_info = {
                    'experiment_id': exp_id,
                    'param_set': params_name.upper(),
                    'data_type': data_type.capitalize(),
                    'input_dimension': f"{input_dim}D",
                    'architecture': str(architecture),
                    'learning_rate': lr,
                    'l05_penalty': l05_penalty,
                    'neural_mse': safe_format_float(data.get('final_mse')),
                    'total_original_complexity': data.get('total_complexity', 0),
                }
                
                # Extract formulas and their properties
                formulas = data.get('formulas', {})
                refined_formulas = data.get('refined_formulas', {})
                symbolic_mse_scores = data.get('symbolic_mse_scores', {})
                
                # Calculate total refined complexity
                total_refined_complexity = sum(
                    sp.count_ops(f) for f in refined_formulas.values() if f is not None
                )
                base_info['total_refined_complexity'] = total_refined_complexity
                
                # Add formula-specific data for each output
                for output_name in ['z_0', 'z_1', 'z_2']:
                    original_formula = formulas.get(output_name)
                    refined_formula = refined_formulas.get(output_name)
                    symbolic_mse = symbolic_mse_scores.get(output_name, float('inf'))
                    
                    # Create a row for this specific output
                    row_data = base_info.copy()
                    row_data.update({
                        'output': output_name,
                        'original_formula': str(original_formula) if original_formula else 'None',
                        'original_complexity': sp.count_ops(original_formula) if original_formula else 0,
                        'refined_formula': str(refined_formula) if refined_formula else 'None',
                        'refined_complexity': sp.count_ops(refined_formula) if refined_formula else 0,
                        'symbolic_mse': safe_format_float(symbolic_mse),
                        'formula_found': 'Yes' if original_formula else 'No',
                        'formula_refined': 'Yes' if refined_formula else 'No'
                    })
                    
                    csv_data.append(row_data)
    
    return csv_data


def save_results_to_csv(all_results: List[Dict[str, Any]], output_dir: Path) -> None:
    """Save experiment results to CSV file."""
    csv_data = extract_experiment_data_for_csv(all_results)
    
    if not csv_data:
        print("No data available for CSV export")
        return
    
    # Create DataFrame
    df = pd.DataFrame(csv_data)
    
    # Sort by parameter set, data type, and neural MSE
    df = df.sort_values(['param_set', 'data_type', 'neural_mse'])
    
    # Save to CSV
    csv_path = output_dir / 'experiment_results.csv'
    df.to_csv(csv_path, index=False)
    
    print(f"CSV results saved to: {csv_path}")
    
    # Also create a summary CSV with best results only
    summary_data = create_summary_csv_data(all_results)
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = output_dir / 'experiment_summary.csv'
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Summary CSV saved to: {summary_csv_path}")


def create_summary_csv_data(all_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create summary data for CSV export with best results per parameter set and architecture."""
    
    # Group results by parameter set, input dimension, and data type
    grouped_results = {}
    
    for result in all_results:
        if 'error' in result:
            continue
            
        exp_id = result['experiment_id']
        params_name = exp_id.split('_')[0] if '_' in exp_id else 'unknown'
        architecture = result['architecture']
        input_dim = f"{architecture[0]}D" if architecture else "unknownD"
        
        for data_type in ['perturbation', 'eigenvalue']:
            if data_type in result and 'error' not in result[data_type]:
                key = f"{params_name}_{input_dim}_{data_type}"
                if key not in grouped_results:
                    grouped_results[key] = []
                grouped_results[key].append(result)
    
    # Find best results for each group
    summary_data = []
    
    for group_key, group_results in grouped_results.items():
        params_name, input_dim, data_type = group_key.split('_')
        
        # Find best result by Neural MSE
        valid_results = [r for r in group_results 
                        if data_type in r and 'error' not in r[data_type]]
        
        if valid_results:
            best_result = min(valid_results, 
                            key=lambda x: x[data_type].get('final_mse', float('inf')))
            
            data = best_result[data_type]
            final_mse = data.get('final_mse', float('inf'))
            total_complexity_orig = data.get('total_complexity', 0)
            
            # Calculate metrics
            refined_formulas = data.get('refined_formulas', {})
            total_complexity_refined = sum(sp.count_ops(f) for f in refined_formulas.values() if f is not None)
            
            symbolic_mses = data.get('symbolic_mse_scores', {})
            valid_symbolic_mses = [mse for mse in symbolic_mses.values() if mse != float('inf')]
            best_symbolic_mse = min(valid_symbolic_mses) if valid_symbolic_mses else float('inf')
            
            successful_formulas = sum(1 for f in data.get('formulas', {}).values() if f is not None)
            successful_refined = sum(1 for f in refined_formulas.values() if f is not None)
            
            summary_data.append({
                'param_set': params_name.upper(),
                'input_dimension': input_dim,
                'data_type': data_type.capitalize(),
                'best_architecture': str(best_result['architecture']),
                'best_experiment_id': best_result['experiment_id'],
                'neural_mse': safe_format_float(final_mse),
                'best_symbolic_mse': safe_format_float(best_symbolic_mse),
                'formulas_found': f"{successful_formulas}/3",
                'formulas_refined': f"{successful_refined}/3",
                'original_complexity': total_complexity_orig,
                'refined_complexity': total_complexity_refined,
                'learning_rate': best_result.get('lr', 'N/A'),
                'l05_penalty': best_result.get('l05_penalty', 'N/A')
            })
    
    return summary_data


def print_console_summary(all_results: List[Dict[str, Any]]) -> None:
    """Print a formatted summary to console."""
    
    successful_results = [r for r in all_results if 'error' not in r]
    total_experiments = len(all_results)
    
    print(f"\nSummary: {len(successful_results)}/{total_experiments} experiments completed successfully")
    
    # Group results by parameter set
    results_by_params = {}
    for result in successful_results:
        exp_id = result['experiment_id']
        params_name = exp_id.split('_')[0] if '_' in exp_id else 'unknown'
        if params_name not in results_by_params:
            results_by_params[params_name] = []
        results_by_params[params_name].append(result)
    
    print(f"\nResults by Parameter Set:")
    for params_name, param_results in results_by_params.items():
        print(f"\n{params_name.upper()} Parameter Set:")
        print(f"  Completed experiments: {len(param_results)}")
        
        if param_results:
            # Find best perturbation result
            pert_results = [r for r in param_results 
                           if 'perturbation' in r and 'error' not in r['perturbation']]
            if pert_results:
                best_pert = min(pert_results, 
                               key=lambda x: x.get('perturbation', {}).get('final_mse', float('inf')))
                best_pert_mse = best_pert.get('perturbation', {}).get('final_mse', float('inf'))
                print(f"  Best Perturbation MSE: {safe_format_float(best_pert_mse)} ({best_pert['experiment_id']})")
            
            # Find best eigenvalue result
            eigen_results = [r for r in param_results 
                            if 'eigenvalue' in r and 'error' not in r['eigenvalue']]
            if eigen_results:
                best_eigen = min(eigen_results,
                                key=lambda x: x.get('eigenvalue', {}).get('final_mse', float('inf')))
                best_eigen_mse = best_eigen.get('eigenvalue', {}).get('final_mse', float('inf'))
                print(f"  Best Eigenvalue MSE: {safe_format_float(best_eigen_mse)} ({best_eigen['experiment_id']})")
    
    # Overall best results
    if successful_results:
        all_pert_results = [r for r in successful_results 
                           if 'perturbation' in r and 'error' not in r['perturbation']]
        all_eigen_results = [r for r in successful_results 
                            if 'eigenvalue' in r and 'error' not in r['eigenvalue']]
        
        if all_pert_results:
            overall_best_pert = min(all_pert_results, 
                                   key=lambda x: x.get('perturbation', {}).get('final_mse', float('inf')))
            best_pert_mse = overall_best_pert.get('perturbation', {}).get('final_mse', float('inf'))
            
            print(f"\nOverall Best Results:")
            print(f"  Best Perturbation Theory Result:")
            print(f"    Experiment: {overall_best_pert['experiment_id']}")
            print(f"    MSE: {safe_format_float(best_pert_mse)}")
        
        if all_eigen_results:
            overall_best_eigen = min(all_eigen_results,
                                    key=lambda x: x.get('eigenvalue', {}).get('final_mse', float('inf')))
            best_eigen_mse = overall_best_eigen.get('eigenvalue', {}).get('final_mse', float('inf'))
            
            print(f"  Best Eigenvalue Result:")
            print(f"    Experiment: {overall_best_eigen['experiment_id']}")
            print(f"    MSE: {safe_format_float(best_eigen_mse)}")


def print_detailed_console_summary(all_results: List[Dict[str, Any]]) -> None:
    """Print detailed summary table to console."""
    
    successful_results = [r for r in all_results if 'error' not in r]
    
    print(f"\n{'='*100}")
    print("COMPREHENSIVE RESULTS SUMMARY")
    print(f"{'='*100}")
    print(f"{'Param Set':<12} {'Input':<5} {'Data Type':<12} {'Neural MSE':<12} {'Best Sym MSE':<12} {'Formulas':<8} {'Orig Compl':<10} {'Ref Compl':<10}")
    print("-" * 100)
    
    # Group results by parameter set and input dimension for summary
    summary_stats = {}
    
    for result in successful_results:
        exp_id = result['experiment_id']
        params_name = exp_id.split('_')[0] if '_' in exp_id else 'unknown'
        architecture = result['architecture']
        input_dim = f"{architecture[0]}D" if architecture else "unknownD"
        
        key = f"{params_name}_{input_dim}"
        if key not in summary_stats:
            summary_stats[key] = {'perturbation': [], 'eigenvalue': []}
        
        for data_type in ['perturbation', 'eigenvalue']:
            if data_type in result and 'error' not in result[data_type]:
                summary_stats[key][data_type].append(result[data_type])
    
    # Print summary for each parameter set and dimension
    for key, data in summary_stats.items():
        params_name, input_dim = key.split('_')
        
        for data_type, results_list in data.items():
            if results_list:
                # Find best neural MSE
                best_neural_mse = min(r.get('final_mse', float('inf')) for r in results_list)
                
                # Find best symbolic MSE across all outputs
                all_symbolic_mses = []
                total_formulas = 0
                total_orig_complexity = 0
                total_refined_complexity = 0
                
                for r in results_list:
                    symbolic_mses = r.get('symbolic_mse_scores', {})
                    all_symbolic_mses.extend([mse for mse in symbolic_mses.values() if mse != float('inf')])
                    
                    formulas = r.get('formulas', {})
                    total_formulas += sum(1 for f in formulas.values() if f is not None)
                    
                    total_orig_complexity += r.get('total_complexity', 0)
                    
                    refined_formulas = r.get('refined_formulas', {})
                    total_refined_complexity += sum(sp.count_ops(f) for f in refined_formulas.values() if f is not None)
                
                best_symbolic_mse = min(all_symbolic_mses) if all_symbolic_mses else float('inf')
                avg_orig_complexity = total_orig_complexity / len(results_list) if results_list else 0
                avg_refined_complexity = total_refined_complexity / len(results_list) if results_list else 0
                total_possible_formulas = len(results_list) * 3  # 3 outputs per experiment
                
                print(f"{params_name.upper():<12} {input_dim:<5} {data_type.capitalize():<12} "
                      f"{safe_format_float(best_neural_mse):<12} {safe_format_float(best_symbolic_mse):<12} "
                      f"{total_formulas}/{total_possible_formulas:<8} {avg_orig_complexity:<10.1f} {avg_refined_complexity:<10.1f}")
    
    print("-" * 100)
    print("Legend: Param Set = Parameter Set, Input = Input Dimension, Neural MSE = Best Neural Network MSE")
    print("        Best Sym MSE = Best Symbolic MSE, Formulas = Successful/Total, Compl = Average Complexity")
    print(f"{'='*100}")

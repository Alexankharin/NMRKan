#!/usr/bin/env python3
"""
PDF report generation utilities for KAN NMR experiments.

This module contains functions for generating PDF reports with plots and tables.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Any, Tuple
from pathlib import Path

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

from report_utils import safe_format_float


def create_dimensionless_plots(data_pert, data_eigen, fig, input_dim=1):
    """Create plots for dimensionless data (1D or 2D inputs)."""

    try:
        # Extract data for plotting
        X_pert = data_pert["train_input"].numpy()
        Z_pert = data_pert["train_label"].numpy()

        X_eigen = data_eigen["train_input"].numpy()
        Z_eigen = data_eigen["train_label"].numpy()

        print("Plotting dimensionless data shapes:")
        print(f"  X_pert: {X_pert.shape}, Z_pert: {Z_pert.shape}")
        print(f"  X_eigen: {X_eigen.shape}, Z_eigen: {Z_eigen.shape}")

        # Sample subset for cleaner plots
        n_samples = min(2000, len(X_pert))
        indices = np.random.choice(len(X_pert), n_samples, replace=False)

        if input_dim == 1:
            # 1D input plots (dJ/J vs output)
            for i in range(min(3, Z_pert.shape[1] if len(Z_pert.shape) > 1 else 1)):
                # Perturbation data
                ax = fig.add_subplot(2, 3, i + 1)

                if len(Z_pert.shape) > 1 and Z_pert.shape[1] > i:
                    z_data_pert = Z_pert[indices, i]
                else:
                    z_data_pert = (
                        Z_pert[indices]
                        if i == 0
                        else np.zeros_like(indices, dtype=float)
                    )

                scatter = ax.scatter(
                    X_pert[indices, 0],
                    z_data_pert,
                    c=z_data_pert,
                    cmap="viridis",
                    s=2,
                    alpha=0.7,
                )
                ax.set_xlabel("dJ/J (dimensionless)")
                ax.set_ylabel(f"Z{i}/J (dimensionless)")
                ax.set_title(f"Perturbation Theory - Output {i}")
                ax.grid(True, alpha=0.3)
                try:
                    plt.colorbar(scatter, ax=ax, shrink=0.8)
                except Exception as e:
                    print(f"Colorbar error for 1D perturbation plot {i}: {e}")

                # Eigenvalue data
                ax = fig.add_subplot(2, 3, i + 4)

                if len(Z_eigen.shape) > 1 and Z_eigen.shape[1] > i:
                    z_data_eigen = Z_eigen[indices, i]
                else:
                    z_data_eigen = (
                        Z_eigen[indices]
                        if i == 0
                        else np.zeros_like(indices, dtype=float)
                    )

                scatter = ax.scatter(
                    X_eigen[indices, 0],
                    z_data_eigen,
                    c=z_data_eigen,
                    cmap="plasma",
                    s=2,
                    alpha=0.7,
                )
                ax.set_xlabel("dJ/J (dimensionless)")
                ax.set_ylabel(f"Z{i}/J (dimensionless)")
                ax.set_title(f"Eigenvalue - Output {i}")
                ax.grid(True, alpha=0.3)
                try:
                    plt.colorbar(scatter, ax=ax, shrink=0.8)
                except Exception as e:
                    print(f"Colorbar error for 1D eigenvalue plot {i}: {e}")

        elif input_dim == 2:
            # 2D input plots (create 3D scatter plots with dJ/J, J/dJ, output)
            for i in range(min(3, Z_pert.shape[1] if len(Z_pert.shape) > 1 else 1)):
                # Perturbation data
                ax = fig.add_subplot(2, 3, i + 1, projection="3d")

                if len(Z_pert.shape) > 1 and Z_pert.shape[1] > i:
                    z_data_pert = Z_pert[indices, i]
                else:
                    z_data_pert = (
                        Z_pert[indices]
                        if i == 0
                        else np.zeros_like(indices, dtype=float)
                    )

                scatter = ax.scatter(
                    X_pert[indices, 0],
                    X_pert[indices, 1],
                    z_data_pert,
                    c=z_data_pert,
                    cmap="viridis",
                    s=1,
                    alpha=0.6,
                )
                ax.set_xlabel("dJ/J (dimensionless)")
                ax.set_ylabel("J/dJ (dimensionless)")
                ax.set_zlabel(f"Z{i}/J (dimensionless)")
                ax.set_title(f"Perturbation Theory - Output {i}")
                try:
                    plt.colorbar(scatter, ax=ax, shrink=0.5)
                except Exception as e:
                    print(f"Colorbar error for 2D perturbation plot {i}: {e}")

                # Eigenvalue data
                ax = fig.add_subplot(2, 3, i + 4, projection="3d")

                if len(Z_eigen.shape) > 1 and Z_eigen.shape[1] > i:
                    z_data_eigen = Z_eigen[indices, i]
                else:
                    z_data_eigen = (
                        Z_eigen[indices]
                        if i == 0
                        else np.zeros_like(indices, dtype=float)
                    )

                scatter = ax.scatter(
                    X_eigen[indices, 0],
                    X_eigen[indices, 1],
                    z_data_eigen,
                    c=z_data_eigen,
                    cmap="plasma",
                    s=1,
                    alpha=0.6,
                )
                ax.set_xlabel("dJ/J (dimensionless)")
                ax.set_ylabel("J/dJ (dimensionless)")
                ax.set_zlabel(f"Z{i}/J (dimensionless)")
                ax.set_title(f"Eigenvalue - Output {i}")
                try:
                    plt.colorbar(scatter, ax=ax, shrink=0.5)
                except Exception as e:
                    print(f"Colorbar error for 2D eigenvalue plot {i}: {e}")

    except Exception as e:
        print(f"Error creating dimensionless plots: {e}")
        # Create a simple text plot instead
        ax = fig.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            f"Error creating dimensionless plots: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.axis("off")


def create_surface_plots(data_pert, data_eigen, fig, start_subplot=1):
    """Create 3D surface plots for the data."""
    
    try:
        # Extract data for plotting
        X_pert = data_pert['train_input'][:, 0].numpy()
        Y_pert = data_pert['train_input'][:, 1].numpy()
        Z_pert = data_pert['train_label'].numpy()
        
        X_eigen = data_eigen['train_input'][:, 0].numpy()
        Y_eigen = data_eigen['train_input'][:, 1].numpy()  
        Z_eigen = data_eigen['train_label'].numpy()

        print("Plotting data shapes:")
        print(f"  X_pert: {X_pert.shape}, Y_pert: {Y_pert.shape}, Z_pert: {Z_pert.shape}")
        print(f"  X_eigen: {X_eigen.shape}, Y_eigen: {Y_eigen.shape}, Z_eigen: {Z_eigen.shape}")
        
        # Sample subset for cleaner plots
        n_samples = min(2000, len(X_pert))
        indices = np.random.choice(len(X_pert), n_samples, replace=False)
        
        for i in range(min(3, Z_pert.shape[1] if len(Z_pert.shape) > 1 else 1)):  # Handle both 2D and 1D outputs
            # Perturbation data
            ax = fig.add_subplot(2, 3, start_subplot + i, projection='3d')
            
            if len(Z_pert.shape) > 1 and Z_pert.shape[1] > i:
                z_data_pert = Z_pert[indices, i]
            else:
                z_data_pert = Z_pert[indices] if i == 0 else np.zeros_like(indices, dtype=float)
                
            scatter = ax.scatter(X_pert[indices], Y_pert[indices], z_data_pert, 
                               c=z_data_pert, cmap='viridis', s=1, alpha=0.6)
            ax.set_xlabel('Jintra')
            ax.set_ylabel('deltaJ')
            ax.set_zlabel(f'Frequency {i}')
            ax.set_title(f'Perturbation Theory - Output {i}')
            try:
                plt.colorbar(scatter, ax=ax, shrink=0.5)
            except Exception as e:
                print(f"Colorbar error for perturbation plot {i}: {e}")
            
            # Eigenvalue data
            ax = fig.add_subplot(2, 3, start_subplot + i + 3, projection='3d')
            
            if len(Z_eigen.shape) > 1 and Z_eigen.shape[1] > i:
                z_data_eigen = Z_eigen[indices, i]
            else:
                z_data_eigen = Z_eigen[indices] if i == 0 else np.zeros_like(indices, dtype=float)
                
            scatter = ax.scatter(X_eigen[indices], Y_eigen[indices], z_data_eigen, 
                               c=z_data_eigen, cmap='plasma', s=1, alpha=0.6)
            ax.set_xlabel('Jintra')
            ax.set_ylabel('deltaJ')
            ax.set_zlabel(f'Frequency {i}')
            ax.set_title(f'Eigenvalue - Output {i}')
            try:
                plt.colorbar(scatter, ax=ax, shrink=0.5)
            except Exception as e:
                print(f"Colorbar error for eigenvalue plot {i}: {e}")
                
    except Exception as e:
        print(f"Error creating surface plots: {e}")
        # Create a simple text plot instead
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f'Error creating plots: {str(e)}', 
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')


def plot_training_curves(results, fig, start_subplot=1):
    """Plot training curves for loss, MSE, and relative error."""
    
    if 'perturbation' not in results or 'eigenvalue' not in results:
        return
        
    pert_data = results['perturbation']
    eigen_data = results['eigenvalue']
    
    if 'error' in pert_data or 'error' in eigen_data:
        return
    
    # Loss curves
    ax1 = fig.add_subplot(2, 3, start_subplot)
    if 'loss_history' in pert_data:
        ax1.plot(pert_data['loss_history'], label='Perturbation', alpha=0.8)
    if 'loss_history' in eigen_data:
        ax1.plot(eigen_data['loss_history'], label='Eigenvalue', alpha=0.8)
    ax1.set_yscale('log')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MSE curves
    ax2 = fig.add_subplot(2, 3, start_subplot + 1)
    if 'mse_history' in pert_data:
        ax2.plot(pert_data['mse_history'], label='Perturbation', alpha=0.8)
    if 'mse_history' in eigen_data:
        ax2.plot(eigen_data['mse_history'], label='Eigenvalue', alpha=0.8)
    ax2.set_yscale('log')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE')
    ax2.set_title('Mean Squared Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Relative error curves
    ax3 = fig.add_subplot(2, 3, start_subplot + 2)
    if 'rel_err_history' in pert_data:
        ax3.plot(pert_data['rel_err_history'], label='Perturbation', alpha=0.8)
    if 'rel_err_history' in eigen_data:
        ax3.plot(eigen_data['rel_err_history'], label='Eigenvalue', alpha=0.8)
    ax3.set_yscale('log')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Relative Error')
    ax3.set_title('Relative Error')
    ax3.legend()
    ax3.grid(True, alpha=0.3)


def create_experiment_summary_page(all_results, fig):
    """Create a summary page with experiment comparison."""
    
    # Clear the figure
    fig.clear()
    
    # Create summary statistics
    summary_data = []
    for result in all_results:
        if 'perturbation' in result and 'eigenvalue' in result:
            pert = result['perturbation']
            eigen = result['eigenvalue']
            
            if 'error' not in pert and 'error' not in eigen:
                summary_data.append({
                    'Architecture': str(result['architecture']),
                    'LR': result['lr'],
                    'L05': result['l05_penalty'],
                    'Pert_MSE': pert.get('final_mse', float('inf')),
                    'Eigen_MSE': eigen.get('final_mse', float('inf')),
                    'Pert_Complexity': pert.get('total_complexity', float('inf')),
                    'Eigen_Complexity': eigen.get('total_complexity', float('inf'))
                })
    
    if not summary_data:
        return
        
    df = pd.DataFrame(summary_data)
    
    # MSE comparison plot
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.scatter(df['Pert_MSE'], df['Eigen_MSE'], alpha=0.7, s=50)
    ax1.set_xlabel('Perturbation MSE')
    ax1.set_ylabel('Eigenvalue MSE')
    ax1.set_title('MSE Comparison: Perturbation vs Eigenvalue')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.plot([df['Pert_MSE'].min(), df['Pert_MSE'].max()], 
             [df['Pert_MSE'].min(), df['Pert_MSE'].max()], 'r--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # Complexity comparison plot
    ax2 = fig.add_subplot(2, 2, 2)
    valid_complexity = (df['Pert_Complexity'] != float('inf')) & (df['Eigen_Complexity'] != float('inf'))
    if valid_complexity.any():
        ax2.scatter(df.loc[valid_complexity, 'Pert_Complexity'], 
                   df.loc[valid_complexity, 'Eigen_Complexity'], alpha=0.7, s=50)
        ax2.set_xlabel('Perturbation Complexity')
        ax2.set_ylabel('Eigenvalue Complexity')
        ax2.set_title('Formula Complexity Comparison')
        ax2.grid(True, alpha=0.3)
    
    # Best architectures table
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.axis('tight')
    ax3.axis('off')
    
    # Sort by MSE and get top 5
    df_sorted = df.nsmallest(5, 'Pert_MSE')
    table_data = df_sorted[['Architecture', 'LR', 'L05', 'Pert_MSE', 'Eigen_MSE']].round(6)
    
    table = ax3.table(cellText=table_data.values,
                     colLabels=table_data.columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)
    ax3.set_title('Top 5 Architectures (by Perturbation MSE)')
    
    # Performance heatmap
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Create pivot table for heatmap
    if len(df) > 1 and SEABORN_AVAILABLE:
        pivot_data = df.pivot_table(values='Pert_MSE', index='Architecture', columns='LR', aggfunc='mean')
        if not pivot_data.empty:
            sns.heatmap(np.log10(pivot_data), annot=True, fmt='.1f', cmap='viridis_r', ax=ax4)
            ax4.set_title('Log10(MSE) Heatmap by Architecture and LR')
            ax4.set_xlabel('Learning Rate')
            ax4.set_ylabel('Architecture')
    else:
        ax4.text(0.5, 0.5, 'Heatmap requires seaborn\nand multiple experiments', 
                ha='center', va='center', transform=ax4.transAxes)


def create_formula_table_page(all_results, fig):
    """Create a comprehensive table of all formulas discovered."""
    
    fig.clear()
    fig.suptitle('Complete Formula Summary', fontsize=16, fontweight='bold')
    
    # Collect all formulas
    formula_data = []
    
    for result in all_results:
        if 'error' in result:
            continue
            
        exp_id = result['experiment_id']
        arch = str(result['architecture'])
        
        for data_type in ['perturbation', 'eigenvalue']:
            if data_type in result and 'error' not in result[data_type]:
                data = result[data_type]
                
                # Original formulas
                formulas = data.get('formulas', {})
                refined_formulas = data.get('refined_formulas', {})
                symbolic_mse = data.get('symbolic_mse_scores', {})
                
                for output_name in ['z_0', 'z_1', 'z_2']:
                    original_formula = formulas.get(output_name)
                    refined_formula = refined_formulas.get(output_name)
                    mse_score = symbolic_mse.get(output_name, float('inf'))
                    
                    if original_formula is not None:
                        formula_data.append({
                            'Experiment': exp_id,
                            'Architecture': arch,
                            'Data_Type': data_type.capitalize(),
                            'Output': output_name,
                            'Original_Formula': str(original_formula)[:100] + ('...' if len(str(original_formula)) > 100 else ''),
                            'Refined_Formula': str(refined_formula)[:100] + ('...' if len(str(refined_formula)) > 100 else '') if refined_formula else 'None',
                            'Symbolic_MSE': safe_format_float(mse_score),
                            'Complexity': sp.count_ops(original_formula) if original_formula else 'N/A'
                        })
    
    if not formula_data:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No formulas found in experiments', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.axis('off')
        return
    
    # Create table
    df = pd.DataFrame(formula_data)
    
    # Sort by symbolic MSE (best first)
    df_sorted = df.sort_values('Symbolic_MSE')
    
    # Split into multiple tables if too many rows
    rows_per_page = 20
    num_pages = (len(df_sorted) + rows_per_page - 1) // rows_per_page
    
    for page in range(num_pages):
        start_idx = page * rows_per_page
        end_idx = min((page + 1) * rows_per_page, len(df_sorted))
        page_data = df_sorted.iloc[start_idx:end_idx]
        
        ax = fig.add_subplot(num_pages, 1, page + 1)
        ax.axis('tight')
        ax.axis('off')
        
        # Create table with appropriate columns
        display_cols = ['Experiment', 'Architecture', 'Data_Type', 'Output', 'Symbolic_MSE', 'Complexity']
        table_data = page_data[display_cols]
        
        table = ax.table(cellText=table_data.values,
                        colLabels=table_data.columns,
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(6)
        table.scale(1.2, 1.5)
        
        if page == 0:
            ax.set_title(f'Formula Results (Page {page+1}/{num_pages}) - Sorted by Symbolic MSE', pad=20)
        else:
            ax.set_title(f'Formula Results (Page {page+1}/{num_pages})', pad=20)


def create_optimization_results_table(all_results, fig):
    """Create a table showing coefficient optimization results."""

    fig.clear()
    fig.suptitle("Coefficient Optimization Results", fontsize=16, fontweight="bold")

    # Collect optimization data
    opt_data = []

    for result in all_results:
        if "error" in result:
            continue

        exp_id = result["experiment_id"]
        architecture = str(result["architecture"])

        for data_type in ["perturbation", "eigenvalue"]:
            if data_type in result and "error" not in result[data_type]:
                data = result[data_type]

                # Extract optimization results
                optimization_results = data.get("optimization_results", {})

                for output_name in ["z_0", "z_1", "z_2"]:
                    opt_result = optimization_results.get(output_name, {})

                    if opt_result and opt_result.get("optimization_success", False):
                        original_mse = opt_result.get("original_mse", float("inf"))
                        optimized_mse = opt_result.get("optimized_mse", float("inf"))
                        epochs_trained = opt_result.get("epochs_trained", 0)
                        improvement = (
                            ((original_mse - optimized_mse) / original_mse * 100)
                            if original_mse > 0
                            else 0
                        )

                        opt_data.append(
                            {
                                "Experiment": exp_id,
                                "Architecture": architecture,
                                "Data_Type": data_type.capitalize(),
                                "Output": output_name,
                                "Original_MSE": safe_format_float(original_mse),
                                "Optimized_MSE": safe_format_float(optimized_mse),
                                "Improvement_%": f"{improvement:.1f}%",
                                "Epochs": epochs_trained,
                                "Status": "Success",
                            }
                        )
                    else:
                        message = opt_result.get("message", "No optimization attempted")
                        opt_data.append(
                            {
                                "Experiment": exp_id,
                                "Architecture": architecture,
                                "Data_Type": data_type.capitalize(),
                                "Output": output_name,
                                "Original_MSE": "N/A",
                                "Optimized_MSE": "N/A",
                                "Improvement_%": "N/A",
                                "Epochs": "N/A",
                                "Status": message[:30]
                                + ("..." if len(message) > 30 else ""),
                            }
                        )

    if not opt_data:
        ax = fig.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "No optimization results found",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.axis("off")
        return

    # Create table
    df = pd.DataFrame(opt_data)

    # Split into multiple tables if too many rows
    rows_per_page = 20
    num_pages = (len(df) + rows_per_page - 1) // rows_per_page

    for page in range(num_pages):
        start_idx = page * rows_per_page
        end_idx = min((page + 1) * rows_per_page, len(df))
        page_data = df.iloc[start_idx:end_idx]

        ax = fig.add_subplot(num_pages, 1, page + 1)
        ax.axis("tight")
        ax.axis("off")

        table = ax.table(
            cellText=page_data.values,
            colLabels=page_data.columns,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.2, 1.5)

        if page == 0:
            ax.set_title(f"Optimization Results (Page {page + 1}/{num_pages})", pad=20)
        else:
            ax.set_title(f"Optimization Results (Page {page + 1}/{num_pages})", pad=20)


def create_comprehensive_summary_table(all_results, fig):
    """Create a comprehensive summary table with MSE, formulas, and complexity for each parameter set and architecture."""
    
    fig.clear()
    fig.suptitle('Comprehensive Results Summary', fontsize=16, fontweight='bold')
    
    # Collect data for summary table
    summary_data = []
    
    for result in all_results:
        if 'error' in result:
            continue
            
        exp_id = result['experiment_id']
        params_name = exp_id.split('_')[0] if '_' in exp_id else 'unknown'  # Extract parameter set name
        architecture = result['architecture']
        input_dim = f"{architecture[0]}D" if architecture else "unknownD"
        
        for data_type in ['perturbation', 'eigenvalue']:
            if data_type in result and 'error' not in result[data_type]:
                data = result[data_type]
                
                # Extract MSE
                final_mse = data.get('final_mse', float('inf'))
                
                # Extract formulas and complexities
                formulas = data.get('formulas', {})
                refined_formulas = data.get('refined_formulas', {})
                symbolic_mse_scores = data.get('symbolic_mse_scores', {})
                
                for output_name in ['z_0', 'z_1', 'z_2']:
                    original_formula = formulas.get(output_name)
                    refined_formula = refined_formulas.get(output_name)
                    symbolic_mse = symbolic_mse_scores.get(output_name, float('inf'))
                    
                    # Calculate complexities
                    orig_complexity = sp.count_ops(original_formula) if original_formula else 0
                    refined_complexity = sp.count_ops(refined_formula) if refined_formula else 0
                    
                    # Format formulas for display (truncate if too long)
                    orig_formula_str = str(original_formula)[:50] + ('...' if len(str(original_formula)) > 50 else '') if original_formula else 'None'
                    refined_formula_str = str(refined_formula)[:50] + ('...' if len(str(refined_formula)) > 50 else '') if refined_formula else 'None'
                    
                    summary_data.append({
                        'Param_Set': params_name.upper(),
                        'Input_Dim': input_dim,
                        'Architecture': str(architecture),
                        'Data_Type': data_type.capitalize(),
                        'Output': output_name,
                        'Neural_MSE': safe_format_float(final_mse),
                        'Symbolic_MSE': safe_format_float(symbolic_mse),
                        'Original_Formula': orig_formula_str,
                        'Original_Complexity': orig_complexity,
                        'Refined_Formula': refined_formula_str,
                        'Refined_Complexity': refined_complexity
                    })
    
    if not summary_data:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No results found for summary table', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.axis('off')
        return
    
    # Create DataFrame and sort by parameter set, input dimension, and MSE
    df = pd.DataFrame(summary_data)
    df_sorted = df.sort_values(['Param_Set', 'Input_Dim', 'Neural_MSE'])
    
    # Split into multiple pages if too many rows
    rows_per_page = 25
    num_pages = (len(df_sorted) + rows_per_page - 1) // rows_per_page
    
    for page in range(num_pages):
        start_idx = page * rows_per_page
        end_idx = min((page + 1) * rows_per_page, len(df_sorted))
        page_data = df_sorted.iloc[start_idx:end_idx]
        
        ax = fig.add_subplot(num_pages, 1, page + 1)
        ax.axis('tight')
        ax.axis('off')
        
        # Select columns for display
        display_cols = ['Param_Set', 'Input_Dim', 'Data_Type', 'Output', 'Neural_MSE', 
                       'Symbolic_MSE', 'Original_Complexity', 'Refined_Complexity']
        table_data = page_data[display_cols]
        
        # Create table
        table = ax.table(cellText=table_data.values,
                        colLabels=table_data.columns,
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.2, 1.8)
        
        # Color-code by parameter set
        param_colors = {'WORKING': '#FFE6E6', 'NONLINEAR': '#E6F3FF', 
                       'REALISTIC': '#E6FFE6', 'BIDIRECTIONAL': '#FFFFE6'}
        
        for i, (idx, row) in enumerate(page_data.iterrows()):
            param_set = row['Param_Set']
            if param_set in param_colors:
                for j in range(len(display_cols)):
                    table[(i+1, j)].set_facecolor(param_colors[param_set])
        
        if page == 0:
            ax.set_title(f'Summary Table (Page {page+1}/{num_pages}) - All Parameter Sets and Architectures', pad=20)
        else:
            ax.set_title(f'Summary Table (Page {page+1}/{num_pages})', pad=20)


def create_best_results_summary(all_results, fig):
    """Create a summary of best results for each parameter set and architecture combination."""
    
    fig.clear()
    fig.suptitle('Best Results Summary by Parameter Set and Architecture', fontsize=16, fontweight='bold')
    
    # Group results by parameter set and input dimension
    grouped_results = {}
    
    for result in all_results:
        if 'error' in result:
            continue
            
        exp_id = result['experiment_id']
        params_name = exp_id.split('_')[0] if '_' in exp_id else 'unknown'
        architecture = result['architecture']
        input_dim = f"{architecture[0]}D" if architecture else "unknownD"
        
        key = f"{params_name}_{input_dim}"
        if key not in grouped_results:
            grouped_results[key] = []
        grouped_results[key].append(result)
    
    # Find best results for each group
    best_results_data = []
    
    for group_key, group_results in grouped_results.items():
        params_name, input_dim = group_key.split('_')
        
        for data_type in ['perturbation', 'eigenvalue']:
            # Find best result by Neural MSE
            valid_results = [r for r in group_results 
                           if data_type in r and 'error' not in r[data_type]]
            
            if valid_results:
                best_result = min(valid_results, 
                                key=lambda x: x[data_type].get('final_mse', float('inf')))
                
                data = best_result[data_type]
                final_mse = data.get('final_mse', float('inf'))
                total_complexity_orig = data.get('total_complexity', 0)
                
                # Calculate total refined complexity
                refined_formulas = data.get('refined_formulas', {})
                total_complexity_refined = sum(sp.count_ops(f) for f in refined_formulas.values() if f is not None)
                
                # Get best symbolic MSE
                symbolic_mses = data.get('symbolic_mse_scores', {})
                best_symbolic_mse = min(symbolic_mses.values()) if symbolic_mses else float('inf')
                
                # Count successful formulas
                successful_formulas = sum(1 for f in data.get('formulas', {}).values() if f is not None)
                successful_refined = sum(1 for f in refined_formulas.values() if f is not None)
                
                best_results_data.append({
                    'Parameter_Set': params_name.upper(),
                    'Input_Dim': input_dim,
                    'Data_Type': data_type.capitalize(),
                    'Architecture': str(best_result['architecture']),
                    'Neural_MSE': safe_format_float(final_mse),
                    'Best_Symbolic_MSE': safe_format_float(best_symbolic_mse),
                    'Formulas_Found': f"{successful_formulas}/3",
                    'Refined_Formulas': f"{successful_refined}/3",
                    'Original_Complexity': total_complexity_orig,
                    'Refined_Complexity': total_complexity_refined,
                    'Experiment_ID': best_result['experiment_id']
                })
    
    if not best_results_data:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No best results found', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.axis('off')
        return
    
    # Create DataFrame and sort
    df = pd.DataFrame(best_results_data)
    df_sorted = df.sort_values(['Parameter_Set', 'Input_Dim', 'Data_Type'])
    
    # Create table
    ax = fig.add_subplot(111)
    ax.axis('tight')
    ax.axis('off')
    
    # Select columns for display
    display_cols = ['Parameter_Set', 'Input_Dim', 'Data_Type', 'Neural_MSE', 
                   'Best_Symbolic_MSE', 'Formulas_Found', 'Original_Complexity', 'Refined_Complexity']
    table_data = df_sorted[display_cols]
    
    # Create table
    table = ax.table(cellText=table_data.values,
                    colLabels=table_data.columns,
                    cellLoc='center',
                    loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2.0)
    
    # Color-code by parameter set
    param_colors = {'WORKING': '#FFE6E6', 'NONLINEAR': '#E6F3FF', 
                   'REALISTIC': '#E6FFE6', 'BIDIRECTIONAL': '#FFFFE6'}
    
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        param_set = row['Parameter_Set']
        if param_set in param_colors:
            for j in range(len(display_cols)):
                table[(i+1, j)].set_facecolor(param_colors[param_set])
    
    ax.set_title('Best Performance for Each Parameter Set and Architecture Combination', pad=20)


def generate_pdf_report(all_results: List[Dict[str, Any]], all_datasets: Dict, config, output_dir: Path, param_names: List[str]) -> None:
    """Generate comprehensive PDF report."""
    
    print("\nGenerating PDF report...")
    pdf_path = output_dir / 'experiment_report.pdf'
    
    with PdfPages(pdf_path) as pdf:
        # Title page
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('KAN NMR Spectroscopy Experiments Report', fontsize=16, fontweight='bold')
        
        ax = fig.add_subplot(111)
        ax.axis('off')

        # Detect experiment type
        experiment_type = (
            "Regular"
            if any("3d" in all_datasets.get(pname, {}) for pname in param_names)
            else "Dimensionless"
        )

        report_text = f"""
        Comprehensive KAN Experiments for NMR Spectroscopy ({experiment_type})
        
        Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}
        
        Experiment Type: {experiment_type}
        {"• Regular experiments use physical variables (deltaJ, deltaJ/Jintra, Jintra, Jintra/deltaJ)" if experiment_type == "Regular" else "• Dimensionless experiments use ratios (dJ/J, J/dJ) and predict Z/J"}
        
        Experiment Configuration:
        • Parameter sets tested: {len(param_names)} ({param_names})
        • Architectures tested: {len(config.architectures)} ({config.architectures})
        • Learning rates: {config.learning_rates}
        • L05 penalties: {config.l05_penalties}
        • Training epochs: {config.epochs}
        • Number of samples: {config.num_samples}
        • Device: {config.device}
        
        Parameter Set Ranges:
        """
        
        for pname in param_names:
            pparams = config.data_params[pname]
            report_text += f"""
        • {pname.upper()}:
          - Jintra: [{pparams["MIN_JINTRA"]}, {pparams["MAX_JINTRA"]}]
          - deltaJ: [{pparams["MIN_DELTAJ"]}, {pparams["MAX_DELTAJ"]}]
        """
        
        total_experiments = len(param_names) * len(config.architectures) * len(config.learning_rates) * len(config.l05_penalties)
        report_text += f"""
        Total experiments completed: {len([r for r in all_results if 'error' not in r])} / {total_experiments}
        """
        
        ax.text(0.1, 0.9, report_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Data visualization pages for each parameter set
        for params_name in param_names:
            # Handle both regular experiments (3d, 4d) and dimensionless experiments (1d, 2d)
            if "3d" in all_datasets[params_name]:
                # Regular experiments
                datasets_3d = all_datasets[params_name]["3d"]
                datasets_4d = all_datasets[params_name]["4d"]

                if datasets_3d:
                    fig = plt.figure(figsize=(15, 10))
                    fig.suptitle(
                        f"3D Data Surfaces - {params_name.upper()} Parameter Set - 3 Input Architecture",
                        fontsize=14,
                        fontweight="bold",
                    )
                    create_surface_plots(datasets_3d["pert"], datasets_3d["eigen"], fig)
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)

                if datasets_4d:
                    fig = plt.figure(figsize=(15, 10))
                    fig.suptitle(
                        f"3D Data Surfaces - {params_name.upper()} Parameter Set - 4 Input Architecture",
                        fontsize=14,
                        fontweight="bold",
                    )
                    create_surface_plots(datasets_4d["pert"], datasets_4d["eigen"], fig)
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)

            elif "1d" in all_datasets[params_name]:
                # Dimensionless experiments
                datasets_1d = all_datasets[params_name]["1d"]
                datasets_2d = all_datasets[params_name]["2d"]

                if datasets_1d:
                    fig = plt.figure(figsize=(15, 10))
                    fig.suptitle(
                        f"Dimensionless Data Plots - {params_name.upper()} Parameter Set - 1 Input Architecture",
                        fontsize=14,
                        fontweight="bold",
                    )
                    create_dimensionless_plots(
                        datasets_1d["pert"], datasets_1d["eigen"], fig, input_dim=1
                    )
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)

                if datasets_2d:
                    fig = plt.figure(figsize=(15, 10))
                    fig.suptitle(
                        f"Dimensionless Data Plots - {params_name.upper()} Parameter Set - 2 Input Architecture",
                        fontsize=14,
                        fontweight="bold",
                    )
                    create_dimensionless_plots(
                        datasets_2d["pert"], datasets_2d["eigen"], fig, input_dim=2
                    )
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
                plt.close(fig)
        
        # Summary page
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle('Experiment Summary and Comparison', fontsize=14, fontweight='bold')
        create_experiment_summary_page(all_results, fig)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Formula table page
        fig = plt.figure(figsize=(15, 12))
        create_formula_table_page(all_results, fig)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Coefficient optimization results
        fig = plt.figure(figsize=(18, 12))
        create_optimization_results_table(all_results, fig)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Comprehensive summary table
        fig = plt.figure(figsize=(20, 15))
        create_comprehensive_summary_table(all_results, fig)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Best results summary
        fig = plt.figure(figsize=(15, 10))
        create_best_results_summary(all_results, fig)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Individual experiment pages
        for result in all_results:
            if 'error' in result:
                continue
                
            fig = plt.figure(figsize=(15, 10))
            fig.suptitle(f"Experiment: {result['experiment_id']}", fontsize=14, fontweight='bold')
            
            # Plot training curves
            plot_training_curves(result, fig, 1)
            
            # Add formula text
            ax_text = fig.add_subplot(2, 3, 4)
            ax_text.axis('off')
            
            formula_text = f"Architecture: {result['architecture']}\n"
            formula_text += f"LR: {result['lr']}, L05: {result['l05_penalty']}\n\n"
            
            if 'perturbation' in result and 'error' not in result['perturbation']:
                formula_text += "Perturbation Formulas:\n"
                for name, formula in result['perturbation'].get('formulas', {}).items():
                    if formula:
                        formula_text += f"{name}: {str(formula)[:60]}...\n"
                        
                formula_text += f"\nPert MSE: {safe_format_float(result['perturbation'].get('final_mse'))}\n\n"
            
            if 'eigenvalue' in result and 'error' not in result['eigenvalue']:
                formula_text += "Eigenvalue Formulas:\n"
                for name, formula in result['eigenvalue'].get('formulas', {}).items():
                    if formula:
                        formula_text += f"{name}: {str(formula)[:60]}...\n"
                        
                formula_text += f"\nEigen MSE: {safe_format_float(result['eigenvalue'].get('final_mse'))}\n"
            
            ax_text.text(0.05, 0.95, formula_text, transform=ax_text.transAxes,
                        fontsize=8, verticalalignment='top', fontfamily='monospace')
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
    print(f"PDF report saved to: {pdf_path}")

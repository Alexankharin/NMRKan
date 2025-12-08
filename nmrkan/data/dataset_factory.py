"""Dataset factory for NMR KAN experiments."""

import pickle
import warnings
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

from ..config import DataConfig

# Import data generation functions (existing)
try:
    from data_generation.nmr_datagen import get_frequences_ordered, get_frequences
    from data_generation.eigenvalues_4CH2 import CalcEigenEnergies
    from data_generation import (
        make_dataset_from_function,
        make_dimensionless_dataset_from_function,
        make_normalized_eigenvalue_dataset,
        get_eigenvalue_dimensionless,
        get_perturbation
    )
except ImportError as e:
    warnings.warn(f"Could not import data generation functions: {e}")

warnings.filterwarnings('ignore', category=UserWarning)


class DatasetFactory:
    """Factory class for creating NMR datasets."""
    
    def __init__(self, config: DataConfig):
        """Initialize dataset factory with configuration.
        
        Args:
            config: Data configuration parameters
        """
        self.config = config
        
        # Ensure cache directory exists
        if self.config.cache_dataset and self.config.cache_dir:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def create_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create dataset based on configuration.
        
        Returns:
            Tuple of (input_tensor, output_tensor)
        """
        # Check cache first
        if self.config.cache_dataset:
            cached_data = self._load_from_cache()
            if cached_data is not None:
                print(f"Loaded dataset from cache: {self.config.get_cache_key()}")
                return cached_data
        
        # Generate new dataset
        print(f"Generating new dataset: {self.config.dataset_type}")
        
        if self.config.dataset_type == "4ch2":
            inputs, outputs = self._create_4ch2_dataset()
        elif self.config.dataset_type == "nmr_datagen":
            inputs, outputs = self._create_nmr_datagen_dataset()
        else:
            raise ValueError(f"Unsupported dataset type: {self.config.dataset_type}")
        
        # Convert to tensors
        input_tensor = torch.tensor(inputs, dtype=torch.float32)
        output_tensor = torch.tensor(outputs, dtype=torch.float32)
        
        # Apply normalization if configured
        if self.config.normalize_inputs:
            input_tensor = self._normalize_inputs(input_tensor)
        
        if self.config.normalize_outputs:
            output_tensor = self._normalize_outputs(output_tensor)
        
        # Cache the dataset
        if self.config.cache_dataset:
            self._save_to_cache((input_tensor, output_tensor))
        
        return input_tensor, output_tensor
    
    def _create_4ch2_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create dataset using 4CH2 eigenvalue method."""
        if self.config.output_type == "eigenvalues" and self.config.num_outputs == 8:
            # Normalized eigenvalue dataset
            return self._create_normalized_eigenvalue_dataset()
        else:
            # Frequency dataset
            return self._create_4ch2_frequency_dataset()
    
    def _create_4ch2_frequency_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create frequency dataset using 4CH2 method."""
        # Generate parameter samples
        inputs = self._generate_parameter_samples()
        
        outputs = []
        valid_inputs = []
        
        for i, params in enumerate(inputs):
            try:
                jintra, delta_j = params[0], params[1]
                
                if self.config.dimensionless:
                    # Use dimensionless approach
                    ratio = delta_j / jintra if jintra != 0 else 0
                    freqs = get_eigenvalue_dimensionless(ratio)
                    valid_inputs.append([ratio])
                else:
                    # Regular approach - use CalcEigenEnergies from 4CH2 module
                    eigenvalues = CalcEigenEnergies(jintra, delta_j)
                    # Calculate frequency differences (transitions)
                    sorted_eigs = np.sort(eigenvalues)
                    # Use the first few eigenvalue differences as frequencies
                    freqs = np.diff(sorted_eigs)[:self.config.num_outputs]
                    valid_inputs.append(params)
                
                outputs.append(freqs[:self.config.num_outputs])
                
            except Exception as e:
                warnings.warn(f"Failed to compute frequencies for params {params}: {e}")
                continue
        
        return np.array(valid_inputs), np.array(outputs)
    
    def _create_normalized_eigenvalue_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create normalized eigenvalue dataset."""
        try:
            # Extract parameter ranges
            jintra_range = self.config.parameter_ranges.get('Jintra', (0.1, 10.0))
            delta_j_range = self.config.parameter_ranges.get('deltaJ', (0.01, 1.0))
            
            inputs, outputs = make_normalized_eigenvalue_dataset(
                num_samples=self.config.num_samples,
                jintra_range=jintra_range,
                delta_j_range=delta_j_range
            )
            
            return inputs, outputs
            
        except Exception as e:
            raise RuntimeError(f"Failed to create normalized eigenvalue dataset: {e}")
    
    def _create_nmr_datagen_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create dataset using NMR datagen method."""
        # Generate parameter samples
        inputs = self._generate_parameter_samples()
        
        outputs = []
        valid_inputs = []
        
        for params in inputs:
            try:
                jintra, delta_j = params[0], params[1]
                
                if self.config.dimensionless:
                    # Use dimensionless ratio
                    ratio = delta_j / jintra if jintra != 0 else 0
                    # Use get_eigenvalue_dimensionless for consistency
                    try:
                        freqs = get_eigenvalue_dimensionless(ratio)
                    except:
                        # Fallback to NMR datagen method
                        freq_inputs = np.array([[jintra, delta_j]])
                        freqs = get_frequences_ordered(freq_inputs)[0] / jintra if jintra != 0 else get_frequences_ordered(freq_inputs)[0]
                    valid_inputs.append([ratio])
                else:
                    # Regular frequency calculation
                    freq_inputs = np.array([[jintra, delta_j]]) 
                    freqs = get_frequences_ordered(freq_inputs)[0]
                    valid_inputs.append(params)
                
                outputs.append(freqs[:self.config.num_outputs])
                
            except Exception as e:
                warnings.warn(f"Failed to compute frequencies for params {params}: {e}")
                continue
        
        return np.array(valid_inputs), np.array(outputs)
    
    def _generate_parameter_samples(self) -> np.ndarray:
        """Generate random parameter samples based on configuration."""
        samples = []
        
        # Extract parameter ranges
        jintra_range = self.config.parameter_ranges.get('Jintra', (0.1, 10.0))
        delta_j_range = self.config.parameter_ranges.get('deltaJ', (0.01, 1.0))
        
        # Generate uniform random samples
        np.random.seed(42)  # For reproducibility
        jintra_samples = np.random.uniform(
            jintra_range[0], jintra_range[1], self.config.num_samples
        )
        delta_j_samples = np.random.uniform(
            delta_j_range[0], delta_j_range[1], self.config.num_samples  
        )
        
        samples = np.column_stack([jintra_samples, delta_j_samples])
        
        return samples
    
    def _normalize_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Normalize input tensor."""
        # Simple min-max normalization
        min_vals, _ = torch.min(inputs, dim=0, keepdim=True)
        max_vals, _ = torch.max(inputs, dim=0, keepdim=True)
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        
        normalized = (inputs - min_vals) / range_vals
        return normalized
    
    def _normalize_outputs(self, outputs: torch.Tensor) -> torch.Tensor:
        """Normalize output tensor."""
        # Simple min-max normalization
        min_vals, _ = torch.min(outputs, dim=0, keepdim=True)
        max_vals, _ = torch.max(outputs, dim=0, keepdim=True)
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        
        normalized = (outputs - min_vals) / range_vals
        return normalized
    
    def _load_from_cache(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Load dataset from cache if available."""
        cache_key = self.config.get_cache_key()
        cache_file = self.config.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                warnings.warn(f"Failed to load cache file {cache_file}: {e}")
        
        return None
    
    def _save_to_cache(self, data: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Save dataset to cache."""
        cache_key = self.config.get_cache_key()
        cache_file = self.config.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"Saved dataset to cache: {cache_file}")
        except Exception as e:
            warnings.warn(f"Failed to save cache file {cache_file}: {e}")
    
    def create_dataloaders(
        self, 
        batch_size: int,
        validation_split: float = 0.2,
        shuffle: bool = True
    ) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation dataloaders.
        
        Args:
            batch_size: Batch size for dataloaders
            validation_split: Fraction of data for validation
            shuffle: Whether to shuffle the data
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Create dataset
        inputs, outputs = self.create_dataset()
        dataset = TensorDataset(inputs, outputs)
        
        # Split into train/val
        total_size = len(dataset)
        val_size = int(validation_split * total_size)
        train_size = total_size - val_size
        
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=self.config.drop_last,
            num_workers=self.config.num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.config.num_workers
        )
        
        return train_loader, val_loader
"""Data loading utilities."""

import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional


def create_dataloader(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    batch_size: int = 1000,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """Create a PyTorch DataLoader from input and output tensors.
    
    Args:
        inputs: Input tensor
        outputs: Output tensor
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data
        drop_last: Whether to drop the last incomplete batch
        num_workers: Number of worker processes for data loading
        
    Returns:
        PyTorch DataLoader
    """
    dataset = TensorDataset(inputs, outputs)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )


def create_train_val_loaders(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    validation_split: float = 0.2,
    batch_size: int = 1000,
    shuffle: bool = True,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders with automatic splitting.
    
    Args:
        inputs: Input tensor
        outputs: Output tensor
        validation_split: Fraction of data to use for validation
        batch_size: Batch size for DataLoaders
        shuffle: Whether to shuffle the training data
        random_seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    from torch.utils.data import random_split
    
    # Create dataset
    dataset = TensorDataset(inputs, outputs)
    
    # Calculate split sizes
    total_size = len(dataset)
    val_size = int(validation_split * total_size)
    train_size = total_size - val_size
    
    # Split dataset
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    
    return train_loader, val_loader
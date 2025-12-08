"""Checkpoint utilities for model saving and loading."""

import warnings
from pathlib import Path
from typing import Dict, Any, Union, Optional

import torch
import torch.nn as nn

warnings.filterwarnings('ignore', category=UserWarning)


def save_checkpoint(
    checkpoint_data: Dict[str, Any],
    checkpoint_path: Union[str, Path]
) -> None:
    """Save model checkpoint to file.
    
    Args:
        checkpoint_data: Dictionary containing model state and metadata
        checkpoint_path: Path to save the checkpoint
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        torch.save(checkpoint_data, checkpoint_path)
        # print(f"Checkpoint saved: {checkpoint_path}")
    except Exception as e:
        warnings.warn(f"Failed to save checkpoint {checkpoint_path}: {e}")


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Load model checkpoint from file.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the checkpoint to
        
    Returns:
        Dictionary containing checkpoint data, or None if loading fails
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        warnings.warn(f"Checkpoint file not found: {checkpoint_path}")
        return None
    
    try:
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint
        
    except Exception as e:
        warnings.warn(f"Failed to load checkpoint {checkpoint_path}: {e}")
        return None


def load_model_from_checkpoint(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    device: Optional[str] = None,
    strict: bool = True
) -> Optional[nn.Module]:
    """Load model state from checkpoint file.
    
    Args:
        model: Model instance to load state into
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model to
        strict: Whether to strictly enforce state dict keys match
        
    Returns:
        Model with loaded state, or None if loading fails
    """
    checkpoint = load_checkpoint(checkpoint_path, device)
    
    if checkpoint is None:
        return None
    
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        if device is not None:
            model = model.to(device)
        
        print(f"Model state loaded from: {checkpoint_path}")
        return model
        
    except Exception as e:
        warnings.warn(f"Failed to load model state from {checkpoint_path}: {e}")
        return None


def get_latest_checkpoint(
    checkpoint_dir: Union[str, Path],
    pattern: str = "checkpoint_epoch_*.pt"
) -> Optional[Path]:
    """Get the latest checkpoint file from a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        pattern: Glob pattern for checkpoint files
        
    Returns:
        Path to the latest checkpoint file, or None if none found
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    checkpoint_files = list(checkpoint_dir.glob(pattern))
    
    if not checkpoint_files:
        return None
    
    # Sort by modification time and return the latest
    latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
    
    return latest_checkpoint


def get_best_checkpoint(
    checkpoint_dir: Union[str, Path],
    pattern: str = "*_best.pt"
) -> Optional[Path]:
    """Get the best checkpoint file from a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        pattern: Glob pattern for best checkpoint files
        
    Returns:
        Path to the best checkpoint file, or None if none found
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    best_files = list(checkpoint_dir.glob(pattern))
    
    if not best_files:
        return None
    
    # If multiple best files, return the one with lowest loss
    if len(best_files) == 1:
        return best_files[0]
    
    best_checkpoint = None
    best_loss = float('inf')
    
    for checkpoint_file in best_files:
        try:
            checkpoint = load_checkpoint(checkpoint_file)
            if checkpoint and checkpoint.get('loss', float('inf')) < best_loss:
                best_loss = checkpoint['loss']
                best_checkpoint = checkpoint_file
        except Exception:
            continue
    
    return best_checkpoint


def cleanup_old_checkpoints(
    checkpoint_dir: Union[str, Path],
    keep_last_n: int = 5,
    pattern: str = "checkpoint_epoch_*.pt"
) -> None:
    """Remove old checkpoint files, keeping only the most recent.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        keep_last_n: Number of recent checkpoints to keep
        pattern: Glob pattern for checkpoint files to cleanup
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return
    
    checkpoint_files = list(checkpoint_dir.glob(pattern))
    
    if len(checkpoint_files) <= keep_last_n:
        return
    
    # Sort by modification time (newest first)
    checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    # Remove old checkpoints
    for old_checkpoint in checkpoint_files[keep_last_n:]:
        try:
            old_checkpoint.unlink()
            # print(f"Removed old checkpoint: {old_checkpoint}")
        except Exception as e:
            warnings.warn(f"Failed to remove checkpoint {old_checkpoint}: {e}")
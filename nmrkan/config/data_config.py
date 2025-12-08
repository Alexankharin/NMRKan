"""Data configuration class."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for dataset generation and loading."""
    
    # Dataset type
    dataset_type: str = "4ch2"  # '4ch2' or 'nmr_datagen'
    
    # Data generation parameters
    num_samples: int = 10000
    parameter_ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'Jintra': (0.1, 10.0),
        'deltaJ': (0.01, 1.0) 
    })
    
    # Data processing
    normalize_inputs: bool = True
    normalize_outputs: bool = False
    dimensionless: bool = False
    
    # Output configuration
    output_type: str = "frequencies"  # 'frequencies' or 'eigenvalues'
    num_outputs: int = 3  # 3 for frequencies, 8 for normalized eigenvalues
    
    # Data loading
    shuffle: bool = True
    drop_last: bool = True
    num_workers: int = 0
    
    # Caching
    cache_dataset: bool = True
    cache_dir: Optional[Path] = None
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.dataset_type not in ['4ch2', 'nmr_datagen']:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
        
        if self.output_type not in ['frequencies', 'eigenvalues']:
            raise ValueError(f"Unsupported output type: {self.output_type}")
        
        if self.num_samples <= 0:
            raise ValueError("Number of samples must be positive")
        
        # Validate parameter ranges
        for param, (min_val, max_val) in self.parameter_ranges.items():
            if min_val >= max_val:
                raise ValueError(f"Invalid range for {param}: min must be less than max")
        
        # Set default cache directory
        if self.cache_dir is None:
            self.cache_dir = Path("data/cache")
    
    def get_cache_key(self) -> str:
        """Generate unique cache key for this data configuration."""
        key_parts = [
            f"type_{self.dataset_type}",
            f"samples_{self.num_samples}",
            f"outputs_{self.output_type}_{self.num_outputs}",
            f"norm_in_{self.normalize_inputs}",
            f"norm_out_{self.normalize_outputs}",
            f"dimensionless_{self.dimensionless}"
        ]
        
        # Add parameter ranges to key
        for param, (min_val, max_val) in sorted(self.parameter_ranges.items()):
            key_parts.append(f"{param}_{min_val}_{max_val}")
        
        return "_".join(key_parts)
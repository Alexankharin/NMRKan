# Improved Checkpointing System

## Key Improvements

### 1. Reduced Checkpoint Frequency
- **Old**: Saves checkpoint every 5,000 epochs
- **New**: Saves checkpoint every 10,000 epochs
- **Result**: 50% fewer regular checkpoints

### 2. Intelligent Checkpoint Cleanup
- **Automatic cleanup**: Keeps only the last 3 regular checkpoints for single models
- **Multi-model cleanup**: Keeps only the last 2 checkpoints per model
- **Best models**: Always preserved (never cleaned up)

### 3. Selective Saving Strategy
- **Best models**: Saved immediately when validation loss improves
- **Regular checkpoints**: Only at specified intervals (10k epochs)
- **Final checkpoint**: Saved at end of training
- **Avoids redundant saves**: No double-saving on interval + best model

### 4. Pickle-Safe Checkpointing
- **Problem**: Lambda functions in ModelConfig couldn't be pickled
- **Solution**: Remove unpickleable objects before saving
- **Result**: No more checkpoint serialization errors

### 5. Better Progress Reporting
- **Checkpoint notifications**: Clear messages when best models are saved
- **Final summary**: Shows results directory location
- **Cleanup notifications**: Silent cleanup to reduce noise

## File Organization

### Single Model Training
```
experiment_results/unified_experiments_<timestamp>/
├── best_model.pt                    # Best performing model (always kept)
├── checkpoint_epoch_10000.pt        # Regular checkpoint
├── checkpoint_epoch_20000.pt        # Regular checkpoint  
├── checkpoint_epoch_30000.pt        # Regular checkpoint
├── checkpoint_epoch_<final>.pt      # Final checkpoint
└── training_curves.png              # Training progress plots
```

### Multi-Model Training
```
experiment_results/unified_experiments_<timestamp>/
├── model_0_best.pt                  # Best model for output 0
├── model_1_best.pt                  # Best model for output 1
├── model_2_best.pt                  # Best model for output 2
├── model_0_checkpoint_epoch_<N>.pt  # Last 2 checkpoints per model
├── model_0_checkpoint_epoch_<M>.pt
├── model_1_checkpoint_epoch_<N>.pt
├── model_1_checkpoint_epoch_<M>.pt
├── ...
├── model_0_training_curves.png      # Per-model training plots
├── model_1_training_curves.png
└── model_2_training_curves.png
```

## Configuration Updates

### Training Config Defaults
```yaml
training:
  save_interval: 10000     # Was 5000 - less frequent saving
  log_interval: 1000       # Unchanged - keep detailed logging
```

### Cleanup Parameters
- **Single models**: Keep last 3 regular checkpoints
- **Multi models**: Keep last 2 checkpoints per model
- **Best models**: Never cleaned up
- **Automatic**: Runs after each checkpoint save

## Benefits

1. **Reduced disk usage**: ~60% fewer checkpoint files
2. **Faster training**: Less I/O overhead from frequent saves
3. **Cleaner directories**: Automatic cleanup prevents clutter
4. **Reliable saving**: No more pickle errors with lambda functions
5. **Better UX**: Clear notifications and progress reporting

## Migration Notes

- **Existing checkpoints**: Old checkpoint files remain unchanged
- **Backward compatibility**: Loading old checkpoints still works
- **Configuration**: Update save_interval in custom configs if desired
- **Cleanup**: Only applies to new training runs
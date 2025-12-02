# ConvLSTM Documentation Index

This document provides an index of all documentation for the ConvLSTM weather prediction system.

## Quick Start

- **Module README**: `convlstm/README.md` - Start here for overview and basic usage
- **Multi-Variable Guide**: `convlstm/MULTI_VARIABLE_GUIDE.md` - Comprehensive guide for multi-variable prediction
- **Training Guide**: `convlstm/TRAINING_GUIDE.md` - Detailed training documentation

## Core Documentation

### Getting Started

1. **README.md** (Main Project)
   - Overview of GraphCast, GenCast, and ConvLSTM
   - Quick start for all modules
   - Installation instructions

2. **convlstm/README.md** (Module README)
   - ConvLSTM module overview
   - Architecture details
   - Data processing
   - Training and inference basics
   - Multi-variable prediction overview

### Multi-Variable Prediction

3. **convlstm/MULTI_VARIABLE_GUIDE.md** (Comprehensive Guide)
   - Why multi-variable prediction?
   - Configuration parameters
   - Training workflows
   - Rolling forecasts
   - Evaluation
   - Troubleshooting
   - Example workflows

4. **convlstm/CONFIGURATION.md** (Parameter Reference)
   - Complete parameter reference
   - Multi-variable parameters
   - Model architecture options
   - Training hyperparameters
   - Memory optimization
   - Example configurations

5. **convlstm/MULTI_VARIABLE_EVALUATION.md** (Evaluation Guide)
   - Evaluation metrics
   - Per-variable analysis
   - Rolling forecast evaluation
   - Visualization
   - Comparison workflows

### Training and Inference

6. **convlstm/TRAINING_GUIDE.md**
   - Detailed training procedures
   - Hyperparameter tuning
   - Memory optimization
   - Checkpoint management
   - Troubleshooting

7. **convlstm/INFERENCE_GUIDE.md**
   - Inference procedures
   - Batch processing
   - Visualization
   - Performance optimization

8. **convlstm/INFERENCE_CLI_GUIDE.md**
   - Command-line interface reference
   - Inference parameters
   - Output formats

### Architecture and Design

9. **docs/ConvLSTM_model.md**
   - Model architecture details
   - ConvLSTM cell implementation
   - U-Net structure
   - Memory optimization strategies

10. **docs/Rolling_forecast.md**
    - Rolling forecast theory
    - Autoregressive prediction
    - Implementation details
    - Error accumulation

11. **docs/Rolling_forecast_inference.md**
    - Rolling forecast inference
    - Multi-step prediction
    - Performance considerations

### Evaluation and Analysis

12. **convlstm/EVALUATION_GUIDE.md**
    - General evaluation procedures
    - Metrics computation
    - Comparative analysis

13. **convlstm/SCRIPTS_OVERVIEW.md**
    - Overview of all scripts
    - Usage examples
    - Workflow integration

### Specialized Topics

14. **convlstm/MPS_GUIDE.md**
    - Apple Silicon (M1/M2/M3) GPU acceleration
    - MPS-specific configuration
    - Performance tips

15. **docs/GPU_optimization_guide.md**
    - GPU optimization strategies
    - Memory management
    - Performance tuning

## Example Scripts

### Training Examples

- **examples/train_convlstm_example.sh**
  - Single-variable training examples
  - Various configurations
  - Memory optimization
  - Device-specific examples

- **examples/train_multi_variable.sh**
  - Multi-variable training examples
  - Rolling forecast training
  - Loss weighting configurations
  - Model variant examples

### Inference Examples

- **examples/rolling_forecast.sh**
  - Rolling forecast examples
  - Multi-step predictions
  - Batch processing
  - Evaluation workflows

### Evaluation Examples

- **examples/evaluate_multi_variable.py**
  - Multi-variable evaluation script
  - Per-variable metrics
  - Rolling forecast evaluation
  - Visualization generation

## Specification Documents

### Requirements and Design

- **.kiro/specs/multi-variable-rolling-forecast/requirements.md**
  - Formal requirements (EARS format)
  - Acceptance criteria
  - Glossary

- **.kiro/specs/multi-variable-rolling-forecast/design.md**
  - System design
  - Architecture decisions
  - Correctness properties
  - Testing strategy

- **.kiro/specs/multi-variable-rolling-forecast/tasks.md**
  - Implementation tasks
  - Task dependencies
  - Progress tracking

## Documentation by Use Case

### I want to train a model

1. Start with: `convlstm/README.md` (Quick Start section)
2. For single-variable: `examples/train_convlstm_example.sh`
3. For multi-variable: `convlstm/MULTI_VARIABLE_GUIDE.md` + `examples/train_multi_variable.sh`
4. For detailed options: `convlstm/TRAINING_GUIDE.md` + `convlstm/CONFIGURATION.md`

### I want to run inference

1. Start with: `convlstm/README.md` (Inference section)
2. For single-step: `convlstm/INFERENCE_GUIDE.md`
3. For rolling forecasts: `convlstm/MULTI_VARIABLE_GUIDE.md` (Rolling Forecasts section) + `examples/rolling_forecast.sh`
4. For CLI reference: `convlstm/INFERENCE_CLI_GUIDE.md`

### I want to evaluate predictions

1. Start with: `convlstm/EVALUATION_GUIDE.md`
2. For multi-variable: `convlstm/MULTI_VARIABLE_EVALUATION.md`
3. For examples: `examples/evaluate_multi_variable.py`

### I want to understand multi-variable prediction

1. Start with: `convlstm/MULTI_VARIABLE_GUIDE.md` (Overview section)
2. For configuration: `convlstm/CONFIGURATION.md` (Multi-Variable Parameters)
3. For theory: `docs/Rolling_forecast.md`
4. For design: `.kiro/specs/multi-variable-rolling-forecast/design.md`

### I want to optimize memory usage

1. Start with: `convlstm/TRAINING_GUIDE.md` (Memory Optimization section)
2. For GPU: `docs/GPU_optimization_guide.md`
3. For Apple Silicon: `convlstm/MPS_GUIDE.md`
4. For examples: `examples/train_multi_variable.sh` (Example 6)

### I'm having issues

1. Start with: `convlstm/MULTI_VARIABLE_GUIDE.md` (Troubleshooting section)
2. For training issues: `convlstm/TRAINING_GUIDE.md` (Troubleshooting)
3. For inference issues: `convlstm/INFERENCE_GUIDE.md` (Troubleshooting)
4. Check logs: `checkpoints/*/training.log`

## Documentation by Topic

### Configuration
- `convlstm/CONFIGURATION.md` - Complete parameter reference
- `convlstm/MULTI_VARIABLE_GUIDE.md` - Multi-variable configuration
- `examples/train_multi_variable.sh` - Configuration examples

### Architecture
- `docs/ConvLSTM_model.md` - Model architecture
- `convlstm/README.md` - Architecture overview
- `.kiro/specs/multi-variable-rolling-forecast/design.md` - Design decisions

### Training
- `convlstm/TRAINING_GUIDE.md` - Comprehensive training guide
- `convlstm/MULTI_VARIABLE_GUIDE.md` - Multi-variable training
- `examples/train_multi_variable.sh` - Training examples

### Inference
- `convlstm/INFERENCE_GUIDE.md` - Inference procedures
- `convlstm/INFERENCE_CLI_GUIDE.md` - CLI reference
- `examples/rolling_forecast.sh` - Rolling forecast examples

### Evaluation
- `convlstm/EVALUATION_GUIDE.md` - General evaluation
- `convlstm/MULTI_VARIABLE_EVALUATION.md` - Multi-variable evaluation
- `examples/evaluate_multi_variable.py` - Evaluation script

### Rolling Forecasts
- `docs/Rolling_forecast.md` - Theory and implementation
- `docs/Rolling_forecast_inference.md` - Inference procedures
- `convlstm/MULTI_VARIABLE_GUIDE.md` - Rolling forecast guide
- `examples/rolling_forecast.sh` - Examples

### Optimization
- `docs/GPU_optimization_guide.md` - GPU optimization
- `convlstm/MPS_GUIDE.md` - Apple Silicon optimization
- `convlstm/TRAINING_GUIDE.md` - Memory optimization

## Quick Reference

### Command-Line Arguments

See `convlstm/CONFIGURATION.md` for complete reference.

Key multi-variable arguments:
- `--multi-variable`: Enable multi-variable mode
- `--precip-loss-weight`: Precipitation loss weight (default: 10.0)
- `--max-rollout-steps`: Maximum rolling steps (default: 6)
- `--enable-rollout-training`: Enable rolling training

### File Locations

- **Checkpoints**: `checkpoints/*/best_model.pt`
- **Normalizers**: `checkpoints/*/normalizer.pkl`
- **Logs**: `checkpoints/*/training.log`
- **Predictions**: `predictions/*/predictions.nc`
- **Visualizations**: `predictions/*/prediction_map_*.png`
- **Metrics**: `predictions/*/metrics/*.csv`

### Common Workflows

1. **Train → Infer → Evaluate**
   ```bash
   python train_convlstm.py --multi-variable --data data.nc --output-dir checkpoints/mv
   python run_inference_convlstm.py --checkpoint checkpoints/mv/best_model.pt --data test.nc --output-dir predictions/mv
   python examples/evaluate_multi_variable.py --predictions predictions/mv/predictions.nc --ground-truth test.nc --output-dir predictions/mv/metrics
   ```

2. **Rolling Forecast Workflow**
   ```bash
   python train_convlstm.py --multi-variable --enable-rollout-training --data data.nc --output-dir checkpoints/rolling
   python run_inference_convlstm.py --checkpoint checkpoints/rolling/best_model.pt --data test.nc --rolling-steps 6 --output-dir predictions/rolling
   python examples/evaluate_multi_variable.py --predictions predictions/rolling/predictions.nc --ground-truth test.nc --rolling-forecast --output-dir predictions/rolling/metrics
   ```

## Contributing

When adding new documentation:
1. Update this index
2. Add cross-references to related documents
3. Include examples where appropriate
4. Update the main README if adding major features

## Version History

- **v1.0** (2024-12): Initial multi-variable prediction implementation
  - Multi-variable mode
  - Rolling forecasts
  - Comprehensive documentation
  - Example scripts

## Contact

For questions or issues:
- Check relevant documentation above
- Review example scripts
- Check specification documents
- Consult training logs

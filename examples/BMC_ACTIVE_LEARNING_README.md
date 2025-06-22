# BMC Active Learning with GNINA Oracle

This directory contains scripts for running active learning experiments on the BMC FEP validation set using GNINA as the molecular evaluation oracle.

## Overview

The BMC (β-secretase 1) FEP validation set contains molecules with experimental binding affinity data. We use active learning to intelligently select molecules from this pool using GNINA docking scores as the selection criterion.

## Files

### Main Scripts

1. **`active_learning_bmc_gnina.py`** - Complete active learning pipeline
   - Loads molecules from SDF file with experimental data
   - Uses GNINA docking oracle for evaluation
   - Implements uncertainty sampling for selection
   - Tracks molecules by selection round
   - Saves comprehensive results

2. **`bmc_al_demo_simple.py`** - Simplified demo version
   - Lighter computational requirements
   - Demonstrates core AL workflow
   - Good for testing and learning

3. **`analyze_bmc_results.py`** - Results analysis and visualization
   - Statistical analysis of selection rounds
   - Correlation analysis between experimental and predicted values
   - Molecular diversity analysis
   - Generates plots and summary reports

### Configuration and Utilities

4. **`bmc_al_config.py`** - Configuration parameters
5. **`run_bmc_pipeline.sh`** - Batch script to run complete pipeline

## Quick Start

### Option 1: Run Simple Demo

```bash
# Run the simplified demo (recommended for first use)
python examples/bmc_al_demo_simple.py

# Analyze results
python examples/analyze_bmc_results.py
```

### Option 2: Run Complete Pipeline

```bash
# Run the complete pipeline script
./examples/run_bmc_pipeline.sh
```

### Option 3: Full Active Learning Experiment

```bash
# Run full experiment with custom parameters
python examples/active_learning_bmc_gnina.py \
    --batch_size 10 \
    --max_rounds 20 \
    --initial_size 50 \
    --output_dir results/bmc_al_custom
```

## Parameters

### Active Learning Parameters

- `--batch_size`: Number of molecules to select per round (default: 10)
- `--max_rounds`: Maximum number of AL rounds (default: 20)
- `--initial_size`: Size of initial random selection (default: 50)
- `--random_seed`: Random seed for reproducibility (default: 42)

### File Paths

- `--sdf_file`: Path to BMC SDF file (default: data/targets/BMC_FEP_validation_set_J_Med_Chem_2020.sdf)
- `--protein_file`: Path to protein PDB file (default: data/BMC_FEP_protein_model_6ZB1.pdb)
- `--output_dir`: Output directory for results (default: results/bmc_al_gnina)

## Output Files

### Results Structure

```
results/bmc_al_[experiment]/
├── experiment_metadata.json          # Experiment configuration and summary
├── bmc_al_[experiment]_complete_results.csv  # Complete results CSV
├── round_00_results.json            # Individual round results
├── round_01_results.json
├── ...
├── selection_analysis.png           # Analysis plots
└── analysis_summary.json           # Analysis summary
```

### Results CSV Columns

- `mol_id`: Unique molecule identifier
- `entry_name`: Molecule name from SDF
- `smiles`: SMILES representation
- `selected_round`: Round when molecule was selected (None if not selected)
- `gnina_score`: GNINA docking score
- `gnina_cnn_score`: GNINA CNN score
- `uncertainty`: Calculated uncertainty for selection
- `pic50_exp`: Experimental PIC50 value
- `exp_dg`: Experimental ΔG value
- `mw`, `logp`, `hbd`, `hba`, `tpsa`: Molecular descriptors

## Active Learning Strategy

### Selection Process

1. **Initial Selection**: Random selection of initial training set
2. **Evaluation**: Score molecules using GNINA docking oracle
3. **Uncertainty Calculation**: Calculate uncertainty based on score distribution
4. **Batch Selection**: Select highest uncertainty molecules for next round
5. **Iteration**: Repeat until stopping criteria met

### Uncertainty Sampling

The current implementation uses a simple uncertainty measure based on deviation from mean scores. More sophisticated methods could include:

- Model ensemble disagreement
- Prediction confidence intervals
- Gradient-based uncertainty
- Bayesian neural network uncertainty

## Requirements

### Software Dependencies

- Python 3.8+
- RDKit
- NumPy, Pandas
- Matplotlib, Seaborn
- GNINA (for docking evaluation)

### Data Files

The following files must be present:

```
data/
├── targets/BMC_FEP_validation_set_J_Med_Chem_2020.sdf
└── BMC_FEP_protein_model_6ZB1.pdb
```

## Example Usage

### Basic Demo

```python
# Run a quick 5-round demo with small batches
python examples/bmc_al_demo_simple.py
```

### Custom Experiment

```python
# Run experiment with specific parameters
python examples/active_learning_bmc_gnina.py \
    --batch_size 5 \
    --max_rounds 10 \
    --initial_size 30 \
    --output_dir results/bmc_small_test
```

### Analysis Only

```python
# Analyze existing results
python examples/analyze_bmc_results.py
```

## Performance Notes

- **GNINA Evaluation**: Each molecule evaluation takes ~1-10 seconds depending on complexity
- **Computational Cost**: Full experiment (20 rounds × 10 molecules) ≈ 30-60 minutes
- **Memory Usage**: Moderate (depends on number of molecules in pool)

## Customization

### Adding New Selection Strategies

Extend the uncertainty calculation in `calculate_uncertainty()` method:

```python
def calculate_uncertainty(molecules):
    # Your custom uncertainty calculation
    for mol in molecules:
        mol['uncertainty'] = your_uncertainty_function(mol)
    return molecules
```

### Adding New Oracles

Replace or supplement GNINA oracle with other evaluation methods:

```python
# In setup_gnina_oracle()
oracle = YourCustomOracle(config=your_config)
```

### Custom Molecular Descriptors

Add descriptors in `_calculate_descriptors()` method:

```python
def _calculate_descriptors(self, mol):
    descriptors = {
        'your_descriptor': your_calculation(mol),
        # ... existing descriptors
    }
    return descriptors
```

## Troubleshooting

### Common Issues

1. **GNINA not found**: Ensure GNINA is installed and in PATH
2. **File not found**: Check that SDF and PDB files exist in data/ directory
3. **Memory issues**: Reduce batch size or use smaller molecule pool
4. **RDKit errors**: Check molecule validity in SDF file

### Debug Mode

Add debug logging:

```python
logging.basicConfig(level=logging.DEBUG)
```

## References

- BMC validation set: J. Med. Chem. 2020 (FEP+ validation)
- GNINA: CNN-based molecular docking
- Active Learning: Uncertainty sampling for molecular discovery

## Contributing

To contribute improvements or new features:

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## License

See main repository LICENSE file.

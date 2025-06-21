# BoltzOracle Usage Guide

## Quick Start

```python
from al_fep.oracles.boltz_oracle import BoltzOracle

# Basic usage
oracle = BoltzOracle(target="test")
result = oracle.evaluate(["CCO"])  # Ethanol
print(result)
```

## Configuration Options

### 1. Default (Recommended)
```python
oracle = BoltzOracle(target="test")
```

### 2. Use Existing YAML File
```python
config = {
    "boltz": {
        "yaml_file_path": "/path/to/your/affinity.yaml",
        "preserve_yaml_files": True
    }
}
oracle = BoltzOracle(target="test", config=config)
```

### 3. High Quality Prediction
```python
config = {
    "boltz": {
        "diffusion_samples": 5,
        "recycling_steps": 5,
        "sampling_steps": 500,
        "diffusion_samples_affinity": 10
    }
}
oracle = BoltzOracle(target="test", config=config)
```

### 4. Template Directory
```python
config = {
    "boltz": {
        "yaml_template_dir": "/tmp/boltz_templates",
        "preserve_yaml_files": True
    }
}
oracle = BoltzOracle(target="test", config=config)
```

## Testing

```bash
# Run comprehensive test suite
python test_boltz_comprehensive.py

# Run basic test
python test_boltz_oracle.py
```

## Expected Output

```python
{
    "score": 6.5,                    # Higher = better binding
    "binding_affinity": -6.5,        # pIC50/pKd value
    "binding_probability": 0.85,     # Probability of binding (0-1)
    "confidence_score": 0.92,        # Model confidence (0-1)
    "structure_file": "path/to/structure.pdb",
    "method": "Boltz-2"
}
```

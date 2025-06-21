#!/usr/bin/env python3
"""
BoltzOracle Test Cleanup and Consolidation Script
"""

import os
import shutil
from pathlib import Path

def cleanup_old_test_files():
    """Remove redundant test files and consolidate functionality."""
    
    # Files to remove (redundant/outdated)
    files_to_remove = [
        "test_boltz_simple.py",
        "demo_boltz_yaml_config.py", 
        "test_existing_yaml.py",
        "use_existing_yaml.py",
        "debug_boltz.py"
    ]
    
    # Files to keep
    files_to_keep = [
        "test_boltz_comprehensive.py",  # New comprehensive test
        "test_boltz_oracle.py"  # Keep as backup if needed
    ]
    
    current_dir = Path(__file__).parent
    
    print("🧹 BoltzOracle Test File Cleanup")
    print("=" * 50)
    
    removed_count = 0
    kept_count = 0
    
    # List all boltz-related files
    all_files = list(current_dir.glob("*boltz*")) + list(current_dir.glob("*existing*"))
    
    print(f"📁 Found {len(all_files)} BoltzOracle-related files:")
    
    for file_path in all_files:
        file_name = file_path.name
        
        if file_name in files_to_remove:
            try:
                # Create backup before removing
                backup_dir = current_dir / "backup_old_tests"
                backup_dir.mkdir(exist_ok=True)
                
                backup_path = backup_dir / file_name
                shutil.copy2(file_path, backup_path)
                
                # Remove original
                file_path.unlink()
                
                print(f"   🗑️  Removed: {file_name} (backed up)")
                removed_count += 1
                
            except Exception as e:
                print(f"   ❌ Failed to remove {file_name}: {e}")
        
        elif file_name in files_to_keep:
            print(f"   ✅ Kept: {file_name}")
            kept_count += 1
        
        else:
            print(f"   ❓ Unknown: {file_name}")
    
    print(f"\n📊 Summary:")
    print(f"   Removed: {removed_count} files")
    print(f"   Kept: {kept_count} files")
    print(f"   Backup location: ./backup_old_tests/")
    
    print(f"\n🎯 Recommended usage:")
    print(f"   python test_boltz_comprehensive.py  # Complete test suite")
    print(f"   python test_boltz_oracle.py        # Basic test (if needed)")


def create_usage_guide():
    """Create a simple usage guide for BoltzOracle."""
    
    usage_guide = '''# BoltzOracle Usage Guide

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
'''
    
    guide_path = Path(__file__).parent / "BOLTZ_USAGE.md"
    with open(guide_path, 'w') as f:
        f.write(usage_guide)
    
    print(f"📚 Created usage guide: {guide_path}")


if __name__ == "__main__":
    cleanup_old_test_files()
    create_usage_guide()
    print(f"\n🎉 Cleanup complete! BoltzOracle tests are now organized.")

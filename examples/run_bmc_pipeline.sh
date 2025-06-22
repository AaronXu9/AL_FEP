#!/bin/bash
"""
Batch script to run BMC Active Learning experiments

This script runs the complete pipeline:
1. Simple demo
2. Full active learning experiment (optional)
3. Results analysis
"""

echo "========================================"
echo "BMC Active Learning Pipeline"
echo "========================================"

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Create results directory
mkdir -p results/bmc_al_demo
mkdir -p results/bmc_al_gnina

echo "Step 1: Running simple active learning demo..."
python examples/bmc_al_demo_simple.py

if [ $? -eq 0 ]; then
    echo "✓ Simple demo completed successfully"
else
    echo "✗ Simple demo failed"
    exit 1
fi

echo
echo "Step 2: Analyzing results..."
python examples/analyze_bmc_results.py

if [ $? -eq 0 ]; then
    echo "✓ Analysis completed successfully"
else
    echo "✗ Analysis failed"
    exit 1
fi

echo
echo "========================================"
echo "Pipeline completed successfully!"
echo "Check results in:"
echo "  - results/bmc_al_demo/"
echo "========================================"

# Optional: Run full experiment (commented out due to computational cost)
# echo "Step 3: Running full active learning experiment..."
# python examples/active_learning_bmc_gnina.py --batch_size 5 --max_rounds 10 --initial_size 30

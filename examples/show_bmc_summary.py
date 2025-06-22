#!/usr/bin/env python3
"""
Summary script for BMC Active Learning Results

This script provides a quick summary of the active learning experiment results.
"""

import pandas as pd
import json
from pathlib import Path


def print_experiment_summary():
    """Print a comprehensive summary of the experiment."""
    results_file = "results/bmc_al_demo/bmc_al_demo_results.csv"
    summary_file = "results/bmc_al_demo/analysis_summary.json"
    
    if not Path(results_file).exists():
        print("❌ Results file not found. Please run the active learning experiment first.")
        return
    
    # Load data
    df = pd.read_csv(results_file)
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    print("🧬 BMC ACTIVE LEARNING EXPERIMENT RESULTS")
    print("=" * 60)
    
    # Basic statistics
    print(f"📊 Experiment Overview:")
    print(f"   • Total molecules processed: {len(df):,}")
    print(f"   • Selection rounds completed: {df['selected_round'].nunique()}")
    print(f"   • Molecules selected: {df['selected_round'].notna().sum():,}")
    print(f"   • Selection rate: {df['selected_round'].notna().mean()*100:.1f}%")
    print()
    
    # GNINA scoring results
    gnina_stats = summary['scoring_statistics']['gnina_scores']
    print(f"🎯 GNINA Docking Results:")
    print(f"   • Mean GNINA score: {gnina_stats['mean']:.3f}")
    print(f"   • Score range: {gnina_stats['min']:.3f} to {gnina_stats['max']:.3f}")
    print(f"   • Standard deviation: {gnina_stats['std']:.3f}")
    print(f"   • Valid scores: {gnina_stats['valid_scores']}/{len(df)}")
    print()
    
    # Experimental data correlation
    exp_stats = summary['scoring_statistics']['experimental_pic50']
    corr_stats = summary.get('correlation_analysis', {})
    print(f"🔬 Experimental Data:")
    print(f"   • Mean experimental PIC50: {exp_stats['mean']:.2f}")
    print(f"   • PIC50 range: {df['pic50_exp'].min():.2f} to {df['pic50_exp'].max():.2f}")
    print(f"   • Correlation (PIC50 vs GNINA): {corr_stats.get('pic50_gnina_correlation', 0):.3f}")
    print()
    
    # Molecular properties
    mol_props = summary['molecular_properties']['molecular_weight']
    print(f"⚗️  Molecular Properties:")
    print(f"   • Mean molecular weight: {mol_props['mean']:.1f} Da")
    print(f"   • MW range: {mol_props['range'][0]:.1f} - {mol_props['range'][1]:.1f} Da")
    print()
    
    # Selection by rounds
    print(f"📈 Selection by Rounds:")
    round_counts = df.groupby('selected_round').size().sort_index()
    for round_num, count in round_counts.items():
        if round_num == 0:
            print(f"   • Round {int(round_num)} (initial): {count} molecules")
        else:
            print(f"   • Round {int(round_num)}: {count} molecules")
    print()
    
    # Top scoring molecules
    print(f"🏆 Top 5 GNINA Scoring Molecules:")
    top_molecules = df.nlargest(5, 'gnina_score')[['entry_name', 'gnina_score', 'pic50_exp', 'selected_round']]
    for _, mol in top_molecules.iterrows():
        round_info = f"Round {int(mol['selected_round'])}" if pd.notna(mol['selected_round']) else "Not selected"
        print(f"   • {mol['entry_name']}: {mol['gnina_score']:.3f} (PIC50: {mol['pic50_exp']:.2f}, {round_info})")
    print()
    
    # Files generated
    print(f"📁 Generated Files:")
    output_dir = Path("results/bmc_al_demo")
    for file in output_dir.glob("*"):
        if file.is_file():
            size_kb = file.stat().st_size / 1024
            print(f"   • {file.name}: {size_kb:.1f} KB")
    print()
    
    print("✅ Active learning experiment completed successfully!")
    print("💡 Next steps:")
    print("   • Analyze selection_analysis.png for visual insights")
    print("   • Run full experiment with more rounds for better coverage")
    print("   • Try different selection strategies or oracles")
    print("=" * 60)


if __name__ == "__main__":
    print_experiment_summary()

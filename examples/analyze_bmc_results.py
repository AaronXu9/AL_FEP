#!/usr/bin/env python3
"""
Analysis script for BMC Active Learning Results

This script analyzes the results of active learning experiments
and generates plots and statistics.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

# Set plot style
plt.style.use('default')
sns.set_palette("husl")


def load_results(results_file):
    """Load results from CSV file."""
    df = pd.read_csv(results_file)
    return df


def analyze_selection_rounds(df):
    """Analyze molecule selection by rounds."""
    print("=== SELECTION ROUND ANALYSIS ===")
    
    round_stats = df.groupby('selected_round').agg({
        'mol_id': 'count',
        'gnina_score': ['mean', 'std', 'min', 'max'],
        'pic50_exp': ['mean', 'std'],
        'mw': ['mean', 'std']
    }).round(3)
    
    print(round_stats)
    return round_stats


def plot_selection_analysis(df, output_dir):
    """Create plots for selection analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: GNINA scores by round
    ax1 = axes[0, 0]
    df_valid = df[df['gnina_score'].notna()]
    sns.boxplot(data=df_valid, x='selected_round', y='gnina_score', ax=ax1)
    ax1.set_title('GNINA Scores by Selection Round')
    ax1.set_xlabel('Selection Round')
    ax1.set_ylabel('GNINA Score')
    
    # Plot 2: Experimental vs Predicted correlation
    ax2 = axes[0, 1]
    df_both = df[(df['pic50_exp'].notna()) & (df['gnina_score'].notna())]
    if len(df_both) > 0:
        ax2.scatter(df_both['pic50_exp'], df_both['gnina_score'], 
                   c=df_both['selected_round'], cmap='viridis', alpha=0.7)
        ax2.set_xlabel('Experimental PIC50')
        ax2.set_ylabel('GNINA Score')
        ax2.set_title('Experimental vs GNINA Score')
        
        # Add correlation
        corr = df_both['pic50_exp'].corr(df_both['gnina_score'])
        ax2.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                transform=ax2.transAxes, verticalalignment='top')
    
    # Plot 3: Molecular weight distribution
    ax3 = axes[1, 0]
    df_valid_mw = df[df['mw'].notna()]
    for round_num in sorted(df_valid_mw['selected_round'].unique()):
        round_data = df_valid_mw[df_valid_mw['selected_round'] == round_num]
        ax3.hist(round_data['mw'], alpha=0.6, label=f'Round {round_num}', bins=20)
    ax3.set_xlabel('Molecular Weight')
    ax3.set_ylabel('Count')
    ax3.set_title('Molecular Weight Distribution by Round')
    ax3.legend()
    
    # Plot 4: Cumulative selection
    ax4 = axes[1, 1]
    cumulative_counts = df.groupby('selected_round').size().cumsum()
    ax4.plot(cumulative_counts.index, cumulative_counts.values, 'o-')
    ax4.set_xlabel('Selection Round')
    ax4.set_ylabel('Cumulative Molecules Selected')
    ax4.set_title('Cumulative Molecule Selection')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = Path(output_dir) / "selection_analysis.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Selection analysis plot saved to: {plot_file}")


def analyze_molecular_diversity(df):
    """Analyze molecular diversity in selected sets."""
    print("\n=== MOLECULAR DIVERSITY ANALYSIS ===")
    
    # Basic statistics
    stats = df.agg({
        'mw': ['mean', 'std', 'min', 'max'],
        'pic50_exp': ['mean', 'std', 'count'],
        'gnina_score': ['mean', 'std', 'count']
    }).round(3)
    
    print("Basic Statistics:")
    print(stats)
    
    # Round-by-round diversity
    print("\nDiversity by Round (MW std as proxy):")
    diversity = df.groupby('selected_round')['mw'].std().round(3)
    print(diversity)
    
    return stats, diversity


def generate_summary_report(df, output_dir):
    """Generate a comprehensive summary report."""
    report = {
        'experiment_summary': {
            'total_molecules': len(df),
            'completed_rounds': df['selected_round'].nunique(),
            'molecules_per_round': df.groupby('selected_round').size().to_dict(),
            'analysis_timestamp': datetime.now().isoformat()
        },
        'scoring_statistics': {
            'gnina_scores': {
                'mean': float(df['gnina_score'].mean()) if df['gnina_score'].notna().any() else None,
                'std': float(df['gnina_score'].std()) if df['gnina_score'].notna().any() else None,
                'min': float(df['gnina_score'].min()) if df['gnina_score'].notna().any() else None,
                'max': float(df['gnina_score'].max()) if df['gnina_score'].notna().any() else None,
                'valid_scores': int(df['gnina_score'].notna().sum())
            },
            'experimental_pic50': {
                'mean': float(df['pic50_exp'].mean()) if df['pic50_exp'].notna().any() else None,
                'std': float(df['pic50_exp'].std()) if df['pic50_exp'].notna().any() else None,
                'count': int(df['pic50_exp'].notna().sum())
            }
        },
        'molecular_properties': {
            'molecular_weight': {
                'mean': float(df['mw'].mean()),
                'std': float(df['mw'].std()),
                'range': [float(df['mw'].min()), float(df['mw'].max())]
            }
        }
    }
    
    # Add correlation analysis
    if df['pic50_exp'].notna().any() and df['gnina_score'].notna().any():
        valid_both = df[(df['pic50_exp'].notna()) & (df['gnina_score'].notna())]
        if len(valid_both) > 1:
            correlation = float(valid_both['pic50_exp'].corr(valid_both['gnina_score']))
            report['correlation_analysis'] = {
                'pic50_gnina_correlation': correlation,
                'samples_with_both': len(valid_both)
            }
    
    # Save report
    report_file = Path(output_dir) / "analysis_summary.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nSummary report saved to: {report_file}")
    return report


def main():
    """Main analysis function."""
    # Default paths
    results_file = "results/bmc_al_demo/bmc_al_demo_results.csv"
    output_dir = "results/bmc_al_demo"
    
    # Check if results file exists
    if not Path(results_file).exists():
        print(f"Results file not found: {results_file}")
        print("Please run the active learning experiment first.")
        return
    
    print("Loading results...")
    df = load_results(results_file)
    
    print(f"Loaded {len(df)} molecules from {df['selected_round'].nunique()} rounds")
    
    # Perform analyses
    round_stats = analyze_selection_rounds(df)
    plot_selection_analysis(df, output_dir)
    diversity_stats = analyze_molecular_diversity(df)
    summary_report = generate_summary_report(df, output_dir)
    
    # Print final summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETED")
    print("="*60)
    print(f"Results analyzed: {len(df)} molecules")
    print(f"Selection rounds: {df['selected_round'].nunique()}")
    print(f"Valid GNINA scores: {df['gnina_score'].notna().sum()}")
    print(f"Experimental data points: {df['pic50_exp'].notna().sum()}")
    print(f"Output directory: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

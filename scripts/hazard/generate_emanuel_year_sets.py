#!/usr/bin/env python3
"""
generate_emanuel_year_sets.py - Generate stochastic TC year-sets from Emanuel event sets

This script generates year-sets for Monte Carlo analysis using the Emanuel TC
event sets with their observed annual frequency distributions.

Key difference from synthetic TC approach:
- Instead of Poisson(λ) with fixed λ, we RESAMPLE from observed freqyear values
- This preserves the empirical (underdispersed) frequency distribution
- Maintains climate variability structure from ERA5/GCM simulations

Usage:
    python scripts/hazard/generate_emanuel_year_sets.py --event_set FL_era5_reanalcal --n_years 10000 --seed 42

Outputs:
    - <IMPACTS_BASE>/<event_set>/year_sets_N10000_seed42.csv

Author: Simona Meiler
Date: December 2025
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Add repo to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from fl_risk_model import config as cfg

# ============================================================================
# CONFIGURATION
# ============================================================================

# Detect if running on cluster (Sherlock) vs local
import os
IS_CLUSTER = 'SLURM_JOB_ID' in os.environ or 'SHERLOCK' in os.environ.get('HOSTNAME', '')

if IS_CLUSTER:
    # Sherlock cluster paths (Stanford HPC)
    IMPACTS_BASE = Path("/home/groups/bakerjw/smeiler/climada_data/data/impact")
else:
    # Local paths - MODIFY THESE FOR YOUR SETUP
    DATA_DIR = Path(cfg.DATA_DIR)
    IMPACTS_BASE = DATA_DIR / "hazard" / "emanuel"

# Default parameters
DEFAULT_N_YEARS = 10000
DEFAULT_SEED = 42
DEFAULT_SAMPLING = "uniform"  # Event selection within each year

# ============================================================================
# YEAR-SET GENERATION
# ============================================================================

def load_event_data(event_set_dir):
    """
    Load all events and annual frequencies for an event set.
    
    Args:
        event_set_dir: Path to event set directory
    
    Returns:
        tuple: (all_events_df, annual_frequencies_df)
    """
    events_path = event_set_dir / "all_events.csv"
    freq_path = event_set_dir / "annual_frequencies.csv"
    
    if not events_path.exists():
        raise FileNotFoundError(
            f"Events file not found: {events_path}\n"
            "Run preprocessing first: submit_precompute_emanuel_sherlock.sh"
        )
    
    if not freq_path.exists():
        raise FileNotFoundError(
            f"Annual frequencies file not found: {freq_path}\n"
            "Should be created during preprocessing"
        )
    
    events_df = pd.read_csv(events_path)
    freq_df = pd.read_csv(freq_path)
    
    print(f"\nLoaded event data:")
    print(f"  Total events: {len(events_df):,}")
    n_nonzero = (events_df['total_damage_usd'] > 0).sum()
    print(f"  Events with damage >$0: {n_nonzero:,} ({n_nonzero/len(events_df)*100:.1f}%)")
    if n_nonzero > 0:
        print(f"  Damage range (non-zero): ${events_df[events_df['total_damage_usd'] > 0]['total_damage_usd'].min()/1e9:.3f}B - ${events_df['total_damage_usd'].max()/1e9:.1f}B")
    print(f"  Historical years: {len(freq_df)}")
    print(f"  Mean annual frequency: {freq_df['mean_frequency'].iloc[0]:.3f}")
    print(f"  Frequency range: [{freq_df['frequency'].min():.3f}, {freq_df['frequency'].max():.3f}]")
    print(f"  Frequency std dev: {freq_df['frequency'].std():.3f}")
    
    return events_df, freq_df


def generate_year_sets_from_freqyear(all_events_df,
                                      annual_frequencies,
                                      n_years=10000,
                                      seed=42,
                                      sampling_mode="uniform"):
    """
    Generate stochastic TC year-sets by resampling observed annual frequencies.
    
    This preserves the empirical frequency distribution rather than assuming Poisson.
    
    Args:
        all_events_df: DataFrame with all events (including zero-damage)
        annual_frequencies: Array of observed annual frequencies
        n_years: Number of simulation years
        seed: Random seed
        sampling_mode: "uniform" or "weighted" (by damage/intensity)
    
    Returns:
        DataFrame with columns: year_id, event_id, event_sequence, n_events_year, 
                               sampled_frequency
    """
    print(f"\nGenerating year-sets from observed frequencies...")
    print(f"  Simulation years: {n_years:,}")
    print(f"  Resampling from {len(annual_frequencies)} observed years")
    print(f"  Sampling mode: {sampling_mode}")
    print(f"  Random seed: {seed}")
    
    rng = np.random.default_rng(seed)
    
    # Resample annual frequencies with replacement
    sampled_frequencies = rng.choice(annual_frequencies, size=n_years, replace=True)
    
    # Convert frequencies to integer event counts
    # Each frequency represents expected events, so we need to round stochastically
    # to preserve the mean
    n_events_per_year = np.zeros(n_years, dtype=int)
    for i, freq in enumerate(sampled_frequencies):
        # Split into integer and fractional parts
        base = int(freq)
        frac = freq - base
        # Stochastically round: base events + Bernoulli(frac) for one more
        n_events_per_year[i] = base + (rng.random() < frac)
    
    print(f"\n  Sampled frequency statistics:")
    print(f"    Mean: {sampled_frequencies.mean():.3f}")
    print(f"    Std: {sampled_frequencies.std():.3f}")
    print(f"    Range: [{sampled_frequencies.min():.3f}, {sampled_frequencies.max():.3f}]")
    print(f"  Realized event counts:")
    print(f"    Mean: {n_events_per_year.mean():.3f}")
    print(f"    Range: [{n_events_per_year.min()}, {n_events_per_year.max()}]")
    
    # Get event pool (ALL events including zero-damage)
    event_ids = all_events_df['event_id'].values
    
    # Calculate sampling weights (if needed)
    if sampling_mode == "weighted":
        weights = all_events_df['total_damage_usd'].values
        weights = weights / weights.sum()
        print(f"  Using damage-weighted event sampling")
    else:
        weights = None
        print(f"  Using uniform event sampling (preserves true intensity distribution)")
    
    # Generate year-sets
    rows = []
    
    for year_id in range(1, n_years + 1):
        n_events = n_events_per_year[year_id - 1]
        sampled_freq = sampled_frequencies[year_id - 1]
        
        if n_events == 0:
            # Zero-event year
            rows.append({
                'year_id': year_id,
                'event_id': None,
                'event_sequence': None,
                'n_events_year': 0,
                'sampled_frequency': sampled_freq
            })
        else:
            # Sample events from viable pool
            if n_events > len(event_ids):
                # Edge case: sample with replacement
                sampled_events = rng.choice(event_ids, size=n_events, replace=True, p=weights)
            else:
                # Normal: sample without replacement within year
                sampled_events = rng.choice(event_ids, size=n_events, replace=False, p=weights)
            
            for seq, event_id in enumerate(sampled_events):
                rows.append({
                    'year_id': year_id,
                    'event_id': event_id,
                    'event_sequence': seq,
                    'n_events_year': n_events,
                    'sampled_frequency': sampled_freq
                })
    
    year_sets_df = pd.DataFrame(rows)
    
    # Summary statistics
    year_counts = year_sets_df.groupby('year_id')['n_events_year'].first()
    zero_years = (year_counts == 0).sum()
    single_event_years = (year_counts == 1).sum()
    multi_event_years = (year_counts > 1).sum()
    max_events = year_counts.max()
    
    print(f"\n  Year-set statistics:")
    print(f"    Zero-event years: {zero_years:,} ({100*zero_years/n_years:.1f}%)")
    print(f"    Single-event years: {single_event_years:,} ({100*single_event_years/n_years:.1f}%)")
    print(f"    Multi-event years: {multi_event_years:,} ({100*multi_event_years/n_years:.1f}%)")
    print(f"    Maximum events in a year: {max_events}")
    
    return year_sets_df


def validate_year_sets(year_sets_df, viable_events_df, original_frequencies):
    """
    Validate year-sets against expected distributions.
    
    Args:
        year_sets_df: Generated year-sets
        viable_events_df: Viable events pool
        original_frequencies: Original observed frequencies
    """
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)
    
    # 1. Frequency distribution
    n_years = year_sets_df['year_id'].nunique()
    sampled_freqs = year_sets_df.groupby('year_id')['sampled_frequency'].first()
    
    print("\nFrequency distribution comparison:")
    print(f"  Original mean: {original_frequencies.mean():.3f}")
    print(f"  Sampled mean: {sampled_freqs.mean():.3f}")
    print(f"  Original std: {original_frequencies.std():.3f}")
    print(f"  Sampled std: {sampled_freqs.std():.3f}")
    print(f"  Variance-to-mean ratio (original): {original_frequencies.var()/original_frequencies.mean():.3f}")
    print(f"  Variance-to-mean ratio (sampled): {sampled_freqs.var()/sampled_freqs.mean():.3f}")
    
    # 2. Event count distribution
    event_counts = year_sets_df.groupby('year_id')['n_events_year'].first()
    
    print(f"\nEvent count distribution:")
    print(f"  Mean: {event_counts.mean():.3f}")
    print(f"  Std: {event_counts.std():.3f}")
    for i in range(min(6, event_counts.max() + 1)):
        count = (event_counts == i).sum()
        print(f"  {i} events: {count:,} years ({100*count/n_years:.1f}%)")
    
    # 3. Event sampling coverage
    non_zero_events = year_sets_df[year_sets_df['event_id'].notna()]['event_id'].value_counts()
    
    print(f"\nEvent sampling coverage:")
    print(f"  Unique events sampled: {len(non_zero_events)} / {len(viable_events_df)} viable")
    print(f"  Coverage: {100*len(non_zero_events)/len(viable_events_df):.1f}%")
    print(f"  Most frequent event: {non_zero_events.iloc[0]} occurrences")
    print(f"  Least frequent event: {non_zero_events.iloc[-1]} occurrences")
    
    # 4. Damage distribution
    year_damage = year_sets_df.merge(
        viable_events_df[['event_id', 'total_damage_usd']],
        on='event_id',
        how='left'
    )
    annual_damage = year_damage.groupby('year_id')['total_damage_usd'].sum()
    
    print(f"\nAnnual damage distribution:")
    print(f"  Mean: ${annual_damage.mean()/1e9:.2f}B")
    print(f"  Median: ${annual_damage.median()/1e9:.2f}B")
    print(f"  90th percentile: ${annual_damage.quantile(0.90)/1e9:.2f}B")
    print(f"  99th percentile: ${annual_damage.quantile(0.99)/1e9:.2f}B")
    print(f"  Maximum: ${annual_damage.max()/1e9:.2f}B")


def save_year_sets(year_sets_df, output_dir, event_set_name, n_years, seed):
    """
    Save year-sets to CSV.
    
    Args:
        year_sets_df: Year-sets DataFrame
        output_dir: Output directory
        event_set_name: Name of event set
        n_years: Number of years
        seed: Random seed
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"year_sets_N{n_years}_seed{seed}.csv"
    output_path = output_dir / filename
    
    year_sets_df.to_csv(output_path, index=False)
    
    print(f"\nSaved year-sets: {output_path}")
    print(f"  Rows: {len(year_sets_df):,}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Also save summary
    summary = {
        'event_set': event_set_name,
        'n_years': n_years,
        'seed': seed,
        'n_rows': len(year_sets_df),
        'n_zero_years': (year_sets_df.groupby('year_id')['n_events_year'].first() == 0).sum(),
        'mean_events_per_year': year_sets_df.groupby('year_id')['n_events_year'].first().mean(),
        'max_events_per_year': year_sets_df['n_events_year'].max()
    }
    
    summary_df = pd.DataFrame([summary])
    summary_path = output_dir / f"year_sets_N{n_years}_seed{seed}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    print(f"Saved summary: {summary_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate stochastic TC year-sets from Emanuel event sets"
    )
    parser.add_argument(
        "--event_set",
        type=str,
        required=True,
        help="Event set name (e.g., FL_era5_reanalcal)"
    )
    parser.add_argument(
        "--n_years",
        type=int,
        default=DEFAULT_N_YEARS,
        help=f"Number of simulation years (default: {DEFAULT_N_YEARS})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})"
    )
    parser.add_argument(
        "--sampling",
        type=str,
        default=DEFAULT_SAMPLING,
        choices=["uniform", "weighted"],
        help=f"Event sampling mode (default: {DEFAULT_SAMPLING})"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print(f"GENERATING EMANUEL TC YEAR-SETS: {args.event_set}")
    print("="*80)
    
    # Set up paths
    event_set_dir = IMPACTS_BASE / args.event_set
    
    if not event_set_dir.exists():
        raise FileNotFoundError(
            f"Event set directory not found: {event_set_dir}\n"
            "Run preprocessing first: submit_precompute_emanuel_sherlock.sh"
        )
    
    print(f"\nEvent set directory: {event_set_dir}")
    
    # Load data
    all_events_df, freq_df = load_event_data(event_set_dir)
    
    # Generate year-sets
    year_sets_df = generate_year_sets_from_freqyear(
        all_events_df=all_events_df,
        annual_frequencies=freq_df['frequency'].values,
        n_years=args.n_years,
        seed=args.seed,
        sampling_mode=args.sampling
    )
    
    # Validate
    validate_year_sets(
        year_sets_df=year_sets_df,
        viable_events_df=all_events_df,
        original_frequencies=freq_df['frequency'].values
    )
    
    # Save
    save_year_sets(
        year_sets_df=year_sets_df,
        output_dir=event_set_dir,
        event_set_name=args.event_set,
        n_years=args.n_years,
        seed=args.seed
    )
    
    print("\n" + "="*80)
    print("[OK] SUCCESS")
    print("="*80)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
setup_emanuel_metadata.py - Generate metadata files for Emanuel MC from existing event CSVs

This script generates the metadata files needed for year-set generation:
1. all_events.csv - Summary of all events with total damage
2. annual_frequencies.csv - Annual frequency data from track .mat file

Run this if you have the per-event CSV files but are missing the metadata files.

Usage:
    python scripts/hazard/setup_emanuel_metadata.py --event_set FL_era5_reanalcal

Author: Simona Meiler
Date: 2025
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm

# Add repo to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from fl_risk_model import config as cfg

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path(cfg.DATA_DIR)

# Default track file location
DEFAULT_TRACKS_DIR = Path.home() / "climada" / "data" / "tracks" / "Kerry" / "Florida"

# Emanuel event directory
EMANUEL_DIR = DATA_DIR / "hazard" / "emanuel"


def get_track_filename(event_set_name):
    """Convert event set name to track filename."""
    if event_set_name.startswith("FL_"):
        base_name = event_set_name[3:]
    else:
        base_name = event_set_name
    return f"Simona_FLA_AL_{base_name}.mat"


def load_freqyear_from_tracks(track_file):
    """Load freqyear array from track .mat file."""
    print(f"\nLoading annual frequencies from: {track_file}")
    
    if not track_file.exists():
        raise FileNotFoundError(f"Track file not found: {track_file}")
    
    mat_data = sio.loadmat(str(track_file))
    
    freq = float(mat_data['freq'].flatten()[0])
    freqyear = mat_data['freqyear'].flatten()
    
    print(f"  Mean annual frequency: {freq:.3f} events/year")
    print(f"  Years in dataset: {len(freqyear)}")
    print(f"  Frequency range: [{freqyear.min():.3f}, {freqyear.max():.3f}]")
    
    return freq, freqyear


def create_all_events_csv(event_dir, output_path):
    """Create all_events.csv from individual event CSVs."""
    print(f"\nCreating all_events.csv from: {event_dir}")
    
    event_files = sorted(event_dir.glob("*.csv"))
    
    if not event_files:
        raise FileNotFoundError(f"No event CSV files found in: {event_dir}")
    
    print(f"  Found {len(event_files)} event files")
    
    events = []
    for f in tqdm(event_files, desc="Reading events"):
        event_id = f.stem
        try:
            df = pd.read_csv(f)
            total_damage = df['value'].sum() if 'value' in df.columns else 0.0
        except Exception as e:
            print(f"  Warning: Could not read {f.name}: {e}")
            total_damage = 0.0
        
        events.append({
            'event_id': event_id,
            'total_damage_usd': total_damage
        })
    
    events_df = pd.DataFrame(events)
    events_df.to_csv(output_path, index=False)
    
    n_nonzero = (events_df['total_damage_usd'] > 0).sum()
    print(f"  Saved {len(events_df)} events to: {output_path}")
    print(f"  Events with damage >$0: {n_nonzero}")
    
    return events_df


def create_annual_frequencies_csv(freq, freqyear, output_path):
    """Create annual_frequencies.csv from freqyear array."""
    print(f"\nCreating annual_frequencies.csv...")
    
    freq_df = pd.DataFrame({
        'year_index': range(len(freqyear)),
        'frequency': freqyear,
        'mean_frequency': freq
    })
    
    freq_df.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")
    
    return freq_df


def main():
    parser = argparse.ArgumentParser(description="Generate Emanuel metadata files")
    parser.add_argument("--event_set", type=str, default="FL_era5_reanalcal",
                        help="Event set name (default: FL_era5_reanalcal)")
    parser.add_argument("--tracks_dir", type=Path, default=DEFAULT_TRACKS_DIR,
                        help="Directory containing track .mat files")
    
    args = parser.parse_args()
    
    print("="*70)
    print(f"GENERATING EMANUEL METADATA: {args.event_set}")
    print("="*70)
    
    # Paths
    event_dir = EMANUEL_DIR / args.event_set
    track_name = get_track_filename(args.event_set)
    track_file = args.tracks_dir / track_name
    
    all_events_path = event_dir / "all_events.csv"
    freq_path = event_dir / "annual_frequencies.csv"
    
    print(f"\nPaths:")
    print(f"  Event directory: {event_dir}")
    print(f"  Track file: {track_file}")
    print(f"  Output all_events: {all_events_path}")
    print(f"  Output frequencies: {freq_path}")
    
    # Check event directory
    if not event_dir.exists():
        raise FileNotFoundError(f"Event directory not found: {event_dir}")
    
    # Load freqyear from tracks
    freq, freqyear = load_freqyear_from_tracks(track_file)
    
    # Create all_events.csv
    create_all_events_csv(event_dir, all_events_path)
    
    # Create annual_frequencies.csv
    create_annual_frequencies_csv(freq, freqyear, freq_path)
    
    print("\n" + "="*70)
    print("METADATA GENERATION COMPLETE")
    print("="*70)
    print(f"\nNext steps:")
    print(f"  1. Generate year-sets:")
    print(f"     python scripts/hazard/generate_emanuel_year_sets.py --event_set {args.event_set}")
    print(f"  2. Run Monte Carlo:")
    print(f"     python scripts/run/run_emanuel_monte_carlo.py --event_set {args.event_set}")


if __name__ == "__main__":
    main()

# Emanuel TC Monte Carlo - Quick Start

## Overview

Run Monte Carlo insurance risk analysis using Emanuel TC event sets (ERA5 historical + 25 GCM future climate scenarios).

**Scripts:**
- `run_emanuel_monte_carlo.py` - Main Monte Carlo runner
- `submit_emanuel_mc_sherlock.sh` - Sherlock SLURM submission

---

## Prerequisites

**1. Precomputed impacts** (all events including zero-damage):
```bash
# Check for impact directory
ls /scratch/groups/bakerjw/smeiler/impacts/FL_era5_reanalcal/
# Should contain: all_events.csv, event_metadata.csv, annual_frequencies.csv, ~8800 event CSVs
```

**2. Generated year-sets** (10,000-year stochastic catalogs):
```bash
# Check for year-sets file
ls /scratch/groups/bakerjw/smeiler/impacts/FL_era5_reanalcal/year_sets_N10000_seed42.csv
```

If missing, generate first:
```bash
# For ERA5
sbatch submit_generate_emanuel_yearsets_sherlock.sh FL_era5_reanalcal

# For all 26 event sets
sbatch --array=0-25 submit_generate_emanuel_yearsets_sherlock.sh
```

---

## Usage

### Local Testing (Small n_years)
```bash
# ERA5 baseline (1000 years for quick test)
python run_emanuel_monte_carlo.py \
    --event_set FL_era5_reanalcal \
    --n_years 1000 \
    --impact_dir results/impacts/FL_era5_reanalcal

# ERA5 with building codes policy
python run_emanuel_monte_carlo.py \
    --event_set FL_era5_reanalcal \
    --policy building_codes_major \
    --n_years 1000
```

### Sherlock Production Runs

**Single event set (ERA5 historical baseline):**
```bash
sbatch submit_emanuel_mc_sherlock.sh FL_era5_reanalcal
```

**Single event set with policy:**
```bash
sbatch submit_emanuel_mc_sherlock.sh FL_era5_reanalcal building_codes_major
```

**Future climate scenario:**
```bash
sbatch submit_emanuel_mc_sherlock.sh FL_canesm5_ssp585_2081-2100
```

**All 26 event sets in parallel (array job):**
```bash
sbatch --array=0-25 submit_emanuel_mc_sherlock.sh
```

---

## Event Sets Available

| ID | Event Set Name | Description | Years | Events |
|----|----------------|-------------|-------|--------|
| 0 | `FL_era5_reanalcal` | ERA5 historical | 44 | ~8,800 |
| 1-5 | `FL_canesm5_*` | CanESM5 baseline + SSP2-4.5/SSP5-8.5 | 20 | ~3,600 |
| 6-10 | `FL_cnrm-cm6-1_*` | CNRM-CM6-1 baseline + SSP2-4.5/SSP5-8.5 | 20 | ~3,700 |
| 11-15 | `FL_ec-earth3_*` | EC-Earth3 baseline + SSP2-4.5/SSP5-8.5 | 20 | ~4,000 |
| 16-20 | `FL_ipsl-cm6a-lr_*` | IPSL-CM6A-LR baseline + SSP2-4.5/SSP5-8.5 | 20 | ~3,800 |
| 21-25 | `FL_miroc6_*` | MIROC6 baseline + SSP2-4.5/SSP5-8.5 | 20 | ~3,900 |

---

## Policy Scenarios

Available policy scenarios (from `fl_risk_model/config.py`):
- `baseline` - No policy changes
- `market_exit_minor` - 10% private market exit
- `market_exit_moderate` - 25% private market exit
- `market_exit_major` - 50% private market exit
- `penetration_minor` - 10% NFIP penetration increase
- `penetration_moderate` - 25% NFIP penetration increase
- `penetration_major` - 50% NFIP penetration increase
- `building_codes_minor` - 10% damage reduction
- `building_codes_moderate` - 25% damage reduction
- `building_codes_major` - 40% damage reduction
- `catbond_minor` - $2B cat bond
- `catbond_moderate` - $5B cat bond
- `catbond_major` - $10B cat bond

---

## Output Structure

Results saved to `/scratch/groups/bakerjw/smeiler/results/emanuel_mc_runs/`:

```
emanuel_era5_baseline_20240115_143022/
├── run_config.json              # Configuration snapshot
├── iterations.csv               # Year-by-year results (10,000 rows)
├── return_period_metrics.csv    # RP10, RP100, RP250, etc.
└── errors_summary.txt           # Any errors encountered
```

**Key output columns in iterations.csv:**
- `year_id` - Simulated year (1-10,000)
- `total_damage_usd` - Total county-level damage
- `wind_total_usd` / `water_total_usd` - Hazard split
- `defaults_post` - Number of insurer defaults
- `figa_residual_deficit_usd` - FIGA fund exhaustion
- `citizens_residual_deficit_usd` - Citizens deficit
- `nfip_borrowed_usd` - NFIP borrowing from Treasury

---

## Monitoring Jobs

**Check job status:**
```bash
squeue -u $USER | grep emanuel_mc
```

**Check logs:**
```bash
tail -f ~/logs/emanuel_mc_<JOB_ID>_<ARRAY_ID>.out
```

**Array job summary:**
```bash
sacct -j <JOB_ID> --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode
```

---

## Typical Resource Usage

- **Time:** 12-24 hours for 10,000 years (full modeling)
- **Memory:** 64-96 GB
- **Storage:** ~500 MB per event set
- **Fast-path:** Set `--fast_threshold 5e9` to skip detailed modeling for low-damage years (<$5B)

---

## Next Steps

1. **Generate ERA5 year-sets** (if not done):
   ```bash
   sbatch submit_generate_emanuel_yearsets_sherlock.sh FL_era5_reanalcal
   ```

2. **Run ERA5 baseline MC** (historical climate, no policy changes):
   ```bash
   sbatch submit_emanuel_mc_sherlock.sh FL_era5_reanalcal
   ```

3. **Compare with synthetic TC baseline** (sanity check):
   - Check mean annual damage
   - Compare RP100 losses
   - Validate default rates

4. **Expand to policy scenarios**:
   ```bash
   sbatch submit_emanuel_mc_sherlock.sh FL_era5_reanalcal building_codes_major
   sbatch submit_emanuel_mc_sherlock.sh FL_era5_reanalcal market_exit_moderate
   ```

5. **Expand to future climate scenarios**:
   ```bash
   # Generate year-sets for all 26 event sets
   sbatch --array=0-25 submit_generate_emanuel_yearsets_sherlock.sh
   
   # Run MC for all event sets (baseline policy)
   sbatch --array=0-25 submit_emanuel_mc_sherlock.sh
   ```

---

## Troubleshooting

**Error: "Year-sets file not found"**
```bash
# Generate year-sets first
sbatch submit_generate_emanuel_yearsets_sherlock.sh FL_era5_reanalcal
```

**Error: "Event metadata not found"**
```bash
# Precompute impacts first
sbatch --array=0 submit_precompute_emanuel_sherlock.sh
```

**Out of memory:**
- Increase `#SBATCH --mem=192G`
- Use fast-path: `--fast_threshold 5e9`

**Slow execution:**
- Current: Full modeling for all years (~20h for 10,000 years)
- Fast-path: Skip detailed model for low-damage years
- Partial run: Use `--n_years 5000` for testing

---

## Technical Notes

**Event sampling:**
- Uses empirical frequency resampling (bootstrap with replacement from freqyear)
- Preserves underdispersion (variance/mean ≈ 0.12)
- Includes ALL events (zero-damage, low-damage, high-damage)

**Wind/water split:**
- Stochastic per event based on county-level historical ratios
- Random seed controls split variability (default: 42)

**Insurance waterfall:**
- Primary insurers → FHCF → FIGA → Cat bonds
- Citizens separate with federal backstop (NFIP similar)
- Full capital model with surplus/deficit tracking

**Climate scaling:**
- `--climate_scaling` flag applies 2050 SSP2-4.5 damage scaling
- Use for sensitivity analysis (not needed for Emanuel GCM scenarios which have physical climate changes)

---

## Related Documentation

- `EMANUEL_TC_WORKFLOW.md` - Full preprocessing pipeline
- `EMANUEL_DISTRIBUTION_FIX.md` - Why we keep all events
- `fl_risk_model/COMPOSITE_SCENARIOS_README.md` - Multi-event scenario framework
- `documentation/STOCHASTIC_POLICY_GUIDE.md` - Policy scenario definitions

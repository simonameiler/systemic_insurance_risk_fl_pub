# Building Codes Analysis - Rerun & Sensitivity

This directory contains scripts to:
1. **Rerun** the ERA5 building codes baseline (fixing December 2025 bug)
2. **Run** comprehensive climate × building codes sensitivity analysis

## Background: December 2025 Bug

The original December 2025 policy runs had a bug where building codes were **only applied to wind damage**, not flood damage. This caused:
- NFIP metrics to be **identical** between Baseline and Building Codes scenarios
- Public burden metrics to be **partially incorrect**

**Fix**: On January 31, 2026 (commit 629caf1), flood damage reduction was added to building codes.

## Quick Start

### On Sherlock:

```bash
# Navigate to project directory
cd ~/repos/systemic_insurance_risk_fl

# Make submission script executable
chmod +x scripts/cluster/submit_building_codes_analysis.sh

# Submit jobs (interactive menu)
./scripts/cluster/submit_building_codes_analysis.sh
```

### Local Testing:

```bash
# Test baseline rerun
python scripts/run/rerun_building_codes_baseline.py --seed 42

# Test sensitivity (one level)
python scripts/run/run_climate_buildingcode_sensitivity_windfloods.py \
    --code_level 2 \
    --climate era5 \
    --seed 42
```

## Job Details

### 1. Building Codes Baseline Rerun

**Script**: `submit_building_codes_rerun.sh`

- **Purpose**: Rerun ERA5 + Building Codes MAJOR with correct implementation
- **Jobs**: 1
- **Time**: ~12 hours
- **Parameters**:
  - Wind reduction: 30%
  - Flood reduction: 25% (FIXED - was 0% in Dec 2025)
  - Retrofit rate: 50%
  - Timeline: 16 years

**Expected outcome**: NFIP stress should **decrease** compared to baseline.

### 2. Climate × Building Codes Sensitivity Analysis

**Script**: `submit_climate_buildingcode_sensitivity_windfloods.sh`

- **Purpose**: Test how building codes interact with climate change
- **Jobs**: 18 (array job with 18 tasks)
- **Time**: ~12 hours each (run in parallel)
- **Design**:
  - 6 building code levels
  - 3 climate scenarios
  - SSP5-8.5 pathway (worst case)

**Building Code Levels** (maintaining 3:2 wind:flood ratio):

| Level | Wind | Flood | Description |
|-------|------|-------|-------------|
| 0 | 0% | 0% | No codes (pure climate effect) |
| 1 | 20% | 13% | Low codes |
| 2 | 30% | 20% | Current strong codes (MAJOR) |
| 3 | 40% | 27% | Enhanced codes |
| 4 | 50% | 33% | Very strong codes |
| 5 | 60% | 40% | Aggressive future codes |

**Climate Scenarios**:
- ERA5 baseline (current climate)
- SSP5-8.5 2050 (near-term)
- SSP5-8.5 2100 (mid-term)

**Scientific Hypothesis**:
- **H0**: Building code effectiveness remains constant regardless of climate intensity
- **H1**: Building code effectiveness degrades as climate intensifies

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# Monitor specific job
squeue -u $USER -j <JOB_ID>

# Watch baseline log
tail -f logs/building_codes_rerun_<JOB_ID>.out

# Watch sensitivity log (task 0)
tail -f logs/climate_bc_sens_<JOB_ID>_0.out

# Check all sensitivity tasks
ls -lh logs/climate_bc_sens_<JOB_ID>_*.out
```

## After Completion

### 1. Verify Results

```bash
# List completed runs
ls -lhd results/mc_runs/era5_building_codes_major_rerun_*
ls -lhd results/mc_runs/*_buildingcode_sens_*

# Check metadata
cat results/mc_runs/era5_building_codes_major_rerun_*/sensitivity_metadata.json
```

### 2. Compare NFIP Metrics

```bash
# Run analysis script
conda run -n climada_env python check_nfip_data.py
```

Expected results:
- **Baseline**: NFIP claims ~$0.652B, stress 5.27%
- **Building Codes (OLD - Dec 2025)**: NFIP claims ~$0.652B, stress 5.27% (BUGGY - identical!)
- **Building Codes (NEW - rerun)**: NFIP claims should be **lower**, stress should **decrease**

### 3. Update Analysis Notebook

Open `notebooks/emanuel_tc_policy_analysis.ipynb` and:
1. Update Section 4 with corrected building codes results
2. Add Section 5 sensitivity analysis plots
3. Test climate × building codes interactions

## File Structure

```
scripts/
├── run/
│   ├── rerun_building_codes_baseline.py          # ERA5 baseline rerun
│   └── run_climate_buildingcode_sensitivity_windfloods.py  # Sensitivity sweep
└── cluster/
    ├── submit_building_codes_rerun.sh            # SBATCH for baseline
    ├── submit_climate_buildingcode_sensitivity_windfloods.sh  # SBATCH for sensitivity
    └── submit_building_codes_analysis.sh         # Master submission script
```

## Parameters Reference

### Building Codes MAJOR Preset (Standard)
```python
{
    "wind_loss_reduction": 0.30,    # 30% wind reduction
    "flood_loss_reduction": 0.25,   # 25% flood reduction
    "retrofit_rate": 0.50,          # 50% of buildings retrofitted
    "timeline_years": 16,           # Over 16 years
}
```

### Sensitivity Levels (Custom)
```python
LEVELS = [
    {"wind": 0.00, "flood": 0.00},  # L0: No codes
    {"wind": 0.20, "flood": 0.13},  # L1: Low
    {"wind": 0.30, "flood": 0.20},  # L2: Current (≈MAJOR)
    {"wind": 0.40, "flood": 0.27},  # L3: Enhanced
    {"wind": 0.50, "flood": 0.33},  # L4: Very strong
    {"wind": 0.60, "flood": 0.40},  # L5: Aggressive future
]
```

## Troubleshooting

**Job fails immediately**:
- Check Python environment: `conda activate climada_env`
- Verify hazard files: `python scripts/run/check_hazard_paths.py`

**Out of memory**:
- Default: 32GB should be sufficient
- If needed, increase in SBATCH header: `#SBATCH --mem=64G`

**Job runs but produces no output**:
- Check error logs: `logs/climate_bc_sens_<JOB_ID>_<TASK>.err`
- Verify MC run completed: Check `results/mc_runs/` for new directories

**Sensitivity results look wrong**:
- Verify metadata: `cat results/mc_runs/*/sensitivity_metadata.json`
- Check that wind/flood reductions match expected levels
- Confirm climate scenario in run label matches intent

## Contact

For questions about this analysis, see:
- Bug documentation: `NFIP_BUILDING_CODES_BUG_ANALYSIS.md`
- Implementation: `fl_risk_model/mc_run_events.py` (lines 1000-1025)
- Code history: `git log -S "water_df_bc"`

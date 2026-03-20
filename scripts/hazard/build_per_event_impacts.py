#!/usr/bin/env python3
"""
build_per_event_impacts.py - Build per-event impacts from Gori et al. hazard mats

Inputs (paths can be changed below):
- maxwindmat_ncep_reanal.mat    -> variable: maxwindmat  [counties x events], wind speed (m/s)
- maxelev_coastcounty_ncep_reanal.mat -> variables: scounty (m MSL), scounty_mhhw (m MHHW) [counties x events]
- ptot_rain_county_ncep_reanal.mat -> variable: ptot_mat [counties x events], rainfall total (mm)
- county_region.csv -> columns include: county_index, stcode (=12 for FL), ccode (=countyfp), fips
- (optional) names_source_csv -> a CSV with columns countyfp, county_name (e.g., /mnt/data/1926255N15314.csv)
- (optional) gdp_csv -> a CSV mapping county FIPS or countyfp to GDP values used in the damage model (units at your discretion)

Outputs:
- Either a single combined CSV with all events, or one CSV per event (see config).
- Columns: event_id, countyfp, county_name, value, wind_share, flood_share, W_ms, R_mm, S_m

Notes:
- "value" by default is an IMPACT INDEX built from a log-linear form mirroring the Gori model structure:
    value = GDP^b_gdp * W_ms^b_wind * (R_mm+1)^b_rain * exp(b_surge * S_m)
  If no GDP is provided, GDP=1 is used.
  You can set the exponents (b_wind, b_rain, b_surge, b_gdp) below or pass them at runtime via CLI.
- If you have the exact regression coefficients for your region/setting, set them here to reproduce dollar damages.
- wind_share / flood_share are apportioned from the *hazard contributions* to the log-impact:
    c_wind = b_wind * ln(max(W_ms, eps))
    c_rain = b_rain * ln(R_mm + 1)
    c_surge = b_surge * S_m         (already linear in the model)
    flood = c_rain + c_surge
    shares = c_wind / (c_wind + flood), flood / (c_wind + flood)
  If both c_wind and flood are zero, shares fall back to (0.0, 0.0).

Citations for data structures:
- Wind: 'maxwindmat' variable gives county x event peak winds (m/s). (Wind_readme.txt) 
- Surge: 'scounty' / 'scounty_mhhw' give county x event peak storm tide; non-coastal counties are 0. (Surge_readme.txt) 
- Rain: 'ptot_mat' gives total rainfall (mm) per county x event. (Rain_readme.txt)
"""

from __future__ import annotations
import argparse, os, math
import numpy as np
import pandas as pd
from typing import Tuple
from scipy.io import loadmat

# ---------------- Configuration defaults ----------------
DEFAULTS = {
    "wind_mat_path": "/mnt/data/maxwindmat_ncep_reanal.mat",
    "surge_mat_path": "/mnt/data/maxelev_coastcounty_ncep_reanal.mat",
    "rain_mat_path":  "/mnt/data/ptot_rain_county_ncep_reanal.mat",
    "county_region_csv": "/mnt/data/county_region.csv",
    "names_source_csv": "/mnt/data/1926255N15314.csv",
    "gdp_mat_path": "/mnt/data/county_gdp_1996_2020.mat",
    "use_surge_var": "scounty_mhhw",
    "florida_stcode": 12,
    "output_combined_csv": "/mnt/data/fl_per_event_impacts.csv",
    "output_dir_per_event": None,
    "b_wind": 1.0,
    "b_rain": 1.0,
    "b_surge": 1.0,
    "b_gdp": 1.0,
    "clip_nonpositive_wind_to": 0.1,
    "epsilon": 1e-9,
}

# ---------------- Threshold-aware impact computation ----------------
def compute_value_and_shares(
    W_ms: np.ndarray, R_mm: np.ndarray, S_m: np.ndarray,
    GDP: np.ndarray, b_wind: float, b_rain: float, b_surge: float, b_gdp: float,
    eps: float, clip_wind_to: float,
    wind_thresh: float, rain_thresh: float, surge_thresh: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Only hazards exceeding their thresholds contribute to log-impact and shares.
    If no hazards exceed thresholds, returns value=0, shares=0.
    """
    mW = W_ms > wind_thresh
    mR = R_mm > rain_thresh
    mS = S_m  > surge_thresh

    W_use = np.maximum(W_ms, clip_wind_to)
    R_use = np.maximum(R_mm, 0.0)
    S_use = np.maximum(S_m, 0.0)

    log_value = b_gdp * np.log(np.maximum(GDP, eps))
    log_value = np.where(mW, log_value + b_wind * np.log(W_use), log_value)
    log_value = np.where(mR, log_value + b_rain * np.log(R_use + 1.0), log_value)
    log_value = np.where(mS, log_value + b_surge * S_use, log_value)

    value = np.exp(log_value)

    c_wind  = np.where(mW, b_wind * np.log(W_use), 0.0)
    c_rain  = np.where(mR, b_rain * np.log(R_use + 1.0), 0.0)
    c_surge = np.where(mS, b_surge * S_use, 0.0)

    c_flood = c_rain + c_surge
    denom = c_wind + c_flood
    with np.errstate(divide="ignore", invalid="ignore"):
        wind_share = np.where(denom > 0, c_wind / denom, 0.0)
        flood_share = np.where(denom > 0, c_flood / denom, 0.0)

    wind_share = np.clip(wind_share, 0.0, 1.0)
    flood_share = np.clip(flood_share, 0.0, 1.0)

    return value, wind_share, flood_share

# ---------------- Main pipeline ----------------
def main(**kwargs):
    cfg = DEFAULTS.copy()
    cfg.update({k:v for k,v in kwargs.items() if v is not None})

    # Load hazard mats and normalize orientations
    wind = loadmat(cfg["wind_mat_path"])["maxwindmat"]  # (5018, 3220)
    surge_raw = loadmat(cfg["surge_mat_path"])[cfg["use_surge_var"]]
    rain = loadmat(cfg["rain_mat_path"])["ptot_mat"]

    gdp_mat = loadmat(cfg["gdp_mat_path"])
    county_gdp_all = gdp_mat["county_gdp"]
    n_counties_total = county_gdp_all.shape[0]  # 3220
    n_events_total = surge_raw.shape[1]

    W_all = wind.T  # transpose to (3220, 5018)
    S_all = surge_raw
    R_all = rain

    # Florida subset
    reg = pd.read_csv(cfg["county_region_csv"])
    fl = reg.loc[reg["stcode"] == cfg["florida_stcode"]].copy()
    fl_idx = fl["county_index"].astype(int).values
    countyfp = fl["ccode"].astype(int).values
    names = pd.read_csv(cfg["names_source_csv"]).set_index("countyfp")["county_name"].to_dict()

    W_fl, S_fl, R_fl = W_all[fl_idx, :], S_all[fl_idx, :], R_all[fl_idx, :]
    GDP_arr = county_gdp_all[fl_idx, -1].astype(float)

    # Thresholds
    wind_thresh, rain_thresh, surge_thresh = 25.0, 0.0, 0.0
    exceed_mask = (W_fl > wind_thresh) | (R_fl > rain_thresh) | (S_fl > surge_thresh)

    GDP_mat = GDP_arr.reshape(-1, 1) * np.ones((1, n_events_total), dtype=float)

    value, wind_share, flood_share = compute_value_and_shares(
        W_fl, R_fl, S_fl, GDP_mat,
        cfg["b_wind"], cfg["b_rain"], cfg["b_surge"], cfg["b_gdp"],
        cfg["epsilon"], cfg["clip_nonpositive_wind_to"],
        wind_thresh, rain_thresh, surge_thresh
    )

    value = value * exceed_mask
    wind_share = wind_share * exceed_mask
    flood_share = flood_share * exceed_mask

    # Build combined DataFrame
    records = []
    for j in range(n_events_total):
        ev_id = int(j + 1)
        df_ev = pd.DataFrame({
            "event_id": ev_id,
            "countyfp": countyfp,
            "county_name": [names.get(int(c), "") for c in countyfp],
            "value": value[:, j],
            "wind_share": wind_share[:, j],
            "flood_share": flood_share[:, j],
            "W_ms": W_fl[:, j],
            "R_mm": R_fl[:, j],
            "S_m": S_fl[:, j],
        })
        # 🔹 Remove zero-impact rows here
        df_ev = df_ev[df_ev["value"] > 0]
        if not df_ev.empty:
            records.append(df_ev)

    combined = pd.concat(records, ignore_index=True)
    if cfg["output_combined_csv"]:
        combined.to_csv(cfg["output_combined_csv"], index=False)

    if cfg["output_dir_per_event"]:
        os.makedirs(cfg["output_dir_per_event"], exist_ok=True)
        for ev_id, df_ev in combined.groupby("event_id", sort=True):
            out_path = os.path.join(cfg["output_dir_per_event"], f"FL_event_{ev_id:05d}.csv")
            df_ev.to_csv(out_path, index=False)

    print(f"Wrote {len(combined):,} rows for {n_events_total} events and {len(fl_idx)} FL counties.")
    print(f"Removed zero-impact rows inside the script.")
    if cfg["output_combined_csv"]:
        print(f"- Combined CSV: {cfg['output_combined_csv']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build per-event impacts for Florida counties.")
    args = parser.parse_args()
    main(**vars(args))


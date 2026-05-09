#!/usr/bin/env python3
"""
simulate_historical_event_losses.py
=====================================
Compute county-level TC wind damage for the four historical hurricane scenarios
used in this study and write the pre-computed impact footprints to
fl_risk_model/data/hazard/historical_events/.

Events
------
  1926255N15314  Great Miami Hurricane (1926)
  1928250N14343  Lake Okeechobee Hurricane (1928)
  1992230N11325  Hurricane Andrew (1992)
  2017242N16333  Hurricane Irma (2017)

Methodology
-----------
1. Build Florida economic exposure from LitPop (GDP × nightlight proxy,
   fin_mode='pc', 120 arc-second resolution, reference year 2024).
2. Download historical tracks from IBTrACS via CLIMADA and interpolate to
   30-minute time steps.
3. Compute surface wind fields using CLIMADA's Holland (2008) parametric
   wind model (the CLIMADA default in TropCyclone.from_tracks).
4. Apply RMSF-calibrated regional TC impact functions
   (ImpfSetTropCyclone.from_calibrated_regional_ImpfSet, region 2 = USA).
5. Aggregate centroid-level damages to Florida counties via spatial join with
   the US county shapefile (fl_risk_model/data/US_counties).
6. Save one CSV per event with columns [countyfp, county_name, value].

Usage
-----
    conda activate climada_env
    python scripts/hazard/simulate_historical_event_losses.py

    # Save to a custom directory:
    python scripts/hazard/simulate_historical_event_losses.py --out_dir /tmp/footprints

Requirements
------------
    CLIMADA >= 4.0  (conda activate climada_env)
    Internet access for the first IBTrACS download (~480 MB cached locally)
    fl_risk_model/data/US_counties  (county shapefile)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

# ---------------------------------------------------------------------------
# Repository path bootstrap
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from climada.hazard import Centroids, TCTracks, TropCyclone
    from climada.entity.exposures import Exposures
    from climada.entity import LitPop
    from climada.entity.impact_funcs.trop_cyclone import ImpfSetTropCyclone
    from climada.engine import ImpactCalc
except ImportError:
    sys.exit(
        "\n[simulate_historical_event_losses] ERROR: CLIMADA is not installed.\n"
        "Activate the correct environment:  conda activate climada_env\n"
    )

from fl_risk_model import config as cfg


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _p(x):
    return x if isinstance(x, Path) else Path(x)


DATA_DIR = _p(getattr(cfg, "DATA_DIR", REPO_ROOT / "fl_risk_model" / "data"))

# Florida bounding box (used for centroids grid)
FL_BOUNDS = dict(min_lat=24.0, max_lat=31.5, min_lon=-90.0, max_lon=-79.0)

# LitPop settings — must match the exposure used for Emanuel event sets
LITPOP_RES_ARCSEC = 120          # 120'' ≈ 3.7 km
LITPOP_FIN_MODE   = "pc"         # purchasing-power proxy
LITPOP_REF_YEAR   = 2024

# IBTrACS IDs for the four historical scenarios
EVENTS = {
    "1926255N15314": "Great Miami Hurricane (1926)",
    "1928250N14343": "Lake Okeechobee Hurricane (1928)",
    "1992230N11325": "Hurricane Andrew (1992)",
    "2017242N16333": "Hurricane Irma (2017)",
}


# ---------------------------------------------------------------------------
# Step 1 — Florida exposure
# ---------------------------------------------------------------------------

def load_florida_exposure(cache_path: Path) -> Exposures:
    """Load (or create and cache) Florida LitPop exposure at 120 arc-seconds."""
    if cache_path.exists():
        print(f"  Loading cached exposure: {cache_path.name}")
        exp = Exposures.from_hdf5(cache_path)
    else:
        print(f"  Building LitPop exposure  "
              f"(fin_mode={LITPOP_FIN_MODE!r}, {LITPOP_RES_ARCSEC}'', {LITPOP_REF_YEAR}) ...")
        usa = LitPop.from_countries(
            "USA",
            fin_mode=LITPOP_FIN_MODE,
            res_arcsec=LITPOP_RES_ARCSEC,
            exponents=(1, 1),
            admin1_calc=True,
            reference_year=LITPOP_REF_YEAR,
        )
        exp = Exposures()
        exp.set_gdf(usa.gdf[usa.gdf.admin1 == "Florida"])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        exp.write_hdf5(cache_path)
        print(f"  Cached to: {cache_path}")

    # Impact function region: 2 = USA (RMSF calibration)
    exp.gdf["impf_TC"] = 2
    print(f"  Exposure points (Florida): {len(exp.gdf):,}  "
          f"  Total value: ${exp.gdf['value'].sum() / 1e12:.2f} T")
    return exp


# ---------------------------------------------------------------------------
# Step 2 — Centroids
# ---------------------------------------------------------------------------

def build_florida_centroids() -> Centroids:
    """Create a regular centroids grid over Florida at 120 arc-second resolution."""
    cent = Centroids.from_pnt_bounds(
        (FL_BOUNDS["min_lon"], FL_BOUNDS["min_lat"],
         FL_BOUNDS["max_lon"], FL_BOUNDS["max_lat"]),
        res=LITPOP_RES_ARCSEC / 3600,
    )
    print(f"  Centroids: {cent.size:,}")
    return cent


# ---------------------------------------------------------------------------
# Step 3 — IBTrACS tracks
# ---------------------------------------------------------------------------

def load_tracks(storm_ids: list[str]) -> TCTracks:
    """Download historical tracks from IBTrACS and interpolate to 30-min steps."""
    print(f"  Fetching {len(storm_ids)} track(s) from IBTrACS ...")
    tracks = TCTracks.from_ibtracs_netcdf(storm_id=storm_ids)
    tracks.equal_timestep(time_step_h=0.5)
    print(f"  Tracks loaded: {tracks.size}")
    return tracks


# ---------------------------------------------------------------------------
# Step 4 — Wind hazard (Holland 2008, CLIMADA default)
# ---------------------------------------------------------------------------

def compute_wind_hazard(tracks: TCTracks, centroids: Centroids) -> TropCyclone:
    """Compute surface wind fields using CLIMADA's Holland (2008) parametric model."""
    print("  Computing wind fields (Holland 2008, CLIMADA default) ...")
    haz = TropCyclone.from_tracks(tracks, centroids=centroids)
    print(f"  Hazard events: {haz.size}  "
          f"  Peak intensity: {float(haz.intensity.max()):.1f} m/s")
    return haz


# ---------------------------------------------------------------------------
# Step 5 — County-level impact aggregation
# ---------------------------------------------------------------------------

def load_fl_counties(county_dir: Path) -> gpd.GeoDataFrame:
    """Load Florida county geometries from the US_counties shapefile."""
    counties = gpd.read_file(county_dir)
    fl = counties[counties["STATEFP"] == "12"].copy()
    fl["geometry"] = fl["geometry"].buffer(0)   # fix any invalid polygons
    fl = fl.to_crs("EPSG:4326")
    print(f"  Florida counties: {len(fl)}")
    return fl


def assign_exposure_to_counties(exp: Exposures, fl_counties: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Spatial-join exposure centroids to counties (done once, reused per event)."""
    if exp.gdf.crs is None:
        exp.gdf = exp.gdf.set_crs("EPSG:4326")
    else:
        exp.set_crs(crs="EPSG:4326")

    joined = gpd.sjoin(
        exp.gdf,
        fl_counties[["COUNTYFP", "NAME", "geometry"]],
        how="left",
        predicate="within",
    )
    joined = joined.drop(columns=["index_right"], errors="ignore")
    joined = joined.rename(columns={"COUNTYFP": "countyfp", "NAME": "county_name"})

    unassigned = joined["countyfp"].isna().sum()
    if unassigned:
        print(f"  Warning: {unassigned} exposure points not assigned to a county")
    return joined


def aggregate_event_to_counties(
    exp_with_county: gpd.GeoDataFrame,
    event_impact_row: np.ndarray,
) -> pd.DataFrame:
    """Aggregate a single event's centroid damages to county totals."""
    gdf = exp_with_county.copy()
    gdf["value"] = event_impact_row
    county_df = (
        gdf.groupby(["countyfp", "county_name"], dropna=False)["value"]
        .sum(min_count=1)
        .reset_index()
    )
    county_df["countyfp"] = county_df["countyfp"].astype(str).str.zfill(3)
    return county_df.sort_values("countyfp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute county-level TC wind impacts for the four historical scenarios."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(DATA_DIR / "hazard" / "historical_events"),
        help="Directory to write per-event CSVs (default: fl_risk_model/data/hazard/historical_events)",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 64)
    print("  simulate_historical_event_losses.py")
    print("=" * 64)
    print(f"  Events  : {', '.join(EVENTS.keys())}")
    print(f"  Output  : {out_dir}")
    print("=" * 64)

    # --- Exposure ---
    print("\n[1/5] Florida exposure ...")
    exp = load_florida_exposure(DATA_DIR / "FL_exposure_120as.hdf5")

    # --- Centroids ---
    print("\n[2/5] Centroids ...")
    cent = build_florida_centroids()

    # --- Tracks ---
    print("\n[3/5] IBTrACS tracks ...")
    tracks = load_tracks(list(EVENTS.keys()))

    # --- Wind hazard ---
    print("\n[4/5] Wind hazard ...")
    haz = compute_wind_hazard(tracks, cent)

    # Assign exposure centroids to the hazard grid (needed for ImpactCalc)
    exp.assign_centroids(haz)

    # --- Impact functions ---
    print("\n[5/5] Impact calculation ...")
    imp_fun_set = ImpfSetTropCyclone.from_calibrated_regional_ImpfSet(
        calibration_approach="RMSF"
    )
    imp = ImpactCalc(exp, imp_fun_set, haz).impact(save_mat=True)
    print(f"  Events with damage > $0: {(imp.at_event > 0).sum()} / {haz.size}")

    # --- County spatial join ---
    print("\n[+] Spatial join to counties ...")
    county_dir = DATA_DIR / "US_counties"
    fl_counties = load_fl_counties(county_dir)
    exp_with_county = assign_exposure_to_counties(exp, fl_counties)

    # --- Write one CSV per event ---
    print("\n[+] Writing county impact CSVs ...")
    written = 0
    for i in range(imp.at_event.size):
        event_id = imp.event_name[i]
        if event_id not in EVENTS:
            continue   # skip any extra tracks loaded by CLIMADA

        impact_row = imp.imp_mat[i, :].toarray().ravel()
        county_df  = aggregate_event_to_counties(exp_with_county, impact_row)

        out_path = out_dir / f"{event_id}.csv"
        county_df.to_csv(out_path, index=False)
        total_usd = float(county_df["value"].sum())
        print(f"  {event_id}  ({EVENTS[event_id]})  "
              f"  total=${total_usd/1e9:.1f}B  →  {out_path.name}")
        written += 1

    print(f"\n  Done — wrote {written} / {len(EVENTS)} event file(s) to {out_dir}\n")


if __name__ == "__main__":
    main()

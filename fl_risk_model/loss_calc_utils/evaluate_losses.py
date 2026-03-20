"""
evaluate_losses.py - Loss evaluation and benchmarking utilities
-----------------------------------------------------------------

Provides functions for evaluating modeled losses against FLOIR claims data
and generating diagnostic outputs.

Public API
----------
- EVENT_ID_TO_SLUG: Dict mapping IBTrACS IDs to storm slugs
- load_floir_statewide
- load_floir_county
- build_benchmarks_from_top50
- append_benchmarks_and_score
- plot_event_county_map
"""
import os
import re

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import spearmanr

# Map IBTrACS event IDs to FLOIR file prefixes
EVENT_ID_TO_SLUG = {
    '2016242N24279': 'hermine',
    '2016273N13300': 'matthew',
    '2017242N16333': 'irma',
    '2018280N18273': 'michael',
    '2019236N10314': 'dorian',
    '2019291N22264': 'nestor',
    '2020138N28281': 'arthur',
    '2020188N28271': 'fay',
    '2020211N13306': 'isaias',
    '2020256N25281': 'sally',
    '2020306N15288': 'eta',
    '2021182N09317': 'elsa',
    '2021252N28273': 'mindy',
    '2022154N21273': 'alex',
    '2022266N12294': 'ian',
    '2022311N21293': 'nicole',
    '2023239N21274': 'idalia',
    '2024214N18298': 'debby',
    '2024268N17278': 'helene',
    '2024279N21265': 'milton',
    '1992230N11325': 'andrew',
    '1926255N15314': 'great_miami_hurricane',
    '1945255N19302': 'homestead',
    '2004223N11301': 'charley',
    '2004238N11325': 'frances',
    '1921293N13280': 'tampa_bay',
    '2005289N18282': 'wilma',
    '1928250N14343': 'lake_okeechobee'
}


def load_floir_statewide(event_id, base_dir="../data/reports"):
    """
    Load and clean the FLOIR statewide summary for a given event.
    """
    slug = EVENT_ID_TO_SLUG.get(event_id)
    if slug is None:
        raise ValueError(f"Unknown event_id: {event_id}")
    path = os.path.join(base_dir, f"{slug}_claims.csv")
    df = pd.read_csv(path)
    # Clean numeric columns
    for col in df.columns[1:]:
        df[col] = (
            df[col].astype(str)
                 .str.replace(r"[^0-9.-]", "", regex=True)
                 .replace('', '0')
                 .astype(float)
        )
    return df


def load_floir_county(event_id, base_dir="../data/reports"):
    """
    Load and clean the FLOIR county-level claims data for a given event.
    Normalizes column names for closed-with-payment claims.
    """
    slug = EVENT_ID_TO_SLUG.get(event_id)
    if slug is None:
        raise ValueError(f"Unknown event_id: {event_id}")
    path = os.path.join(base_dir, f"{slug}_counties.csv")
    df = pd.read_csv(path)
    # Standardize column names
    df.columns = [col.strip() for col in df.columns]
    # Handle alternate closed-paid column names
    alt_closed = None
    for alt in [
        'Number of Claims Closed with Payment',
        'Closed Claims with Payment',
        'Closed Claims (Paid)'
    ]:
        if alt in df.columns:
            alt_closed = alt
            break
    if alt_closed and alt_closed != 'Number of Claims Closed with Payment':
        df.rename(columns={alt_closed: 'Number of Claims Closed with Payment'}, inplace=True)
    # Clean numeric columns except the first
    for col in df.columns[1:]:
        df[col] = (
            df[col].astype(str)
                 .str.replace(r"[^0-9.-]", "", regex=True)
                 .replace('', '0')
                 .astype(float)
        )
    # Standardize county names
    df['County'] = df[df.columns[0]].str.strip().str.title()
    return df

def pick_event_value_col(exp_with_county, event_id=None, event_col=None):
    """
    Return the column name to use as modeled values:
    - If `event_col` provided, return it.
    - Else prefer an exact `event_id` match if present.
    - Else raise for clarity.
    """
    if event_col and event_col in exp_with_county.columns:
        return event_col
    if event_id and event_id in exp_with_county.columns:
        return event_id
    raise ValueError("Provide `event_col` (exact column name in exp_with_county) or ensure `event_id` is a column.")


def eval_disaggregated_county(
    gdf,
    statewide_df,
    county_df,
    value_col,              # e.g. your event column in exp_with_county (modeled losses per exposure)
    county_col="county_name",
    floir_county_col="County",
    floir_totals_label="Totals",
    others_label="All Other Counties"
):
    """
    Compare modeled per-county totals against FLOIR county shares applied to statewide insured losses.

    Parameters
    ----------
    gdf : GeoDataFrame with modeled per-exposure values in `value_col` and county in `county_col`
    statewide_df : DataFrame from load_floir_statewide(...)
    county_df : DataFrame from load_floir_county(...)
    value_col : str, column in gdf with modeled values (e.g., your event name/ID column)
    county_col : str, county name column in gdf (default 'county_name')
    floir_county_col : str, county name column in county_df (default 'County')
    floir_totals_label : str, label of totals row in statewide_df (default 'Totals')
    others_label : str, bucket name for “all other” counties
    """

    # 1) Observed statewide insured total (last numeric column in 'Totals' row)
    observed_insured_state = (
        statewide_df
        .loc[statewide_df.iloc[:, 0].str.strip().str.lower() == floir_totals_label.lower(), statewide_df.columns[-1]]
        .values[0]
    )

    # 2) County claim shares from FLOIR county file
    claims = (
        county_df
        .assign(**{floir_county_col: county_df[floir_county_col].str.strip().str.title()})
        .set_index(floir_county_col)['Number of Claims']
    )
    share = claims / claims.sum()
    obs_by_county_insured = share * observed_insured_state  # insured-only by county

    # 3) Modeled per-county totals from your exposures
    gdf = gdf.copy()
    gdf[county_col] = gdf[county_col].str.strip().str.title()
    mod_by_county = gdf.groupby(county_col)[value_col].sum()

    # 4) Align & add “All Other Counties” bucket
    #    -> Keep only counties present in FLOIR; bucket the rest into 'others_label'
    df = pd.DataFrame({
        'observed_insured': obs_by_county_insured,
        'modeled': mod_by_county
    }).fillna(0)

    # modeled “others” = counties not in FLOIR top list
    others_modeled = mod_by_county.loc[~mod_by_county.index.isin(obs_by_county_insured.index)].sum()

    # Keep only FLOIR counties, then append the “others” row
    df = df.loc[obs_by_county_insured.index].copy()
    df.loc[others_label, 'modeled'] = others_modeled

    # observed “others” is any residual after summing named counties (should be 0 if FLOIR has all counties)
    residual_obs = observed_insured_state - obs_by_county_insured.sum()
    if abs(residual_obs) > 1e-6:
        df.loc[others_label, 'observed_insured'] = residual_obs

    # 5) Error metrics on insured basis
    df['error_abs_insured'] = df['modeled'] - df['observed_insured']
    df['error_pct_insured'] = df['error_abs_insured'] / df['observed_insured'].replace(0, np.nan)

    # 6) Totals row
    total_mod = df['modeled'].sum()
    total_obs_insured = df['observed_insured'].sum()
    totals = pd.Series({
        'observed_insured': total_obs_insured,
        'modeled': total_mod,
        'error_abs_insured': total_mod - total_obs_insured,
        'error_pct_insured': (total_mod - total_obs_insured) / total_obs_insured if total_obs_insured > 0 else np.nan
    }, name='Totals')
    df = pd.concat([df, totals.to_frame().T], axis=0)

    # 7) Rank correlation (county distribution)
    # Exclude the totals/others when ranking (optional)
    rank_scope = df.index.difference([others_label, 'Totals'])
    rho, _ = spearmanr(df.loc[rank_scope, 'modeled'].rank(ascending=False),
                       df.loc[rank_scope, 'observed_insured'].rank(ascending=False))
    df['rank_corr'] = rho

    return df

def append_benchmarks_and_score(df_modeled: pd.DataFrame, benchmarks: pd.DataFrame,
                                storm_col: str = "Storm",
                                modeled_bn_col: str | None = "Modeled_Total_Loss_BnUSD") -> pd.DataFrame:
    """
    Merge Muller/Weinkle benchmarks into your modeled table and compute errors.
    If `Modeled_Total_Loss_BnUSD` isn't present, will try to derive it from `Modeled_Total_Loss` (/1e9).
    Returns a new DataFrame.
    """
    if storm_col not in df_modeled.columns:
        # try a lowercase variant
        if storm_col.lower() in df_modeled.columns:
            df_modeled = df_modeled.rename(columns={storm_col.lower(): storm_col})
        else:
            raise KeyError(f"'{storm_col}' column not found in df_modeled")

    df = df_modeled.copy()

    # Determine modeled value in billions
    if modeled_bn_col and modeled_bn_col in df.columns:
        bn_col = modeled_bn_col
    elif "Modeled_Total_Loss" in df.columns:
        df["Modeled_Total_Loss_BnUSD"] = df["Modeled_Total_Loss"] / 1e9
        bn_col = "Modeled_Total_Loss_BnUSD"
    else:
        raise KeyError("Provide a 'Modeled_Total_Loss_BnUSD' or 'Modeled_Total_Loss' column in your modeled df.")

    # Merge
    merged = df.merge(benchmarks, left_on=storm_col, right_index=True, how="left")

    # Error vs Muller PL22
    merged["abs_err_vs_PL22"] = merged[bn_col] - merged["Muller_PL22_USDb"]
    merged["pct_err_vs_PL22"] = merged["abs_err_vs_PL22"] / merged["Muller_PL22_USDb"]

    # Error vs Muller CL22
    merged["abs_err_vs_CL22"] = merged[bn_col] - merged["Muller_CL22_USDb"]
    merged["pct_err_vs_CL22"] = merged["abs_err_vs_CL22"] / merged["Muller_CL22_USDb"]

    # Weinkle CI inclusion
    lo = merged[["Weinkle_PL05_USDb", "Weinkle_CL05_USDb"]].min(axis=1)
    hi = merged[["Weinkle_PL05_USDb", "Weinkle_CL05_USDb"]].max(axis=1)
    merged["within_Weinkle_CI"] = (merged[bn_col] >= lo) & (merged[bn_col] <= hi)

    # Distance to CI (0 if within; else how far in Bn USD to closest bound)
    below = (merged[bn_col] < lo).astype(float) * (lo - merged[bn_col])
    above = (merged[bn_col] > hi).astype(float) * (merged[bn_col] - hi)
    merged["dist_to_Weinkle_CI_Bn"] = (below + above).fillna(np.nan)

    return merged

def build_benchmarks_from_top50(df_top50: pd.DataFrame) -> pd.DataFrame:
    targets = {
        "lake_okeechobee": (1928, "Lake Okeechobee"),
        "tampa_bay": (1921, "Tampa Bay"),
        "homestead": (1945, "Homestead"),
        "great_miami_hurricane": (1926, "Great Miami"),
        "charley": (2004, "Charley"),
        "frances": (2004, "Frances"),
        "irma": (2017, "Irma"),
        "michael": (2018, "Michael"),
        "wilma": (2005, "Wilma"),
        "hermine": (2016, "Hermine"),
        "andrew": (1992, "Andrew")
    }
    col = "Storm Year and Name"
    rows = []
    for slug, (yr, kw) in targets.items():
        mask = df_top50[col].astype(str).str.contains(str(yr), na=False) & \
               df_top50[col].astype(str).str.contains(kw, case=False, na=False)
        sub = df_top50[mask]
        if sub.empty:
            rows.append({"Storm": slug, "Muller_PL22_USDb": np.nan, "Muller_CL22_USDb": np.nan,
                         "Weinkle_PL05_USDb": np.nan, "Weinkle_CL05_USDb": np.nan})
        else:
            r = sub.iloc[0]
            rows.append({"Storm": slug,
                         "Muller_PL22_USDb": float(r["Muller et al PL22 US$B"]),
                         "Muller_CL22_USDb": float(r["Muller et al CL22 US$B"]),
                         "Weinkle_PL05_USDb": float(r["Weinkle 2018 PL05 US$B"]),
                         "Weinkle_CL05_USDb": float(r["Weinkle 2018 CL05 US$B"])})
    bmk = pd.DataFrame(rows).set_index("Storm")
    return bmk

# --- A) compute county totals for one event_id and merge with polygons ---
def county_totals_gdf(exp_with_county, fl_counties, event_id,
                      countyfp_col_exp="countyfp", countyname_col_exp="county_name",
                      countyfp_col_poly="COUNTYFP", countyname_col_poly="NAME"):
    # group modeled totals per county
    totals = (exp_with_county
              .groupby([countyfp_col_exp, countyname_col_exp], dropna=False)[event_id]
              .sum(min_count=1)
              .rename("value")
              .reset_index())

    # normalize county FIPS types/format
    totals[countyfp_col_exp] = totals[countyfp_col_exp].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(3)
    flc = fl_counties.copy()
    flc[countyfp_col_poly] = flc[countyfp_col_poly].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(3)

    # keep only needed columns & align CRS with exposures (assumes EPSG:4326 for exp.gdf)
    if flc.crs is None:
        flc = flc.set_crs(epsg=4326)  # set if you know it's WGS84
    # merge on FIPS primarily (more stable than name)
    merged = flc[[countyfp_col_poly, countyname_col_poly, "geometry"]].merge(
        totals, left_on=countyfp_col_poly, right_on=countyfp_col_exp, how="left"
    )
    # clean labels
    merged = merged.rename(columns={
        countyfp_col_poly: "countyfp",
        countyname_col_poly: "county_name"
    })
    return merged  # GeoDataFrame with 'value' column


# --- B) plot helper (single event) ---
def plot_event_county_map(exp_with_county, fl_counties, event_id,
                          title=None, cmap="cividis", logscale=True,
                          edgecolor="white", linewidth=0.6, missing_color="#f0f0f0"):
    gdf_ev = county_totals_gdf(exp_with_county, fl_counties, event_id)

    fig, ax = plt.subplots(figsize=(7, 8))
    # background counties (no data) for context
    gdf_ev.plot(ax=ax, color=missing_color, linewidth=0, zorder=0)

    values = gdf_ev["value"].to_numpy(copy=False)
    # choose normalization
    if logscale:
        # guard against non-positive
        positive = values[np.isfinite(values) & (values > 0)]
        vmin = positive.min() if positive.size else 1
        vmax = positive.max() if positive.size else 1
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        finite = values[np.isfinite(values)]
        vmin = np.nanmin(finite) if finite.size else 0
        vmax = np.nanmax(finite) if finite.size else 1
        norm = None

    # filled polygons where value is not NaN
    gdf_ev.dropna(subset=["value"]).plot(
        ax=ax, column="value", cmap=cmap, norm=norm, vmin=None if norm else vmin, vmax=None if norm else vmax,
        edgecolor=edgecolor, linewidth=linewidth, zorder=1
    )

    # county outlines on top
    gdf_ev.boundary.plot(ax=ax, color=edgecolor, linewidth=linewidth, zorder=2)

    ax.set_axis_off()
    ttl = title if title is not None else f"County Totals - {event_id}"
    ax.set_title(ttl, pad=8)

    # colorbar
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(values)
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Modeled loss (USD)")

    plt.tight_layout()
    return fig, ax


# --- C) batch plot & save many events (optional) ---
def save_event_maps(exp_with_county, fl_counties, event_ids, out_dir="outputs/event_maps",
                    **plot_kwargs):
    import os
    os.makedirs(out_dir, exist_ok=True)
    for ev in event_ids:
        fig, ax = plot_event_county_map(exp_with_county, fl_counties, ev, **plot_kwargs)
        fig.savefig(f"{out_dir}/{ev}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

# --- D) utility to auto-detect event columns in your exp_with_county ---
def detect_event_columns(exp_with_county):
    pat = re.compile(r"^\d{7}N\d{5}$")  # e.g., 2018280N18273
    return [c for c in exp_with_county.columns if isinstance(c, str) and pat.match(c)]



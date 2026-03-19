import os
import pandas as pd
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from shapely.ops import unary_union

# import CLIMADA modules:
from climada.hazard import Centroids, Hazard, TropCyclone, TCTracks
from climada.entity.exposures import Exposures
from climada.entity import LitPop

from climada.entity.impact_funcs.trop_cyclone import ImpfSetTropCyclone
from climada.engine import ImpactCalc

from fl_risk_model import config as cfg

# ---- Path coercion shim (robust even if config exports strings) -------------
def P(x):
    return x if isinstance(x, Path) else Path(x)

DATA_DIR          = P(getattr(cfg, "DATA_DIR", "data"))

#%% Load exposure data

exp = LitPop.from_countries("USA", fin_mode="pc", res_arcsec=120, exponents=(1,1), admin1_calc=True, reference_year=2024)

# subset to Florida
FL = Exposures()
FL.set_gdf(exp.gdf[exp.gdf.admin1=="Florida"])

FL.write_hdf5(os.path.join(DATA_DIR, 'FL_exposure_120as.hdf5'))

exp = Exposures.from_hdf5(os.path.join(DATA_DIR, 'FL_exposure_120as.hdf5'))

#%% add impact function set id
exp.gdf['impf_TC'] = 2 # default to region 2 (USA)

#%% Load hazard data
tc_haz = Hazard.from_hdf5(DATA_DIR.joinpath('TC_FL_120as_1980_2024_H08.hdf5'))
exp.assign_centroids(tc_haz)

#%% calculate impacts
imp_fun_set_TC = ImpfSetTropCyclone.from_calibrated_regional_ImpfSet(calibration_approach='RMSF')
imp = ImpactCalc(exp, imp_fun_set_TC, tc_haz).impact(save_mat=True)

# %%
county_fp   = DATA_DIR / "US_counties"
counties    = gpd.read_file(county_fp)
fl_counties = counties[counties["STATEFP"] == "12"]
fl_state    = fl_counties.dissolve()
# %%
for i in range(imp.at_event.size):
    if imp.at_event[i] > 0:
        exp.gdf[imp.event_name[i]] = imp.imp_mat[i, :].toarray().ravel()

# %%
# construct centroids
min_lat, max_lat, min_lon, max_lon = 24.0, 31.5, -90.0, -79.0
cent = Centroids.from_pnt_bounds((min_lon, min_lat, max_lon, max_lat), res=120/3600)
cent.plot()

# older/newest hurricanes
storm_ids = ["1926255N15314", "1945255N19302", "1921293N13280", "1928250N14343", "2024268N17278", "2024279N21265"]
hurricanes = TCTracks.from_ibtracs_netcdf(storm_id=storm_ids)
hurricanes.equal_timestep(time_step_h=0.5)

hurr_haz = TropCyclone.from_tracks(hurricanes, centroids=cent)

# %%
hurr_imp = ImpactCalc(exp, imp_fun_set_TC, hurr_haz).impact(save_mat=True)
# %%
for i in range(hurr_imp.at_event.size):
    if hurr_imp.at_event[i] > 0:
        exp.gdf[hurr_imp.event_name[i]] = hurr_imp.imp_mat[i, :].toarray().ravel()

# %%

# Ensure both layers declare the same CRS (WGS84 lon/lat)
if exp.gdf.crs is None:
    exp.gdf = exp.set_crs(crs='EPSG:4326')
else:
    exp.set_crs(crs='EPSG:4326')

if fl_counties.crs is None:
    # Only do this if you KNOW the county geometries are already in lon/lat WGS84
    fl_counties = fl_counties.set_crs(epsg=4326)
else:
    fl_counties = fl_counties.to_crs(exp.gdf.crs)

# (Optional) fix invalid polygons that can break spatial joins
fl_counties['geometry'] = fl_counties['geometry'].buffer(0)

# Spatial join: assign county attrs to each exposure point
exp_with_county = gpd.sjoin(
    exp.gdf,
    fl_counties[['COUNTYFP', 'NAME', 'geometry']],
    how='left',
    predicate='within',   # or 'intersects' if points may lie exactly on borders
)

# Clean up and rename as you like
exp_with_county = exp_with_county.drop(columns=['index_right'])
exp_with_county = exp_with_county.rename(columns={'COUNTYFP': 'countyfp', 'NAME': 'county_name'})

# %%
from fl_risk_model.loss_calc_utils.evaluate_losses import EVENT_ID_TO_SLUG

# storms of interest
storms = ["lake_okeechobee", "helene", "milton", "tampa_bay", "homestead", "great_miami_hurricane", "andrew", "charley", "frances", "irma", "michael", "wilma", "hermine"]

rows = []
for slug in storms:
    event_id = [k for k,v in EVENT_ID_TO_SLUG.items() if v == slug][0]
    total_loss = exp.gdf[event_id].sum()
    rows.append((slug, event_id, total_loss))

df = pd.DataFrame(rows, columns=["Storm", "Event_ID", "Modeled_Total_Loss"])

# Add row with totals in billions USD
totals_raw = df["Modeled_Total_Loss"].sum()

# Add a convenience column in billions
df["Modeled_Total_Loss_BnUSD"] = df["Modeled_Total_Loss"] / 1e9

print(df)

# %%
import pandas as pd
from fl_risk_model.loss_calc_utils.evaluate_losses import append_benchmarks_and_score, build_benchmarks_from_top50, plot_event_county_map

benchmarks = build_benchmarks_from_top50(pd.read_csv(DATA_DIR / "reports" / "15_Top 50 CL_1RMW.csv"))
merged = append_benchmarks_and_score(df, benchmarks)
# %%
# import os
# import re
# import pandas as pd
# from pathlib import Path

# gdf = exp_with_county.copy()

# # (optional) tidy county fields
# gdf["county_name"] = gdf["county_name"].astype(str).str.strip()
# gdf["countyfp"]    = gdf["countyfp"].astype(str).str.zfill(3)   # keep leading zeros like "001"

# # detect event-id columns (pattern like 2018280N18273)
# event_cols = [c for c in gdf.columns if re.fullmatch(r"\d{4}\d{3}N\d{5}", str(c))]

# # write one CSV per event
# for event_id in event_cols:
#     df_out = (
#         gdf.groupby(["countyfp", "county_name"], dropna=False)[event_id]
#            .sum(min_count=1)           # keep NaN if all are NaN in a county
#            .rename("value")            # column name in output
#            .reset_index()
#     )
#     df_out.to_csv(DATA_DIR / "hazard" / f"{event_id}.csv", index=False)

# print(f"Wrote {len(event_cols)} files")

# %%
fig, ax = plot_event_county_map(exp_with_county, fl_counties, event_id="1926255N15314",
                                title="Great Miami Hurricane (1926) — County totals", logscale=True)

fig, ax = plot_event_county_map(exp_with_county, fl_counties, event_id="1928250N14343",
                                title="Lake Okeechobee Hurricane (1928) — County totals", logscale=True)

fig, ax = plot_event_county_map(exp_with_county, fl_counties, event_id="1992230N11325",
                                title="Hurricane Andrew (1992) — County totals", logscale=True)
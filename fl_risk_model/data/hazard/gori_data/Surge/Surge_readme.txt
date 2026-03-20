Synthetic Surge

Storm tides (surge plus astronomical tide) are simulated by applying the synthetic wind fields to the Advanced Circulation (ADCIRC) hydrodynamic model. The ADCIRC mesh was developed in Marsooli & Lin (2018) and has varying resolution down to 1 km along the coastline. Astronomical tides are simulated using eight major tidal constituents. 

scounty  = contains the peak storm tide in m above mean sea level for each county (rows) and each synthetic storm (columns). Default value is zero (0) for non-coastal counties. Counties are listed in the "US_counties.shp" file. 

scounty_mhhw = contains the peak storm tide in m above mean higher high water for each county (rows) and each synthetic storm (columns). Default value is zero (0) for non-coastal counties. 
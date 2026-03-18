import numpy as np
import pandas as pd
from pathlib import Path

from climada.entity import ImpactFunc, ImpactFuncSet
from climada.entity.impact_funcs.trop_cyclone import ImpfTropCyclone

PATH_CSV_TC_CAPRA_BEM = Path(__file__).parent.parent / 'data' / 'CAPRA_TO_BEM_TC_WIND_IMPACT_FUNCTIONS.csv'

# =============================================================================
# Conversion from PAGER building classification to HAZUS building classification
# =============================================================================

DICT_PAGER_TCIMPF_CAPRA = {
    'C1' : 1,
    'C2' : 2,
    'C3' : 3,
    'RM2' : 4,
    'S1' : 5,
    'S4' : 6,
    'S5' : 7,
    'W' : 8,
    'A' : 9,
    'C' : 10,
    'C1H' : 11,
    'C1L' : 12,
    'C1M' : 13,
    'C2L' : 14,
    'C3L' : 15,
    'DS3' : 16,
    'INF' : 17,
    'MH' : 18,
    'PC1' : 19,
    'PC2L' : 20,
    'RM' : 21,
    'RM1L' : 22,
    'RM2L' : 23,
    'RM2M' : 24,
    'S' : 25,
    'S1H' : 26,
    'S1L' : 27,
    'S1M' : 28,
    'S2L' : 29,
    'S3' : 30,
    'S4H' : 31,
    'S4L' : 32,
    'S4M' : 33,
    'S5L' : 34,
    'UCB' : 35,
    'UFB' : 36,
    'UFB3' : 37,
    'W1' : 38,
    'W2' : 39,
    'S2' : 40,
    'A2' : 41,
    'C3M' : 42,
    'CH' : 43,
    'CL' : 44,
    'CM' : 45,
    'DS' : 46,
    'DS2' : 47,
    'MS' : 48,
    'RM1' : 49,
    'RS' : 50,
    'RS1' : 51,
    'RS2' : 52,
    'UFB1' : 53,
    'UFB2' : 54,
    'UFB4' : 55,
    'UNK' : 56,
    'PC' : 57,
    'A4' : 58,
    'M' : 59,
    'PC2' : 60,
    'W4' : 61,
    'A1' : 62,
    'C4' : 63,
    'M2' : 64,
    'RS3' : 65,
    'DS4' : 66,
    'RS4' : 67,
    'DS1' : 68,
    'RM2H' : 69,
    'RS5' : 70,
    'RE' : 71,
    'A3' : 72,
    'URM' : 73 # added manually
}

def if_from_sig_funcs_capra(bldg_type):
    """Return the impact function for a given building subtype."""
    df_impf = pd.read_csv(PATH_CSV_TC_CAPRA_BEM)
    v_thresh = df_impf[df_impf.building_type ==
                             bldg_type]['v_thresh_median'].values[0]
    v_half = df_impf[df_impf.building_type ==
                           bldg_type]['v_half_median'].values[0]

    return ImpfTropCyclone.from_emanuel_usa(
        impf_id=DICT_PAGER_TCIMPF_CAPRA[f'{bldg_type}'],
        intensity=np.arange(0, 121, 2),
        v_thresh=v_thresh, v_half=v_half, scale=1.0)


IMPF_SET_TC_CAPRA = ImpactFuncSet([if_from_sig_funcs_capra(bldg_sub)
                                   for bldg_sub in DICT_PAGER_TCIMPF_CAPRA.keys()])
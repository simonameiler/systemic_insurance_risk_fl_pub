"""Generate manuscript numerical macros directly from the validated runs."""
import argparse
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd
from publication_financial_cleanup import clean_figa_roundoff

COMPONENTS = ['figa_residual_deficit_usd', 'citizens_residual_deficit_usd', 'nfip_borrowed_usd']
GCMS = ['canesm', 'cnrm6', 'ecearth6', 'ipsl6', 'miroc6']


def load(path):
    df = pd.read_csv(path, low_memory=False)
    zero = df.scenario.eq('zero_events') if 'scenario' in df else pd.Series(False, index=df.index)
    numeric = df.select_dtypes(include='number').columns
    df.loc[zero, numeric] = df.loc[zero, numeric].fillna(0)
    clean_figa_roundoff(df)
    df['rfr'] = df[COMPONENTS].sum(axis=1)
    df['uninsured'] = df[['wind_uninsured_usd', 'wind_underinsured_usd', 'flood_un_derinsured_usd']].sum(axis=1)
    return df


def stats(df):
    return {
        'Loss': float(df.total_damage_usd.mean()/1e9),
        'Rfr': float(df.rfr.mean()/1e9),
        'Uninsured': float(df.uninsured.mean()/1e9),
        'DefaultProbability': float((df.defaults_post > 10).mean()*100),
        'DeficitProbability': float((df.largest_entity_deficit_usd > 1e9).mean()*100),
        'FigaProbability': float((df.figa_residual_deficit_usd > 0).mean()*100),
        'CitizensProbability': float((df.citizens_residual_deficit_usd > 0).mean()*100),
        'NfipProbability': float((df.nfip_claims_paid_usd > 2*df.nfip_fl_premium_base_usd).mean()*100),
        'RfrOneGdpProbability': float((df.rfr > 17e9).mean()*100),
        'RfrTenGdpProbability': float((df.rfr > 170e9).mean()*100),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    index = json.loads(args.index.read_text())
    baseline = load(index['era5_baseline'])
    base = stats(baseline)
    values = {}
    def add(key, val, precision=1):
        # Remove binary arithmetic residue before rounding displayed values.
        # Probability differences such as 0.85 pp then consistently read 0.9.
        rounded = Decimal(str(round(float(val), 10))).quantize(
            Decimal(1).scaleb(-precision), rounding=ROUND_HALF_UP)
        values[key] = {'value': float(val), 'text': format(rounded, 'f')}
    for k, v in base.items():
        add('Baseline'+k, v, 2 if k == 'RfrTenGdpProbability' else 1)
    for name, prefix in [('market_exit_moderate','Exit'), ('penetration_major','Penetration'), ('building_codes_major','Mitigation')]:
        result = stats(load(index['era5_policy_'+name]))
        for key, val in result.items():
            add(prefix+key, val, 2 if key == 'RfrTenGdpProbability' else 1)
            add(prefix+key+'Change', val-base[key])
    for period, prefix in [('ssp245cal','Mid'), ('ssp245_2cal','End')]:
        hist = [stats(load(index[f'gcm_baseline_{gcm}_20thcal'])) for gcm in GCMS]
        future = [stats(load(index[f'gcm_baseline_{gcm}_{period}'])) for gcm in GCMS]
        for key in base:
            delta = np.array([f[key]-h[key] for h,f in zip(hist,future)])
            p,lo,hi = base[key]+np.quantile(delta,[.5,.1,.9])
            if 'Probability' in key:
                p,lo,hi = np.clip([p,lo,hi],0,100)
            for suffix,val in [('',p),('Low',lo),('High',hi)]:
                add(prefix+key+suffix,val,2 if key=='RfrTenGdpProbability' else 1)
            if key in ['Loss','Rfr']:
                add(prefix+key+'Ratio',p/base[key])
    for job,prefix in [('great_miami','Miami'),('andrew','Andrew'),('lake_okeechobee','Okeechobee'),('irma','Irma'),('gm_then_andrew','MiamiAndrew'),('double_gm','DoubleMiami'),('double_irma','DoubleIrma')]:
        df=load(index['historical_'+job])
        for key,val in stats(df).items():
            add(prefix+key,val,2 if prefix=='Irma' and key=='Rfr' else 1)
        add(prefix+'UninsuredShare',100*df.uninsured.mean()/df.total_damage_usd.mean())
        prob = float((baseline.total_damage_usd > df.total_damage_usd.mean()).mean())
        add(prefix+'ReturnPeriod',1/prob,0)
    rps=[10,25,50,100,250,500,1000]
    loss=baseline.total_damage_usd.quantile([1-1/r for r in rps]).to_numpy()/1e9
    rfr=baseline.rfr.quantile([1-1/r for r in rps]).to_numpy()/1e9
    add('LossTen',loss[0]); add('LossHundred',loss[3])
    add('RfrTen',rfr[0],2); add('RfrHundred',rfr[3])
    add('LossAmplification',loss[3]/loss[0],1)
    add('RfrAmplification',rfr[3]/rfr[0],0)
    add('RfrBeta',np.polyfit(np.log(loss),np.log(rfr),1)[0],2)
    add('ZeroRfrPercent',100*(baseline.rfr==0).mean(),1)
    add('RfrOnsetReturnPeriod',1/(baseline.rfr>0).mean(),1)
    for col,prefix in zip(COMPONENTS,['Figa','Citizens','Nfip']):
        add(prefix+'Hundred',baseline[col].quantile(.99)/1e9,1)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    (args.out_dir/'text_values.json').write_text(json.dumps(values,indent=2)+'\n')
    tex='% Generated from the validated September 2026 production campaign.\n'
    tex+='\n'.join('\\newcommand{\\'+k+'}{'+v['text']+'}' for k,v in values.items())+'\n'
    (args.out_dir/'values.tex').write_text(tex)
    print(f'Wrote {len(values)} manuscript values')


if __name__ == '__main__':
    main()

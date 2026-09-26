#!/usr/bin/env python3
"""Regenerate publication tables solely from explicitly validated validated simulation outputs.

The index is a JSON object mapping campaign job names to absolute iterations.csv
paths. No archive discovery, model execution, or fallbacks to old runs occur.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from publication_financial_cleanup import clean_figa_roundoff

RPS = np.array([10, 25, 50, 100, 250, 500, 1000])
GCM_MODELS = ['canesm', 'cnrm6', 'ecearth6', 'ipsl6', 'miroc6']
PERIODS = {
    'ssp245cal': '2050 SSP2-4.5', 'ssp585cal': '2050 SSP5-8.5',
    'ssp245_2cal': '2100 SSP2-4.5', 'ssp585_2cal': '2100 SSP5-8.5',
}
POLICIES = {
    'era5_baseline': 'Baseline',
    'era5_policy_market_exit_moderate': 'Market exit',
    'era5_policy_penetration_major': 'Insurance penetration',
    'era5_policy_building_codes_major': 'Building codes',
}
HISTORY = {
    'historical_great_miami': 'Great Miami', 'historical_andrew': 'Andrew',
    'historical_lake_okeechobee': 'Lake Okeechobee', 'historical_irma': 'Irma',
    'historical_gm_then_andrew': 'Great Miami then Andrew',
    'historical_double_gm': 'Double Great Miami', 'historical_double_irma': 'Double Irma',
}
METRICS = {
    'total_damage_usd': 'Total loss',
    'wind_insured_private_usd': 'Insured wind, private',
    'wind_insured_citizens_usd': 'Citizens wind',
    'flood_insured_capped_usd': 'Insured flood, NFIP',
    'wind_un_underinsured_usd': 'Un/underinsured wind',
    'flood_un_derinsured_usd': 'Un/underinsured flood',
    'residual_financing_requirement_usd': 'Residual financing requirement',
    'fhcf_shortfall_usd': 'FHCF shortfall',
    'figa_residual_deficit_usd': 'FIGA residual deficit',
    'citizens_residual_deficit_usd': 'Citizens residual deficit',
    'nfip_borrowed_usd': 'NFIP financing requirement',
}
EXTRA_METRICS = {
    'defaults_post': 'Private defaults',
    'largest_entity_deficit_usd': 'Largest single-entity deficit',
    'fhcf_reimbursement_usd': 'FHCF reimbursements',
    'fhcf_utilization': 'FHCF utilization factor',
    'citizens_assessment_stress': 'Citizens assessment stress factor',
    'figa_unfunded_share': 'FIGA unfunded share',
    'nfip_florida_stress': 'NFIP Florida stress factor',
}
PROBABILITIES = {
    'defaults_gt10': 'Defaults > 10',
    'single_deficit_gt1b': 'Single deficit > $1B',
    'fhcf_capacity_exhausted': 'FHCF statewide capacity exhausted',
    'figa_residual_positive': 'FIGA residual deficit > 0',
    'citizens_residual_positive': 'Citizens residual deficit > 0',
    'nfip_claims_gt2premium': 'NFIP claims > 200% annual premium',
    'rfr_gt1pct_gdp': 'Residual financing requirement > 1% Florida GDP',
    'rfr_gt10pct_gdp': 'Residual financing requirement > 10% Florida GDP',
}


def load_run(path):
    path = Path(path)
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f'Index must contain an existing absolute file path: {path}')
    d = pd.read_csv(path, low_memory=False)
    if 'scenario' in d and d.scenario.astype(str).str.lower().eq('error').any():
        raise ValueError(f'Error rows in {path}')
    if 'year_id' in d and d.year_id.duplicated().any():
        raise ValueError(f'Duplicate season ids in {path}')
    # Sparse zero-event rows have no financial calculation. Their absent amounts
    # mean zero; missing fields in any other row remain an error.
    zero_events = d['scenario'].eq('zero_events') & d['total_damage_usd'].eq(0)
    zero_fields = set(METRICS) | {'wind_underinsured_usd','wind_uninsured_usd',
        'underinsured_wind_usd','uninsured_wind_usd','defaults_post',
        'largest_entity_deficit_usd','fhcf_recovery_private_usd',
        'fhcf_recovery_citizens_usd','citizens_tier1_capacity_usd',
        'citizens_tier2_capacity_usd','figa_collected_usd',
        'nfip_claims_paid_usd','nfip_fl_premium_base_usd'}
    for col in zero_fields.intersection(d.columns):
        d.loc[zero_events & d[col].isna(), col] = 0.0
    clean_figa_roundoff(d)
    # The event-level writer records both these canonical attribution fields.
    w = 'wind_underinsured_usd' if 'wind_underinsured_usd' in d else 'underinsured_wind_usd'
    u = 'wind_uninsured_usd' if 'wind_uninsured_usd' in d else 'uninsured_wind_usd'
    d['wind_un_underinsured_usd'] = d[w] + d[u]
    d['residual_financing_requirement_usd'] = d[
        ['figa_residual_deficit_usd', 'citizens_residual_deficit_usd', 'nfip_borrowed_usd']
    ].sum(axis=1, skipna=False)
    d['public_burden_corrected_usd'] = d['residual_financing_requirement_usd']
    d['fhcf_reimbursement_usd'] = d['fhcf_recovery_private_usd'] + d['fhcf_recovery_citizens_usd']
    d['fhcf_utilization'] = d['fhcf_reimbursement_usd'] / 17e9
    den = d['citizens_tier1_capacity_usd'] + d['citizens_tier2_capacity_usd']
    d['citizens_assessment_stress'] = d['citizens_residual_deficit_usd'] / den.replace(0, np.nan)
    den = d['figa_residual_deficit_usd'] + d['figa_collected_usd']
    d['figa_unfunded_share'] = d['figa_residual_deficit_usd'] / den.replace(0, np.nan)
    d['nfip_florida_stress'] = d['nfip_claims_paid_usd'] / d['nfip_fl_premium_base_usd'].replace(0, np.nan)
    d.loc[zero_events, ['citizens_assessment_stress', 'nfip_florida_stress']] = 0.0
    for col in [*METRICS, 'defaults_post', 'largest_entity_deficit_usd', 'fhcf_reimbursement_usd']:
        if not np.isfinite(d[col].to_numpy(float)).all():
            raise ValueError(f'Nonfinite publication metric {col} in {path}')
    return d


def indicator_frame(d):
    f = pd.DataFrame(index=d.index)
    f['defaults_gt10'] = d.defaults_post > 10
    f['single_deficit_gt1b'] = d.largest_entity_deficit_usd > 1e9
    if 'fhcf_cap_binding' in d:
        # Explicit robust conversion also handles object-typed True/False values.
        f['fhcf_capacity_exhausted'] = d.fhcf_cap_binding.astype(str).str.lower().isin(['true', '1', '1.0'])
    else:
        f['fhcf_capacity_exhausted'] = d.fhcf_shortfall_usd > 0
    f['figa_residual_positive'] = d.figa_residual_deficit_usd > 0
    f['citizens_residual_positive'] = d.citizens_residual_deficit_usd > 0
    f['nfip_claims_gt2premium'] = d.nfip_claims_paid_usd > 2 * d.nfip_fl_premium_base_usd
    f['rfr_gt1pct_gdp'] = d.residual_financing_requirement_usd > .01 * 1.7e12
    f['rfr_gt10pct_gdp'] = d.residual_financing_requirement_usd > .10 * 1.7e12
    return f


def tex_escape(s):
    return str(s).replace('–', '--').replace('\\', '\\textbackslash{}').replace('&', r'\&').replace('%', r'\%').replace('$', r'\$').replace('_', r'\_')


def write_tabular(path, headers, rows, group_breaks=()):
    lines = [r'\begin{tabular}{l' + 'r' * (len(headers)-1) + '}', r'\toprule',
             ' & '.join(tex_escape(x) for x in headers) + r' \\', r'\midrule']
    for i, row in enumerate(rows):
        if i in group_breaks:
            lines.append(r'\midrule')
        lines.append(' & '.join(tex_escape(x) for x in row) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}', ''])
    path.write_text('\n'.join(lines))


def interval_string(v, lo, hi, scale=1, digits=1):
    if not np.isfinite(v):
        return 'n/a'
    return f'{v/scale:.{digits}f} ({lo/scale:.{digits}f}–{hi/scale:.{digits}f})'


def bootstrap_return_table(d, n_boot, seed):
    MR = METRICS
    columns = list(MR)
    arr = d[columns].to_numpy(float)
    qs = 1 - 1 / RPS
    point = np.quantile(arr, qs, axis=0)
    vals = np.empty((n_boot, len(qs), len(columns)))
    rng = np.random.default_rng(seed)
    for b in range(n_boot):
        vals[b] = np.quantile(arr[rng.integers(0, len(arr), size=len(arr))], qs, axis=0)
    lo, hi = np.percentile(vals, [10, 90], axis=0)
    rows = []
    for j, col in enumerate(columns):
        for i, rp in enumerate(RPS):
            rows.append(dict(metric=col, label=MR[col], return_period=int(rp),
                             point=point[i,j], p10=lo[i,j], p90=hi[i,j]))
    return pd.DataFrame(rows)


def bootstrap_probabilities(d, n_boot, seed):
    a = indicator_frame(d).to_numpy(float)
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, a.shape[1]))
    for b in range(n_boot):
        boot[b] = a[rng.integers(0, len(a), size=len(a))].mean(axis=0)
    lo, hi = np.percentile(boot, [10, 90], axis=0)
    return {c: (a[:,i].mean(), lo[i], hi[i]) for i,c in enumerate(PROBABILITIES)}


def summarize_rows(d, label, lo=10, hi=90):
    rows=[]
    for col, name in {**METRICS, **EXTRA_METRICS}.items():
        a = d[col].dropna().to_numpy(float)
        vals = [float(np.mean(a)), *np.percentile(a, [lo, hi])] if len(a) else [np.nan]*3
        rows.append(dict(scenario=label, metric=col, label=name, point=vals[0], p10=vals[1], p90=vals[2],
                         interval=f'Across realizations, percentiles {lo} and {hi}', n_rows=len(d), n_ratio_defined=len(a)))
    return rows


def climate_summaries(runs, scenario_summaries, prob_summaries):
    """Absolute within-GCM future-minus-historical deltas added to ERA5."""
    mean_rows=[]; prob_rows=[]; deltas=[]
    for period, scenario in PERIODS.items():
        for metric, label in {**METRICS, **EXTRA_METRICS}.items():
            changes=[]
            for model in GCM_MODELS:
                h=runs[f'gcm_baseline_{model}_20thcal'][metric].mean()
                f=runs[f'gcm_baseline_{model}_{period}'][metric].mean()
                changes.append(float(f-h))
                deltas.append(dict(statistic='annual_mean', metric=metric, model=model, period=period,
                                   historical=float(h), future=float(f), delta=float(f-h)))
            base = runs['era5_baseline'][metric].mean()
            center, lo, hi = base + np.quantile(changes, [.5, .1, .9])
            mean_rows.append(dict(scenario=scenario, period=period, metric=metric, label=label,
                                  point=center,p10=lo,p90=hi,interval='Across five GCM absolute deltas',n_models=5))
        for metric, label in PROBABILITIES.items():
            changes=[]
            for model in GCM_MODELS:
                h=indicator_frame(runs[f'gcm_baseline_{model}_20thcal'])[metric].mean()
                f=indicator_frame(runs[f'gcm_baseline_{model}_{period}'])[metric].mean()
                changes.append(float(f-h))
                deltas.append(dict(statistic='annual_exceedance_probability',metric=metric, model=model,period=period,
                                   historical=float(h),future=float(f),delta=float(f-h)))
            base=prob_summaries['Baseline'][metric][0]
            center, lo, hi = base + np.quantile(changes, [.5,.1,.9])
            # Keep pre-bound values in numeric files for a transparent audit.
            prob_rows.append(dict(scenario=scenario, period=period, metric=metric,label=label,
                                  point=float(np.clip(center,0,1)),p10=float(np.clip(lo,0,1)),p90=float(np.clip(hi,0,1)),
                                  raw_point=center,raw_p10=lo,raw_p90=hi,interval='Across five GCM absolute probability deltas',n_models=5))
    return mean_rows,prob_rows,deltas


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root',type=Path,required=True);p.add_argument('--index',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--n-boot',type=int,default=1000)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args(); out=args.out_dir;out.mkdir(parents=True,exist_ok=True)
    index=json.loads(args.index.read_text())
    required=list(POLICIES)+list(HISTORY)+[f'gcm_baseline_{m}_{p}' for m in GCM_MODELS for p in ['20thcal',*PERIODS]]+[f'insured_fraction_{f:.1f}' for f in [.1,.2,.3,.4,.5]]
    missing=set(required)-set(index)
    if missing:raise ValueError(f'Missing runs from validated index: {sorted(missing)}')
    runs={name:load_run(index[name]) for name in required}
    print(f'Loaded {len(runs)} indexed campaign runs.',flush=True)
    # Omit zero-only FHCF display rows while retaining full numerical CSV diagnostics.
    for name, frame in runs.items():
        if (frame.fhcf_shortfall_usd != 0).any() or indicator_frame(frame).fhcf_capacity_exhausted.any():
            raise ValueError(f'Nonzero FHCF shortfall in {name}; restore its separate display.')
    baseline=runs['era5_baseline']
    rp=bootstrap_return_table(baseline,args.n_boot,args.seed);rp.to_csv(out/'table1_return_levels.csv',index=False)
    table_rows=[]
    for col,label in METRICS.items():
        if col == 'fhcf_shortfall_usd':
            continue
        rr=rp[rp.metric.eq(col)].set_index('return_period')
        table_rows.append([label]+[f'{rr.loc[int(r),"point"]/1e9:.1f}' for r in RPS])
    write_tabular(out/'table1_return_levels.tex',['Metric (USD B)',*RPS],table_rows,group_breaks=(6,))
    print('Return levels and 1,000-resample bootstrap complete.',flush=True)
    rp_loss=rp[rp.metric.eq('total_damage_usd')].set_index('return_period').point
    rp_rfr=rp[rp.metric.eq('residual_financing_requirement_usd')].set_index('return_period').point
    amp={
        'loss_RP100_over_RP10':float(rp_loss[100]/rp_loss[10]),
        'rfr_RP100_over_RP10':float(rp_rfr[100]/rp_rfr[10]),
        'marginal_quantile_slope':float(np.polyfit(np.log(rp_loss),np.log(rp_rfr),1)[0]),
        'rfr_to_loss_marginal_quantile_ratio_RP10':float(rp_rfr[10]/rp_loss[10]),
        'rfr_to_loss_marginal_quantile_ratio_RP100':float(rp_rfr[100]/rp_loss[100]),
        'interpretation':'Matched return periods of marginal distributions, not same-season component shares or causal effects.'}
    (out/'amplification_summary.json').write_text(json.dumps(amp,indent=2)+'\n')
    hist=pd.DataFrame([row for name,label in HISTORY.items() for row in summarize_rows(runs[name],label,5,95)])
    hist.to_csv(out/'historical_summary.csv',index=False)
    hr=[]
    for col,label in {**METRICS,**EXTRA_METRICS}.items():
        if col in {'fhcf_reimbursement_usd', 'fhcf_shortfall_usd'}:continue
        row=[label];scale=1e9 if col.endswith('_usd') else 1;digits=1 if col.endswith('_usd') or col=='defaults_post' else 2
        for scenario in HISTORY.values():
            x=hist[hist.metric.eq(col)&hist.scenario.eq(scenario)].iloc[0]
            row.append(interval_string(x.point,x.p10,x.p90,scale,digits))
        hr.append(row)
    write_tabular(out/'tableS3_historical.tex',['Metric (USD B, counts or ratios)',*HISTORY.values()],hr,group_breaks=(6,10,12))
    means=[row for name,label in POLICIES.items() for row in summarize_rows(runs[name],label)]
    probs={label:bootstrap_probabilities(runs[name],args.n_boot,args.seed) for name,label in POLICIES.items()}
    probrows=[]
    for scenario,values in probs.items():
        for metric,(point,lo,hi) in values.items():
            probrows.append(dict(scenario=scenario,metric=metric,label=PROBABILITIES[metric],point=point,p10=lo,p90=hi,
                                 interval='1,000 whole-season bootstrap resamples, percentiles 10 and 90',n_models=0))
    cm,cp,deltas=climate_summaries(runs,means,probs);means.extend(cm);probrows.extend(cp)
    sm=pd.DataFrame(means);sp=pd.DataFrame(probrows)
    sm.to_csv(out/'climate_policy_means.csv',index=False);sp.to_csv(out/'climate_policy_probabilities.csv',index=False)
    pd.DataFrame(deltas).to_csv(out/'climate_deltas_by_gcm.csv',index=False)
    labels=['Baseline',*PERIODS.values(),*list(POLICIES.values())[1:]]
    for stem,df,metrics,scale in [('tableS4_climate_policy_means',sm,METRICS,1e9),('tableS5_probabilities',sp,PROBABILITIES,.01)]:
        rows=[]
        for metric,label in metrics.items():
            if metric in {'fhcf_shortfall_usd', 'fhcf_capacity_exhausted'}:
                continue
            row=[label]
            for scenario in labels:
                x=df[df.metric.eq(metric)&df.scenario.eq(scenario)].iloc[0]
                row.append(interval_string(x.point,x.p10,x.p90,scale,1 if scale==1e9 else 2))
            rows.append(row)
        write_tabular(out/(stem+'.tex'),['Metric (USD B)' if scale==1e9 else 'Metric (%)',*labels],rows,group_breaks=(6,) if scale==1e9 else (2,5))
    # Paired policy bootstrap preserves common season IDs and their dependence.
    prow=[]
    b=indicator_frame(baseline.set_index('year_id').sort_index())
    for name,label in list(POLICIES.items())[1:]:
        d=runs[name].set_index('year_id').sort_index()
        if not b.index.equals(d.index):raise ValueError(f'Unmatched policy season ids: {name}')
        a=indicator_frame(d).astype(float).to_numpy()-b.astype(float).to_numpy()
        rng=np.random.default_rng(args.seed);boot=np.empty((args.n_boot,a.shape[1]))
        for i in range(args.n_boot):boot[i]=a[rng.integers(0,len(a),size=len(a))].mean(axis=0)
        lo,hi=np.quantile(boot,[.1,.9],axis=0)
        for i,metric in enumerate(PROBABILITIES):
            prow.append(dict(scenario=label,metric=metric,label=PROBABILITIES[metric],point_pp=100*a[:,i].mean(),p10_pp=100*lo[i],p90_pp=100*hi[i]))
    pd.DataFrame(prow).to_csv(out/'policy_probability_changes.csv',index=False)
    sys.path.insert(0,str(args.repo_root/'scripts/analysis/publication'))
    import insured_fraction as sf
    fr={f:runs[f'insured_fraction_{f:.1f}'] for f in [.1,.2,.3,.4,.5]}
    fm=sf.means_table(fr);fe=sf.elasticities(fr);freturn=sf.return_period_by_fraction(fr)
    for frame in [fm, fe]:
        frame['Metric'] = frame['Metric'].replace({'NFIP Treasury borrowing': 'NFIP financing requirement'})
    fm.to_csv(out/'insured_fraction_means.csv',index=False);fe.to_csv(out/'insured_fraction_elasticities.csv',index=False)
    freturn.to_csv(out/'insured_fraction_return_levels.csv',index=False)
    rows=[]
    for i,rec in fm.iterrows():
        if rec.Metric == 'FHCF shortfall (diagnostic)':
            continue
        unit=1 if rec.Metric=='Insurer defaults (count)' else 1e9
        row=[rec.Metric]+[f'{rec[f"f={f}"]/unit:.2f}' for f in fr]
        er=fe.iloc[i]
        for f in [.1,.2,.3,.5]:
            v=er[f'delta_vs_f0.4_at_f={f}'];row.append(f'{v*100:+.0f}%' if np.isfinite(v) else 'n/a')
        v=er['elasticity_centered_loglog_0.3_0.5']
        row.append(f'{v:+.2f}' if np.isfinite(v) else 'n/a');rows.append(row)
    write_tabular(out/'tableS6_insured_fraction.tex',['Metric (USD B or count)',*[f'f={f}' for f in fr],*[f'Change, f={f}' for f in [.1,.2,.3,.5]],'Elasticity'],rows,group_breaks=(3,7))
    compare=[]
    for c,label in sf.METRICS.items():
        a=float(baseline[c].mean());bmean=float(fr[.4][c].mean())
        compare.append(dict(metric=c,label=label,beta46_mean=a,fixed04_mean=bmean,relative_difference=(bmean/a-1) if a else None))
    pd.DataFrame(compare).to_csv(out/'insured_fraction_fixed_vs_beta.csv',index=False)
    import decomposition as sd
    # Keep the already-reviewed fixed dollar severity edges, not moving bins.
    bins=sd.assign_bins(baseline,50);decomp=sd.bin_summary(baseline,bins,args.n_boot,args.seed)
    # The reviewed binning groups losses below USD 1 with exact zeros.
    # Correct the publication label without changing any bin membership.
    decomp['bin'] = decomp['bin'].replace({'Zero loss': 'Zero or negligible loss (up to USD 1)'})
    decomp.to_csv(out/'severity_bin_decomposition.csv',index=False)
    if decomp.reconciliation_check_diff.abs().max()>1e-12:raise ValueError('Decomposition failed to reconcile')
    names={'figa_residual_deficit_usd':'FIGA','citizens_residual_deficit_usd':'Citizens','nfip_borrowed_usd':'NFIP'}
    dr=[]
    for _,row in decomp.iterrows():
        dr.append([row['bin'],str(int(row.n_seasons)),f'{row.mean_total_loss_usd/1e9:.2f}',*[f'{row[f"mean_{c}"]/1e9:.2f}' for c in names],f'{100*row.total_nonoverlapping_share:.2f}'])
    write_tabular(out/'tableS_decomposition.tex',['Loss bin','Seasons','Mean loss (USD B)',*names.values(),'Financing share (%)'],dr)
    (out/'table_methods.json').write_text(json.dumps({
        'input_index':args.index.name,'index_sha256':hashlib.sha256(args.index.read_bytes()).hexdigest(),
        'n_boot':args.n_boot,'bootstrap_seed':args.seed,'return_periods':RPS.tolist(),
        'rfr_components':['figa_residual_deficit_usd','citizens_residual_deficit_usd','nfip_borrowed_usd'],
        'RP_method':'Linear empirical quantiles of complete season rows, including zero-loss seasons; RFR summed before quantiles.',
        'RP_intervals':'1,000 whole-season row resamples, 10th and 90th percentiles.',
        'historical_intervals':'5th and 95th percentiles across realization values, not a confidence interval on the mean.',
        'means_intervals':'ERA5/policy 10th–90th percentiles of seasonal values. Climate median and 10th–90th of five within-GCM future-minus-historical mean deltas, added to the ERA5 mean.',
        'probability_method':'Threshold indicators evaluated separately on every run. Climate probability deltas computed directly, added to ERA5; bounded to [0,1], with unbounded values retained.',
                'policy_intervals':'Paired season bootstrap of indicator differences, 10th–90th percentile.',
        'FIGA_unfunded_share':'residual / (residual + collected). Undefined zero-deficit rows omitted from ratio means, consistently with prior historical report.',
        'decomposition':'Fixed reviewed dollar edges with a zero or negligible loss bin up to USD 1, and minimum 50 seasons per merged nonzero bin; three non-overlapping component ratios of means; FHCF diagnostics retained in CSV outputs.',
        'FHCF_display':'Structurally zero shortfall rows omitted from LaTeX tables; full numerical CSV diagnostics retained. Nonzero utilization retained.',
        'changes_no_model_rerun':True},indent=2)+'\n')
    print(f'Publication table regeneration complete: {out}',flush=True)

if __name__=='__main__':main()

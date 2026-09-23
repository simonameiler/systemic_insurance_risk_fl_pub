#!/usr/bin/env python3
"""Regenerate the revised manuscript figures from explicitly validated runs.

This is postprocessing only. Every input is an iterations.csv resolved through
the supplied index. No simulation, source inputs, or archived results are changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch, Rectangle, PathPatch
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
from publication_financial_cleanup import clean_figa_roundoff

RFR_COLUMNS = ["figa_residual_deficit_usd", "citizens_residual_deficit_usd", "nfip_borrowed_usd"]
GCMS = ["canesm", "cnrm6", "ecearth6", "ipsl6", "miroc6"]
HISTORICAL = {
    "great_miami": "Great Miami", "andrew": "Andrew",
    "lake_okeechobee": "Lake Okeechobee", "irma": "Irma",
    "gm_then_andrew": "Great Miami then Andrew", "double_gm": "Great Miami twice",
    "double_irma": "Irma twice",
}
LEVELS = [(0, 0, 0), (20, 20, 13), (30, 30, 20), (40, 40, 27),
          (50, 50, 33), (60, 60, 40), (70, 70, 47), (80, 80, 53),
          (90, 90, 60), (100, 100, 67), (110, 100, 70),
          (120, 100, 80), (130, 100, 90)]
METRIC_LABELS = ["Private defaults > 10", "Largest deficit > USD 1 billion",
                 "FHCF statewide cap reached", "FIGA assessment capacity exceeded",
                 "Citizens assessment capacity exceeded", "NFIP claims > twice annual premium",
                 "Residual requirement > 1% of Florida GDP",
                 "Residual requirement > 10% of Florida GDP"]
COLORS = {"figa": "#8E44AD", "citizens": "#F39C12", "nfip": "#3498DB", "fhcf": "#D7BDE2"}


def style():
    # Restore the submitted manuscript notebook's publication settings.
    plt.rcParams.update({
        'font.family':        'sans-serif',
        'font.sans-serif':    ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size':          7,
        'axes.labelsize':     8,
        'axes.titlesize':     9,
        'xtick.labelsize':    7,
        'ytick.labelsize':    7,
        'legend.fontsize':    6.5,
        'axes.linewidth':     0.5,
        'xtick.major.width':  0.5,
        'ytick.major.width':  0.5,
        'xtick.major.size':   3,
        'ytick.major.size':   3,
        'xtick.direction':    'out',
        'ytick.direction':    'out',
        'xtick.major.pad':    3,
        'ytick.major.pad':    3,
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'pdf.fonttype':       42,
        'ps.fonttype':        42,
    })
    plt.rcParams["savefig.dpi"] = 300


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


class Runs:
    def __init__(self, repo, index):
        self.repo = repo
        self.index_path = index
        raw = json.loads(index.read_text())
        self.raw = raw
        entries = raw.get("runs", raw.get("jobs", raw))
        if isinstance(entries, list):
            self.entries = {r.get("name", r.get("job_name")): r for r in entries}
        else:
            self.entries = entries
        self.cache = {}

    def load(self, name):
        if name in self.cache:
            return self.cache[name]
        entry = self.entries[name]
        if isinstance(entry, str):
            path = Path(entry)
        else:
            path = Path(next(entry[k] for k in ["iterations_path", "iterations_csv", "iterations", "path", "run_dir", "output_dir"] if k in entry))
        if not path.is_absolute():
            path = self.repo / path
        if path.is_dir():
            path = path / "iterations.csv"
        wanted = set(RFR_COLUMNS + ["year_id", "iteration", "scenario", "events", "total_damage_usd",
            "wind_total_usd", "water_total_usd", "wind_insured_private_usd",
            "wind_insured_citizens_usd", "flood_insured_capped_usd", "defaults_post",
            "largest_entity_deficit_usd", "nfip_claims_paid_usd", "nfip_fl_premium_base_usd",
            "fhcf_shortfall_usd", "fhcf_cap_binding", "fhcf_recovery_private_usd",
            "fhcf_recovery_citizens_usd"])
        df = pd.read_csv(path, usecols=lambda x: x in wanted)
        if "scenario" in df and (df.scenario == "error").any():
            raise ValueError(f"Error rows in validated run {name}")
        expected = 1000 if name.startswith("historical_") else 10000
        if len(df) != expected:
            raise ValueError(f"{name} has {len(df)} rows, expected {expected}")
        no_events = df["scenario"].astype(str).eq("zero_events") if "scenario" in df else np.zeros(len(df),dtype=bool)
        for c in wanted.intersection(df.columns)-{"year_id", "iteration", "scenario", "events"}:
            missing = df[c].isna()
            if missing.any():
                allowed = no_events & (df["total_damage_usd"] == 0)
                if (missing & ~allowed).any():
                    raise ValueError(f"Undocumented missing {c} in {name}")
                df.loc[missing,c] = False if c=="fhcf_cap_binding" else 0.0
        for c in RFR_COLUMNS + ["total_damage_usd"]:
            if not np.isfinite(df[c].to_numpy(dtype=float)).all():
                raise ValueError(f"Nonfinite {c} in {name}")
        clean_figa_roundoff(df)
        df["rfr_usd"] = df[RFR_COLUMNS].sum(axis=1)
        df["fhcf_reimbursement_usd"] = df["fhcf_recovery_private_usd"] + df["fhcf_recovery_citizens_usd"]
        self.cache[name] = df
        return df


def indicators(df):
    if "fhcf_cap_binding" in df:
        raw = df["fhcf_cap_binding"]
        cap = raw if raw.dtype == bool else raw.astype(str).str.lower().isin(["true", "1", "1.0"])
    else:
        cap = df["fhcf_shortfall_usd"] > 0
    return np.column_stack([
        df["defaults_post"] > 10, df["largest_entity_deficit_usd"] > 1e9, cap,
        df["figa_residual_deficit_usd"] > 0,
        df["citizens_residual_deficit_usd"] > 0,
        (df["nfip_fl_premium_base_usd"] > 0) & (df["nfip_claims_paid_usd"] > 2 * df["nfip_fl_premium_base_usd"]),
        df["rfr_usd"] > 1.7e10, df["rfr_usd"] > 1.7e11,
    ]).astype(float)


def bootstrap_means(values, n_boot=1000, seed=42):
    """Ordinary empirical bootstrap of complete rows; works for paired deltas."""
    rng = np.random.default_rng(seed)
    result = np.empty((n_boot, values.shape[1]))
    for i in range(n_boot):
        result[i] = values[rng.integers(0, len(values), len(values))].mean(axis=0)
    return np.percentile(result, [10, 90], axis=0)


def historical_figure(runs, out, horizontal=True):
    """Historical stacks, with horizontal and original vertical display variants."""
    frames = [runs.load("historical_" + key) for key in HISTORICAL]
    scenario_order = [
        "Great Miami", "Andrew", "Lake Okeechobee", "Irma",
        "Great Miami then Andrew", "Double Great Miami", "Double Irma",
    ]
    loss_columns = [
        "wind_insured_private_usd", "wind_insured_citizens_usd",
        "flood_insured_capped_usd",
    ]
    physical = []
    for frame in frames:
        insured = [frame[column].mean() / 1e9 for column in loss_columns]
        uninsured_wind = (
            frame.wind_total_usd - frame.wind_insured_private_usd
            - frame.wind_insured_citizens_usd
        ).mean() / 1e9
        uninsured_flood = (
            frame.water_total_usd - frame.flood_insured_capped_usd
        ).mean() / 1e9
        components = insured + [uninsured_wind, uninsured_flood]
        if not np.isclose(sum(components), frame.total_damage_usd.mean() / 1e9):
            raise ValueError("Historical loss components do not sum to gross loss")
        physical.append(components)
    physical = np.array(physical)
    residual = np.array([
        [frame[column].mean() / 1e9 for column in RFR_COLUMNS]
        for frame in frames
    ])
    if not all((frame.fhcf_shortfall_usd == 0).all() for frame in frames):
        raise ValueError("Historical FHCF shortfall is nonzero; update separate diagnostic")
    totals = physical.sum(axis=1)
    insured_total = physical[:, :3].sum(axis=1)
    uninsured_total = physical[:, 3:].sum(axis=1)
    insured_pct = 100 * insured_total / totals
    uninsured_pct = 100 * uninsured_total / totals
    if horizontal:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 3.2), sharey=True)
        y = np.arange(len(frames))
        height = 0.40
        physical_labels = ["Insured wind (private)", "Citizens wind", "Insured flood (NFIP)",
                           "Un/underinsured wind", "Un/underinsured flood"]
        physical_colors = ["#E74C3C", "#F39C12", "#3498DB", "#95A5A6", "#BDC3C7"]
        left = np.zeros(len(frames))
        for values, label, color in zip(physical.T, physical_labels, physical_colors):
            ax1.barh(y, values, height, left=left, label=label, color=color, edgecolor="none")
            left += values
        ax1.set_xlim(0, totals.max()*1.3)
        ax1.set_xlabel("Loss (billion USD)")
        left = np.zeros(len(frames))
        for values, label, color in zip(residual.T,
                ["FIGA residual", "Citizens residual", "NFIP financing requirement"],
                ["#8E44AD", "#F39C12", "#3498DB"]):
            ax2.barh(y, values, height, left=left, label=label, color=color, edgecolor="none")
            left += values
        ax2.set_xlim(0, left.max()*1.3)
        ax2.set_xlabel("Residual financing requirement\n(billion USD)")
        ax1.set_yticks(y, [label.replace(" then ", " then\n") for label in scenario_order])
        ax1.set_ylim(len(frames)-0.3, -0.5)
        ax2.tick_params(axis="y", left=True, labelleft=False)
        for ax, letter in [(ax1, "a"), (ax2, "b")]:
            ax.legend(loc="upper right", frameon=True, framealpha=0.85, edgecolor="none",
                      handlelength=1.2, handletextpad=0.4, borderpad=0.4, labelspacing=0.3)
            ax.text(-0.06, 1.04, letter, transform=ax.transAxes, fontsize=9,
                    fontweight="bold", ha="right", va="bottom")
            ax.grid(False)
        fig.subplots_adjust(left=0.18, right=0.98, bottom=0.14, top=0.88, wspace=0.22)
        save(fig, out, "fig_loss_institutional_stress")
        return

    x_pos = np.arange(len(frames))
    width = 0.35

    fig = plt.figure(figsize=(7.1, 2.8))
    gs = fig.add_gridspec(1, 2, wspace=0.25)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    labels = [
        "Insured wind (private)", "Citizens wind", "Insured flood (NFIP)",
        "Un/underinsured wind", "Un/underinsured flood",
    ]
    colors = ["#E74C3C", "#F39C12", "#3498DB", "#95A5A6", "#BDC3C7"]
    bottom = np.zeros(len(frames))
    for column, label, color in zip(physical.T, labels, colors):
        ax1.bar(x_pos, column, width, bottom=bottom, label=label,
                color=color, edgecolor="none")
        bottom += column
    for i, total in enumerate(totals):
        ax1.plot([i - width / 2, i + width / 2],
                 [insured_total[i], insured_total[i]],
                 color="black", linewidth=0.8, alpha=0.6)
        ax1.text(i + width / 2 + 0.05, insured_total[i] / 2,
                 f"{insured_pct[i]:.0f}%", ha="left", va="center",
                 fontsize=5.5, fontweight="bold", color="#2C3E50")
        ax1.text(i + width / 2 + 0.05,
                 insured_total[i] + uninsured_total[i] / 2,
                 f"{uninsured_pct[i]:.0f}%", ha="left", va="center",
                 fontsize=5.5, fontweight="bold", color="#7F8C8D")
        ax1.text(i, total + 3, f"${total:.0f}B", ha="center", va="bottom",
                 fontsize=6, fontweight="bold")
    ax1.set_ylabel("Loss (billion USD)")

    bottom = np.zeros(len(frames))
    labels = ["FIGA residual", "Citizens residual", "NFIP financing requirement"]
    colors = ["#8E44AD", "#F39C12", "#3498DB"]
    for column, label, color in zip(residual.T, labels, colors):
        ax2.bar(x_pos, column, width, bottom=bottom, label=label,
                color=color, edgecolor="none")
        bottom += column
    for i, total in enumerate(bottom):
        if total > 0:
            ax2.text(i, total + 0.2, f"${total:.1f}B", ha="center", va="bottom",
                     fontsize=6, fontweight="bold")
    ax2.set_ylabel("Residual financing requirement\n(billion USD)")

    for ax, letter in [(ax1, "a"), (ax2, "b")]:
        ax.set_xticks(x_pos)
        ax.set_xticklabels(scenario_order, rotation=45, ha="right")
        ax.legend(loc="upper left", frameon=False, reverse=True,
                  handlelength=1.2, handletextpad=0.4, borderpad=0.3)
        ax.grid(False)
        ax.set_title(letter, fontweight="bold", loc="left", pad=6)
        ax.spines["left"].set_color("black")
        ax.spines["bottom"].set_color("black")
        ax.tick_params(axis="both", colors="black")
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")
        ax.title.set_color("black")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.28,
                        top=0.92, wspace=0.25)
    save(fig, out, "fig_loss_institutional_stress_vertical")


def scaling_figure(runs,out):
    df=runs.load("era5_baseline")
    loss=df.total_damage_usd.to_numpy()/1e9
    rfr=df.rfr_usd.to_numpy()/1e9
    rp=np.geomspace(2,1000,350); q=1-1/rp
    x,y=np.quantile(loss,q),np.quantile(rfr,q)
    boot=np.empty((1000,len(rp)));rng=np.random.default_rng(42)
    for i in range(1000):
        idx=rng.integers(0,len(df),len(df))
        boot[i]=np.quantile(rfr[idx],q)
    lo,hi=np.percentile(boot,[10,90],axis=0)
    marks=np.array([10,25,50,100,250,500,1000]);xm=np.quantile(loss,1-1/marks);ym=np.quantile(rfr,1-1/marks)
    beta,intercept=np.polyfit(np.log(xm),np.log(ym),1)
    fig,ax=plt.subplots(figsize=(3.45,2.85))
    reference,=ax.plot([0,x.max()*1.02],[0,x.max()*1.02*ym[0]/xm[0]],
        "--",color="#8E8E8A",lw=1,zorder=1)
    band=ax.fill_between(x,lo,hi,color="#A32D2D",alpha=.16,linewidth=0,zorder=2)
    curve,=ax.plot(x,y,color="#A32D2D",lw=1.9,zorder=3)
    ax.plot(xm,ym,"o",ms=3.8,color="#222222",mec="white",mew=.6,zorder=4)
    for rp_i,xi,yi in zip(marks,xm,ym):
        offset=(4,3) if rp_i==10 else ((4,-7) if rp_i==500 else (4,-1))
        ax.annotate(f"RP{rp_i}",(xi,yi),xytext=offset,textcoords="offset points",fontsize=6.5)
    ax.set(xlim=(0,x.max()*1.03),ylim=(0,max(y.max(),hi.max())*1.08),
        xlabel="Total seasonal loss (USD billion)",
        ylabel="Residual financing requirement (USD billion)")
    ax.grid(False)
    ax.tick_params(axis="both",which="both",direction="out",length=3,width=.5)
    slope=plt.Line2D([],[],linestyle="none",color="none")
    ax.legend([curve,band,reference,slope],
        ["Residual financing requirement","Bootstrap P10–P90",
         "Proportional reference (anchored at RP10)",f"Log–log fit slope beta = {beta:.2f}"],
        frameon=False,fontsize=6.5,loc="upper left",handlelength=1.4,labelspacing=.25)
    fig.tight_layout()
    save(fig,out,"fig_public_burden_scaling_linear")
    pd.DataFrame({"return_period":rp,"total_loss_billion":x,"rfr_billion":y,"rfr_p10_billion":lo,"rfr_p90_billion":hi}).to_csv(out/"scaling_curve.csv",index=False)
    summary={"beta":float(beta),"fit_return_periods":marks.tolist(),"fit_return_period_range":[10,1000],"bootstrap_resamples":1000,
        "loss_10":float(xm[0]),"loss_100":float(xm[3]),"rfr_10":float(ym[0]),"rfr_100":float(ym[3]),
        "loss_amplification_10_to_100":float(xm[3]/xm[0]),"rfr_amplification_10_to_100":float(ym[3]/ym[0]),
        "fraction_zero_rfr":float((rfr==0).mean()),"rfr_emergence_return_period":float(1/(rfr>0).mean()),
        "interpretation":"Marginal loss and residual financing return levels, not paired outcomes conditional on the same season."}
    (out/"scaling_summary.json").write_text(json.dumps(summary,indent=2))


def climate_figures(runs,out):
    base=runs.load("era5_baseline");base_ind=indicators(base);base_p=base_ind.mean(axis=0)*100
    base_ci=bootstrap_means(base_ind)*100
    climate={}
    rows=[]
    for pathway in ["ssp245","ssp585"]:
        for period in ["cal","_2cal"]:
            delta=[]
            for gcm in GCMS:
                present=indicators(runs.load(f"gcm_baseline_{gcm}_20thcal")).mean(axis=0)*100
                future=indicators(runs.load(f"gcm_baseline_{gcm}_{pathway}{period}")).mean(axis=0)*100
                delta.append(future-present)
            arr=np.array(delta)
            med=base_p+np.median(arr,axis=0);lo=base_p+np.percentile(arr,10,axis=0);hi=base_p+np.percentile(arr,90,axis=0)
            climate[pathway+period]=(med,lo,hi)
            for j,name in enumerate(METRIC_LABELS):
                rows.append({"scenario":pathway+period,"metric":name,"estimate":med[j],"p10":lo[j],"p90":hi[j]})
    policy={}
    for key,label in [("market_exit_moderate","Market exit"),("penetration_major","Expanded coverage"),("building_codes_major","Loss reduction")]:
        df=runs.load("era5_policy_"+key)
        if "year_id" not in base or "year_id" not in df:
            raise ValueError("Paired policy bootstrap requires year IDs")
        if not np.array_equal(base.year_id.to_numpy(),df.year_id.to_numpy()):
            raise ValueError("Policy and baseline year IDs do not match")
        delta=indicators(df)-base_ind;ci=bootstrap_means(delta)*100
        policy[label]=(delta.mean(axis=0)*100,ci[0],ci[1])
        for j,name in enumerate(METRIC_LABELS):
            rows.append({"scenario":key,"metric":name,"estimate":policy[label][0][j],"p10":ci[0,j],"p90":ci[1,j]})
    # Use the jointly generated publication tables for exactly matching intervals.
    table_dir=runs.index_path.parent/"tables"
    ptab=pd.read_csv(table_dir/"climate_policy_probabilities.csv")
    dtab=pd.read_csv(table_dir/"policy_probability_changes.csv")
    metric_order=["defaults_gt10","single_deficit_gt1b","fhcf_capacity_exhausted",
        "figa_residual_positive","citizens_residual_positive","nfip_claims_gt2premium",
        "rfr_gt1pct_gdp","rfr_gt10pct_gdp"]
    known=set(ptab.metric)
    if not set(metric_order).issubset(known):
        raise ValueError(f"Table metric names differ: {sorted(known)}")
    def from_table(scenario):
        d=ptab[ptab.scenario==scenario].set_index("metric").loc[metric_order]
        return tuple(d[c].to_numpy()*100 for c in ["point","p10","p90"])
    check=from_table("Baseline")
    if not np.allclose(check[0],base_p):raise ValueError("Independent figure and table baseline probabilities differ")
    base_p=check[0];base_ci=np.stack(check[1:])
    for path,lab in [("ssp245","SSP2-4.5"),("ssp585","SSP5-8.5")]:
        for period,year in [("cal","2050"),("_2cal","2100")]:
            saved=from_table(year+" "+lab)
            if not np.allclose(saved[0],np.clip(climate[path+period][0],0,100)):
                raise ValueError("Independent climate figure and table point estimates differ")
            climate[path+period]=saved
    for name,saved_name in [("Market exit","Market exit"),("Expanded coverage","Insurance penetration"),("Loss reduction","Building codes")]:
        d=dtab[dtab.scenario==saved_name].set_index("metric").loc[metric_order]
        saved=tuple(d[c].to_numpy() for c in ["point_pp","p10_pp","p90_pp"])
        if not np.allclose(saved[0],policy[name][0]):raise ValueError("Independent policy figure and table estimates differ")
        policy[name]=saved
    pd.DataFrame(rows).to_csv(out/"probability_figure_independent_check.csv",index=False)
    # Preserve all numerical outputs, but omit the structurally zero FHCF display row.
    omitted = metric_order.index("fhcf_capacity_exhausted")
    displayed = [i for i in range(len(metric_order)) if i != omitted]
    for values in [(base_p, base_ci[0], base_ci[1]), *climate.values(), *policy.values()]:
        if any(value[omitted] != 0 for value in values):
            raise ValueError("FHCF shortfall is nonzero; restore its separate display")
    base_p = base_p[displayed]
    base_ci = base_ci[:, displayed]
    climate = {key: tuple(value[displayed] for value in values) for key, values in climate.items()}
    policy = {key: tuple(value[displayed] for value in values) for key, values in policy.items()}
    # Reuse the submitted notebook's bar layout and policy colours.
    labels = ["Defaults > 10", "Single deficit > $1B",
              "FIGA > 100% capacity", "Citizens > 100% capacity",
              "NFIP > 200% annual premium", "Residual financing > 1% FL GDP",
              "Residual financing > 10% FL GDP"]
    for path,suffix in [("ssp245",""),("ssp585","_ssp585")]:
        fig = plt.figure(figsize=(7.1, 3))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 0.8], wspace=0.05)
        ax_a = fig.add_subplot(gs[0])
        ax_b = fig.add_subplot(gs[1], sharey=ax_a)
        y_pos = np.arange(len(labels))
        # Retain the original five-slot group envelope and 0.15-high bars.
        # The selected pathway now occupies three slots, matching the policy panel.
        w = 0.75 / 5
        offsets = np.array([-1, 0, 1]) * w
        if path == "ssp245":
            pathway_label = "SSP2-4.5"
            colors = ["#6B7280", "#EF9A9A", "#B71C1C"]
        else:
            pathway_label = "SSP5-8.5"
            colors = ["#6B7280", "#EF9A9A", "#B71C1C"]
        climate_bars = [
            ("ERA5 baseline", (base_p, base_ci[0], base_ci[1]), colors[0], 0.85),
            ("2050 " + pathway_label, climate[path + "cal"], colors[1], 0.90),
            ("2100 " + pathway_label, climate[path + "_2cal"], colors[2], 0.95),
        ]
        error_style = {"linewidth": 0.5, "ecolor": "#333333", "capsize": 1.5, "capthick": 0.5}
        for i, (label, (med, lo, hi), color, alpha) in enumerate(climate_bars):
            ax_a.barh(y_pos + offsets[i], med, w,
                xerr=np.stack([med-lo, hi-med]), color=color, edgecolor="none",
                linewidth=0, alpha=alpha, error_kw=error_style)
        xmax = max(np.max(vals[2]) for _, vals, _, _ in climate_bars)
        ax_a.set_xlim(0, xmax * 1.05)
        ax_a.set_yticks(y_pos, labels, fontsize=7)
        ax_a.invert_yaxis()
        ax_a.set_xlabel("Annual probability (%)")
        ax_a.text(-0.05, 1.00, "a", transform=ax_a.transAxes, fontsize=9,
                  fontweight="bold", va="bottom", ha="right")
        for boundary in [1.5, 4.5]:
            ax_a.axhline(boundary, color="black", linewidth=0.7,
                         xmin=-0.5, xmax=1.0, clip_on=False)
        legend_a = [Patch(facecolor=color, label=label) for label, _, color, _ in climate_bars]
        ax_a.legend(handles=legend_a, loc="lower right", frameon=False,
                    handlelength=1.2, handletextpad=0.4, borderpad=0.3, labelspacing=0.2)

        policy_bars = [("Building Codes", "Loss reduction", "#3B82F6"),
                       ("Penetration", "Expanded coverage", "#F59E0B"),
                       ("Market Exit", "Market exit", "#10B981")]
        bounds = []
        for i, (label, key, color) in enumerate(policy_bars):
            med, lo, hi = policy[key]
            ax_b.barh(y_pos + offsets[i], med, w,
                xerr=np.stack([med-lo, hi-med]), color=color, edgecolor="none",
                linewidth=0, label=label, error_kw=error_style)
            bounds.extend(lo.tolist()); bounds.extend(hi.tolist())
        ax_b.axvline(0, color="k", linewidth=0.5)
        ax_b.set_xlim(min(bounds) * 1.15, max(bounds) * 1.15)
        for boundary in [1.5, 4.5]:
            ax_b.axhline(boundary, color="black", linewidth=0.7, clip_on=False)
        ax_b.set_xlabel("Change relative to ERA5 baseline (pp)")
        ax_b.text(-0.05, 1.00, "b", transform=ax_b.transAxes, fontsize=9,
                  fontweight="bold", va="bottom", ha="right")
        plt.setp(ax_b.get_yticklabels(), visible=False)
        ax_b.legend(loc="lower right", frameon=False, handlelength=1.2,
                    handletextpad=0.4, borderpad=0.3, labelspacing=0.2)
        for ax in [ax_a, ax_b]:
            ax.grid(False)
            ax.tick_params(axis="both", which="both", direction="out", length=3,
                           width=0.5, colors="black")
        for center, title in [(0.5, "PRIVATE\nMARKET"), (3, "INSTITUTIONAL\nSTRESS"),
                              (5.5, "RESIDUAL\nFINANCING")]:
            ax_a.text(-0.61, center, title, transform=ax_a.get_yaxis_transform(),
                      fontsize=6, va="center", ha="center", color="#555555", rotation=90)
        fig.subplots_adjust(left=0.14, right=0.98, bottom=0.13, top=0.93, wspace=0.05)
        save(fig, out, "fig_combined_climate_policy_systemic_risk" + suffix)


def mean_metrics(df):
    return np.array([df.total_damage_usd.mean()/1e9,df.defaults_post.mean(),df.rfr_usd.mean()/1e9])


def first_crossing(x,y,target):
    if y[0]<=target:
        return float(x[0])
    for i in range(1,len(x)):
        if y[i]<=target<y[i-1]:
            return float(x[i-1]+(target-y[i-1])*(x[i]-x[i-1])/(y[i]-y[i-1]))
    return None


def building_figures(runs,out):
    baseline=mean_metrics(runs.load("era5_baseline"))
    hist={g:mean_metrics(runs.load(f"gcm_baseline_{g}_20thcal")) for g in GCMS}
    data=[];detail=[]
    for level_index,(level,wind,flood) in enumerate(LEVELS):
        arr=[]
        for g in GCMS:
            df=runs.load(f"buildingcode_{g}_L{level_index:02d}")
            calibrated=baseline+mean_metrics(df)-hist[g]
            arr.append(calibrated)
            for i,metric in enumerate(["total_loss_billion","defaults","rfr_billion"]):
                detail.append({"gcm":g,"level_index":level_index,"level_label":level,"wind_reduction_pct":wind,"flood_reduction_pct":flood,"mean_prescribed_reduction_pct":(wind+flood)/2,"metric":metric,"era5_plus_gcm_delta":calibrated[i]})
        arr=np.array(arr)
        row={"level_index":level_index,"level_label":level,"wind_reduction_pct":wind,"flood_reduction_pct":flood,"mean_prescribed_reduction_pct":(wind+flood)/2}
        for i,metric in enumerate(["total_loss_billion","defaults","rfr_billion"]):
            row[metric]=np.median(arr[:,i]);row[metric+"_p10"]=np.percentile(arr[:,i],10);row[metric+"_p90"]=np.percentile(arr[:,i],90)
        data.append(row)
    table=pd.DataFrame(data);table.to_csv(out/"building_code_curves.csv",index=False)
    pd.DataFrame(detail).to_csv(out/"building_code_gcm_curves.csv",index=False)
    offsets=[]
    x=table.mean_prescribed_reduction_pct.to_numpy()
    for i,metric in enumerate(["total_loss_billion","defaults","rfr_billion"]):
        for stat in ["median","p10","p90"]:
            col=metric+("" if stat=="median" else "_"+stat)
            crossing=first_crossing(x,table[col].to_numpy(),baseline[i])
            offsets.append({"metric":metric,"ensemble_statistic":stat,"baseline":baseline[i],
                "mean_prescribed_reduction_pct":crossing,
                "wind_reduction_pct":float(np.interp(crossing,x,table.wind_reduction_pct)) if crossing is not None else None,
                "flood_reduction_pct":float(np.interp(crossing,x,table.flood_reduction_pct)) if crossing is not None else None})
    pd.DataFrame(offsets).to_csv(out/"building_code_offsets.csv",index=False)
    (out/"building_code_offsets.json").write_text(json.dumps({"method":"First crossing of ERA5 mean using linear interpolation between tested wind/flood pairs. The x coordinate is their unweighted arithmetic mean, not realized total loss reduction. P10/P90 are ensemble curves, not percentiles of individual GCM crossing locations.","offsets":offsets,"negative_delta_adjusted_values":int(sum((table[c]<0).sum() for c in table if c.endswith(("_p10","_p90")) or c in ["total_loss_billion","defaults","rfr_billion"]))},indent=2))
    for full, name in [(True, "fig_climate_buildingcode_sensitivity"),
                       (False, "fig_climate_buildingcode_sensitivity_public_burden")]:
        panels = [(0, "total_loss_billion", "Annual total loss (billion USD)", "a"),
                  (1, "defaults", "Annual defaults", "b"),
                  (2, "rfr_billion", "Annual residual financing\nrequirement (billion USD)", "c")]
        if not full:
            panels = panels[-1:]
        fig, axs = plt.subplots(1, len(panels), figsize=(7.1, 2.0) if full else (3.2, 2.6),
                                sharex=True, squeeze=False)
        for j, (i, metric, ylabel, letter) in enumerate(panels):
            ax = axs[0, j]
            ax.plot(x, table[metric], "o-", color="k", alpha=0.85,
                    linewidth=0.8, markersize=2.5,
                    label="SSP2-4.5 2050 (median)" if j == 0 else None, zorder=3)
            ax.fill_between(x, table[metric + "_p10"], table[metric + "_p90"],
                            color="k", alpha=0.12, linewidth=0,
                            label="P10–P90 range" if j == 0 else None, zorder=1)
            ax.axhline(baseline[i], color="k", linestyle="--", linewidth=0.6,
                       alpha=0.5, label="ERA5 baseline" if j == 0 else None, zorder=2)
            crossing = first_crossing(x, table[metric].to_numpy(), baseline[i])
            if crossing is not None:
                ax.plot(crossing, baseline[i], marker="o", markersize=5, color="black",
                        zorder=5)
                ax.annotate(f"{crossing:.1f}%", (crossing, baseline[i]),
                            xytext=(0, 6), textcoords="offset points", fontsize=7,
                            ha="center", va="bottom")
                ax.set_ylim(bottom=0)
            ax.set_ylabel(ylabel)
            if full:
                ax.set_title(letter, fontweight="bold", loc="left", pad=4)
            ax.set_xlim(-5, 105)
            # Original notebook autoscaling is retained when all plotted values are positive.
            if float(table[metric + "_p10"].min()) < 0:
                ax.set_ylim(bottom=0)
            ax.grid(False)
        # The original axis wording is made explicit about the prescribed mean.
        # All thirteen pairs and the correct unweighted wind/flood coordinate are retained.
        fig.text(0.5, -0.02, "Mean prescribed wind/flood reduction (%)", ha="center")
        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.88, 0.5),
                   frameon=False)
        fig.tight_layout()
        fig.subplots_adjust(right=0.88)
        save(fig, out, name)


def overview(out, reference):
    """Restore the four-panel composition with the corrected financing sum.

    The source raster is clipped to panels a and b only. Its risk-flow/output
    panels are never shown. This preserves existing exposure/hazard artwork
    exactly without inventing missing track/source data.
    """
    reference = Path(reference)
    source_image = plt.imread(reference)
    h, w = source_image.shape[:2]
    if (w, h) != (3081, 1212):
        raise ValueError(f'Unexpected original overview size {(w, h)}')
    with plt.rc_context({'font.family':'Arial','font.size':9,
                         'pdf.fonttype':42,'ps.fonttype':42,
                         'axes.grid':False}):
        fig=plt.figure(figsize=(15.405,6.06))
        ax=fig.add_axes([0,0,1,1])
        ax.set(xlim=(0,3081),ylim=(1212,0))
        ax.axis('off')
        # Native image clipping retains the unchanged original input panels.
        # All raster content to the right is fully excluded from the output.
        im=ax.imshow(source_image,extent=(0,3081,1212,0),interpolation='none')
        clip=Rectangle((0,0),1320,1212,transform=ax.transData)
        im.set_clip_path(clip)
        def text(x,y,s,size=10,bold=False,ha='left',color='black',va='center'):
            return ax.text(x,y,s,fontsize=size,fontweight='bold' if bold else 'normal',
                    ha=ha,va=va,color=color,linespacing=1.2)
        def box(x,y,ww,hh,title,body='',face='#e4effb',edge='#5c92c6',fs=10.3,bodyfs=9.7):
            ax.add_patch(FancyBboxPatch((x,y),ww,hh,boxstyle='round,pad=0,rounding_size=16',
                           facecolor=face,edgecolor=edge,lw=1.0))
            if body:
                text(x+ww/2,y+hh*.36,title,fs,True,'center')
                text(x+ww/2,y+hh*.70,body,bodyfs,False,'center')
            else: text(x+ww/2,y+hh*.50,title,fs,True,'center',color='#484a46')
        def arrow(a,b,color='#4c4f4a',dash=False,style='-|>',curve=None,lw=1.35):
            kw={'connectionstyle':curve} if curve else {}
            ax.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=16,
                           color=color,lw=lw,linestyle=(0,(4,3)) if dash else '-',**kw))
        def layer(x,y,ww,hh,title,col):
            ax.add_patch(FancyBboxPatch((x,y),ww,hh,boxstyle='round,pad=0,rounding_size=3',
                         facecolor='none',edgecolor=col,lw=1))
            text(x+29,y+32,title,10.3,True,color=col)
        # The original stepped outline leaves a separate output-metrics panel.
        vertices=[(1340,20),(3061,20),(3061,378),(2615,378),(2615,1190),(1340,1190),(1340,20)]
        ax.add_patch(PathPatch(MplPath(vertices),facecolor='white',edgecolor='black',lw=2))
        text(1357,65,'c. Risk propagation model',11.2,True)
        layer(1370,110,1600,240,'Layer 1   risk absorption','#226ab0')
        box(1400,172,362,148,'Private insurers','Wind coverage')
        box(1792,172,362,148,'Citizens','Residual wind market')
        box(2184,172,362,148,'NFIP','Flood coverage')
        box(2576,172,362,148,'Uninsured and\nunderinsured assets',face='#f8f8f7',edge='#b0b1ac',fs=10)
        layer(1370,412,1206,242,'Layer 2   risk transfer and capital support','#087e64')
        green='#e3eee2'; greenedge='#54a28c'
        box(1400,472,362,151,'FHCF reinsurance','Company limits\nStatewide cap USD 17B',green,greenedge,10.2,9.4)
        box(1792,472,362,151,'Catastrophe bonds','Per-sponsor limits',green,greenedge)
        box(2184,472,362,151,'Insurer capital','Statutory surplus',green,greenedge)
        # Schematic arrows follow the submitted figure, with the NFIP branch
        # bypassing wind insurer support. FHCF shortfall is kept upstream.
        arrow((1581,320),(1581,408))
        arrow((1973,320),(1973,408))
        ax.plot([2365,2365,2562,2562,2365],[320,366,366,691,691], color='#4c4f4a', lw=1.35, ls=(0,(4,3)))
        arrow((2365,691),(2365,710),dash=True)
        arrow((1581,623),(1581,710))
        arrow((1973,623),(1973,710))
        capital_path=MplPath([(2275,623),(2275,671),(2275,671),(2225,671),(2080,671),(2045,671),(2045,710)], [MplPath.MOVETO,MplPath.CURVE3,MplPath.CURVE3,MplPath.LINETO,MplPath.LINETO,MplPath.CURVE3,MplPath.CURVE3])
        ax.add_patch(FancyArrowPatch(path=capital_path,arrowstyle='-|>',mutation_scale=16,color='#4c4f4a',lw=1.35))
        layer(1370,714,1206,242,'Layer 3   public backstops','#a63418')
        red='#f8ded8';rededge='#c67661'
        box(1400,775,362,149,'FIGA assessments','2% normal + 4%\nemergency cap',red,rededge,10,9.5)
        box(1792,775,362,149,'Citizens assessments','Tier 1 15% / Tier 2 10%',red,rededge,10,9.3)
        box(2184,775,362,149,'NFIP financing','U.S. Treasury requirement',red,rededge,10,9.3)
        ax.add_patch(FancyBboxPatch((1370,1014),1210,135,boxstyle='round,pad=0,rounding_size=5',
                      facecolor='#fbebeb',edgecolor='#ac3034',lw=2))
        text(1975,1048,'Residual financing requirement',11.2,True,'center')
        text(1975,1092,'FIGA residual + Citizens residual + NFIP financing requirement',9.3,ha='center')
        for xx in (1670,1973,2365):
            arrow((xx,924),(xx,1013),'#ac3034',dash=True)
        # An upstream shortfall affects downstream residuals. It is not an
        # additional fourth addend, so its former direct red arrow is removed.
        text(1975,1125,'FHCF shortfall is reported separately',8.1,ha='center',color='#555555')
        arrow((1375,1168),(1483,1168))
        arrow((1515,1168),(1630,1168),dash=True)
        text(1649,1171,'flow of losses',9,color='#4c4f4a')
        arrow((1970,1168),(2090,1168),'#ac3034',dash=True)
        text(2111,1171,'threshold exceeded',9,color='#ac3034')
        # Output panel retains the submitted compact icon-based composition.
        ax.add_patch(Rectangle((2645,412),416,778,facecolor='white',edgecolor='black',lw=2))
        text(2671,444,'d. Output metrics',11.1,True)
        def tinyaxis(x,y,ww=72,hh=84):
            ax.plot([x,x,x+ww],[y-hh,y,y],color='#858882',lw=.8)
        tinyaxis(2682,584,61,105)
        for x0,hs in [(2687,[38,29,34]),(2718,[27,21,27])]:
            b=584
            for hh,col in zip(hs,['#b8b6ae','#82b9e8','#1d63a9']):
                ax.add_patch(Rectangle((x0,b-hh),20,hh,color=col,lw=0));b-=hh
        text(2770,540,'Loss decomposition',9.2)
        tinyaxis(2687,702,59,82)
        b=702
        for hh,col in zip([22,23,22],['#ef9998','#dc4744','#8e292d']):
            ax.add_patch(Rectangle((2700,b-hh),37,hh,color=col,lw=0));b-=hh
        text(2770,658,'Residual financing\nrequirement',9.2)
        tinyaxis(2674,805,80,58)
        for x0,hh,col in [(2681,52,'#659b21'),(2700,34,'#659b21'),(2722,-38,'#d85727'),(2742,-25,'#d85727')]:
            ax.add_patch(Rectangle((x0,min(805,805-hh)),15,abs(hh),color=col,lw=0))
        text(2770,800,'Capital depletion\nand defaults',9.2)
        tinyaxis(2674,989,1,102)
        for y0,ww,col in [(900,66,'#388dda'),(923,42,'#659b21'),(947,88,'#ed474a'),(971,58,'#a32a2d')]:
            ax.add_patch(Rectangle((2675,y0),ww,14,color=col,lw=0))
        text(2770,936,'Stress exceedance\nprobabilities',9.2)
        tinyaxis(2674,1149,100,88)
        x=np.linspace(2680,2758,40); t=(x-2680)/78
        ax.plot(x,1140-75*t**1.2,color='#1d63a9',lw=2)
        ax.plot(x,1148-58*t**1.3,color='#e74c4c',lw=1.7,ls='--')
        text(2770,1112,'Return period\ncurves',9.2)
        save(fig,out,'fig1_systemic_risk_overview_florida')
    Path(out,'overview_source_provenance.json').write_text(json.dumps({
        'source':str(reference),'sha256':hashlib.sha256(reference.read_bytes()).hexdigest(),
        'unchanged_reference_panels':['a. Data input','b. Scenario development'],
        'source_display_clip_pixels':[0,0,1320,1212],
        'redrawn_panels':['c. Risk propagation model','d. Output metrics'],
        'accounting':'FIGA residual deficit + Citizens residual deficit + NFIP financing requirement; FHCF shortfall is an upstream diagnostic reported separately.'
    },indent=2))


def physical_figures(runs,out,source):
    """Restore the author's original notebook style, preserving revised data."""
    source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    files=[source/'all_events.csv']
    provenance={str(p):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in files}
    events=pd.read_csv(source/'all_events.csv')
    if len(events)!=8800:raise ValueError('Unexpected event catalog count')
    loss=np.sort(events.total_damage_usd.to_numpy())[::-1]
    rp=len(events)/(1.8*np.arange(1,len(events)+1))
    seasons=np.sort(runs.load('era5_baseline').total_damage_usd.to_numpy())[::-1]
    srp=len(seasons)/np.arange(1,len(seasons)+1)
    with plt.rc_context(matplotlib.rcParamsDefault):
        plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42})
        fig,ax=plt.subplots(figsize=(4,3))
        ax.plot(srp,seasons/1e9,linewidth=2.5,label='Year set-based',color='k',alpha=.85)
        ax.plot(rp,loss/1e9,linewidth=2.5,label='Event-based',color='#E74C3C',alpha=.85,linestyle='--')
        ax.set(xscale='log',yscale='log',ylim=(1e-3,2e3),xlim=(1,5000))
        ax.set_xlabel('Return Period (Years)',fontsize=11,fontweight='normal')
        ax.set_ylabel('Total Loss (Billion USD)',fontsize=11,fontweight='normal')
        ax.set_xticks([1,10,100,1000]);ax.set_yticks([1e-3,1e-2,1e-1,1,10,100,1000])
        for side in ('bottom','left'):
            ax.spines[side].set_visible(True);ax.spines[side].set_color('black');ax.spines[side].set_linewidth(1)
        for side in ('top','right'):ax.spines[side].set_visible(False)
        ax.tick_params(axis='both',which='major',bottom=True,left=True,top=False,right=False,
                       direction='out',length=4,width=1,color='k',labelsize=10)
        ax.grid(True,alpha=.3,which='both',linestyle='-',linewidth=.5)
        ax.legend(fontsize=8,frameon=True,loc='lower right')
        fig.tight_layout();save(fig,out,'fig_loss_return_period')
    pd.DataFrame({'rank':np.arange(1,len(events)+1),'event_loss_usd':loss,'return_period_years':rp}).to_csv(out/'event_loss_return_curve.csv',index=False)
    (out/"physical_source_provenance.json").write_text(json.dumps({
        "files":provenance,"event_catalog_mean_annual_frequency":1.8,
        "catalog_events":len(events),"definition":"Event and seasonal economic loss return curves; unchanged physical-loss inputs.",
        "style_sources":["notebooks/emanuel_tc_policy_analysis.ipynb cell 58"]},indent=2))


def decomposition_figure(runs,out):
    table=pd.read_csv(runs.index_path.parent/"tables"/"severity_bin_decomposition.csv")
    table=table[~table.bin.astype(str).str.startswith("Zero")].copy()
    x=np.arange(len(table));fig,ax=plt.subplots(figsize=(3.4,2.65));bottom=np.zeros(len(table))
    for c,label,color in zip(RFR_COLUMNS,["FIGA residual","Citizens deficit","NFIP financing"],
                             [COLORS[k] for k in ["figa","citizens","nfip"]]):
        vals=table["share_"+c].to_numpy()*100
        ax.bar(x,vals,bottom=bottom,width=.4,label=label,color=color);bottom+=vals
    labels=[]
    for _,row in table.iterrows():
        label=row["bin"].split(" (")[0].replace("yr","").replace(" -- ","–")
        labels.append(label+f"\n(n={int(row.n_seasons):,})")
    ax.set_xticks(x,labels,fontsize=6)
    ax.set_ylabel("Residual financing / economic loss (%)")
    ax.set_xlabel("Loss severity (approximate return period in years)")
    handles,labels=ax.get_legend_handles_labels()
    ax.legend(handles,labels,frameon=False,loc="upper left",fontsize=6.5,
              handlelength=1.2,labelspacing=.25)
    ax.grid(False)
    ax.set_ylim(0,max(bottom)*1.08)
    fig.tight_layout()
    save(fig,out,"fig_residual_financing_severity_decomposition")


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--repo-root",required=True,type=Path);ap.add_argument("--index",required=True,type=Path);ap.add_argument("--out-dir",required=True,type=Path)
    ap.add_argument("--physical-data",type=Path,required=True)
    args=ap.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True);style();runs=Runs(args.repo_root,args.index)
    for fn in [historical_figure,scaling_figure,climate_figures,building_figures]:
        print("Generating",fn.__name__,flush=True);fn(runs,args.out_dir)
    historical_figure(runs, args.out_dir, horizontal=False)
    overview(args.out_dir, Path(__file__).resolve().parent/"assets"/"original_overview_reference.png")
    physical_figures(runs,args.out_dir,args.physical_data)
    decomposition_figure(runs,args.out_dir)
    (args.out_dir/"figure_methods_notes.json").write_text(json.dumps({
        "historical":"Panel a stacks mean physical loss components. Panel b stacks mean FIGA, Citizens and NFIP residual financing requirements. The main variant uses horizontal bars; a vertical alternative retains the original layout. Colours, totals and insured shares are preserved. The zero FHCF shortfall is explained in the text and omitted from displays. Historical scenarios use 1,000 realizations; the 5th–95th percentile ranges are in SI Table S3.",
        "scaling":"Both coordinates are separately estimated marginal return levels. Residual requirement is summed within each season before quantiles. Shading is the pointwise 10th–90th percentile from 1,000 whole-season bootstrap resamples of the residual requirement, at fixed return periods; it does not include horizontal uncertainty. The log-log fit uses the seven table return periods of 10, 25, 50, 100, 250, 500 and 1,000 years.",
        "climate":"Absolute future probability equals present ERA5 probability plus the median future-minus-historical GCM change. Climate error bars show the 10th–90th percentile across five GCM deltas. Present-day error bars use 1,000 whole-season bootstrap resamples. Policy change intervals use paired whole-season bootstrap resamples after verifying identical year IDs. The all-zero FHCF shortfall row is omitted from both panels. Future periods use light and dark red; policy colours are retained.",
        "building":"Expected annual metric is ERA5 mean plus each future-with-reduction-minus-historical GCM mean. Lines/bands summarize these five delta-adjusted values. Raw delta-adjusted values are retained. Axes follow the original plot range, with a nonnegative display floor. Horizontal coordinate is the unweighted mean of prescribed wind/flood loss reductions. Offsets use linear interpolation at the first baseline crossing and report both component reductions. The wider main panel marks the median ERA5 crossing in black. Crossings of the P10 and P90 curves are not percentiles of per-model crossings.",
        "physical_figures":"Event return curve uses the unchanged 8,800-event catalog with equal event frequency 1.8/8800 per year, following the original notebook. Seasonal curve uses all 10,000 corrected ERA5 production seasons.",
        "severity_decomposition":"Stacks each institutional mean residual divided by mean total loss within the same gross-loss severity bin. The zero or negligible loss bin, containing seasons with total loss up to USD 1, is omitted. This is an accounting decomposition among common seasons, not causal attribution to institutional thresholds.",
        "overview":"The original four-panel layout is restored. Input exposure maps and historical/synthetic scenario artwork are preserved from the submitted raster. Institutional architecture and output icons are redrawn as vectors. The financing aggregate has three downstream components, with FHCF shortfall separate; NFIP bypasses insurer capital. Original artwork source and clipping are recorded in overview_source_provenance.json."},indent=2))
    print("Figures and numerical summaries written to",args.out_dir)


if __name__=="__main__":main()

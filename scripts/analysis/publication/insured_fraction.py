from __future__ import annotations
import numpy as np
import pandas as pd
from common import RETURN_PERIODS, empirical_return_level, to_billion

METRICS = {
    "wind_insured_private_usd": "Wind insured, private",
    "wind_insured_citizens_usd": "Wind insured, Citizens",
    "wind_un_underinsured_usd": "Wind un/underinsured",
    "fhcf_shortfall_usd": "FHCF shortfall (diagnostic)",
    "figa_residual_deficit_usd": "FIGA residual deficit",
    "citizens_residual_deficit_usd": "Citizens residual deficit",
    "nfip_borrowed_usd": "NFIP Treasury borrowing",
    "public_burden_corrected_usd": "Residual financing requirement",
    "defaults_post": "Insurer defaults (count)",
    "largest_entity_deficit_usd": "Largest entity deficit",
}

def means_table(runs: dict[float, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for col, label in METRICS.items():
        row = {"Metric": label}
        for f, df in sorted(runs.items()):
            row[f"f={f}"] = float(df[col].mean())
        rows.append(row)
    return pd.DataFrame(rows)

def elasticities(runs: dict[float, pd.DataFrame], f0: float = 0.4) -> pd.DataFrame:
    fs = sorted(runs.keys())
    if f0 not in fs:
        raise ValueError(f"reference fraction {f0} not in sweep {fs}")
    rows = []
    for col, label in METRICS.items():
        means = {f: float(runs[f][col].mean()) for f in fs}
        row = {"Metric": label}
        for f in fs:
            if f == f0:
                continue
            m0, mf = means[f0], means[f]
            row[f"delta_vs_f0.4_at_f={f}"] = (
                (mf - m0) / m0 if m0 != 0 else float("nan")
            )
        # Centered log-log finite difference using f=0.3 and f=0.5 around 0.4,
        # on UNROUNDED means (Section 7 requirement).
        if 0.3 in means and 0.5 in means and means[0.3] > 0 and means[0.5] > 0:
            eps_central = (
                (np.log(means[0.5]) - np.log(means[0.3])) / (np.log(0.5) - np.log(0.3))
            )
        else:
            eps_central = float("nan")
        row["elasticity_centered_loglog_0.3_0.5"] = eps_central
        rows.append(row)
    return pd.DataFrame(rows)

def return_period_by_fraction(runs: dict[float, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for f, df in sorted(runs.items()):
        rl_loss = empirical_return_level(df["total_damage_usd"].to_numpy(dtype=float))
        rl_burden = empirical_return_level(df["public_burden_corrected_usd"].to_numpy(dtype=float))
        row = {
            "f": f,
            "loss_RP10_B": to_billion(rl_loss["RP10"]),
            "loss_RP100_B": to_billion(rl_loss["RP100"]),
            "loss_amp_100_over_10": rl_loss["RP100"] / rl_loss["RP10"] if rl_loss["RP10"] > 0 else np.nan,
            "burden_RP10_B": to_billion(rl_burden["RP10"]),
            "burden_RP100_B": to_billion(rl_burden["RP100"]),
            "burden_amp_100_over_10": rl_burden["RP100"] / rl_burden["RP10"] if rl_burden["RP10"] > 0 else np.nan,
            "P(burden>1%GDP)": float((df["public_burden_corrected_usd"] > 0.01 * 1.7e12).mean()),
            "mean_public_burden_corrected_B": to_billion(df["public_burden_corrected_usd"].mean()),
        }
        rows.append(row)
    return pd.DataFrame(rows)

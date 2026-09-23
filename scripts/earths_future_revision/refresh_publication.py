"""Rebuild the Earth's Future figures, tables and numerical reference values.

This command reads completed simulation outputs. It does not run or change
the financial model, submit jobs, or overwrite the source iterations files.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

MAIN_FIGURES = [
    'fig1_systemic_risk_overview_florida',
    'fig_loss_institutional_stress',
    'fig_public_burden_scaling_linear',
    'fig_combined_climate_policy_systemic_risk',
    'fig_climate_buildingcode_sensitivity_public_burden',
]
SI_FIGURES = [
    'fig_loss_return_period',
    'fig_climate_buildingcode_sensitivity',
    'fig_combined_climate_policy_systemic_risk_ssp585',
    'fig_residual_financing_severity_decomposition',
]


def install_artifacts(artifacts, manuscript):
    for part, figures in [('main', MAIN_FIGURES), ('si', SI_FIGURES)]:
        dest = manuscript / part
        (dest / 'figures').mkdir(parents=True, exist_ok=True)
        (dest / 'tables').mkdir(exist_ok=True)
        for name in figures:
            for extension in ['pdf', 'png']:
                shutil.copy2(artifacts/'figures'/f'{name}.{extension}', dest/'figures'/f'{name}.{extension}')
        fragments = ['table1_return_levels.tex'] if part == 'main' else [
            'tableS3_historical.tex', 'tableS4_climate_policy_means.tex',
            'tableS5_probabilities.tex', 'tableS6_insured_fraction.tex', 'tableS_decomposition.tex',
        ]
        for name in fragments:
            shutil.copy2(artifacts/'tables'/name, dest/'tables'/name)
    # The author edits numerical values directly in the manuscript prose.
    # Keep generated reference values in the results directory for manual review.
    print('Manuscript prose uses literal numbers. Review text_values.json and '
          'figures/building_code_offsets.json when numerical results change.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--campaign-root', type=Path)
    ap.add_argument('--log-dir', type=Path)
    ap.add_argument('--physical-data', type=Path, required=True,
                    help='Read-only folder containing all_events.csv.')
    ap.add_argument('--out-dir', type=Path)
    ap.add_argument('--manuscript-dir', type=Path)
    args = ap.parse_args()
    repo = args.repo_root.resolve()
    manifest = args.manifest
    campaign = Path(json.loads(manifest.read_text())['out_root']).name
    out = args.out_dir or repo/'results/earths_future_revision'/campaign.replace('production_', 'publication_')
    manuscript = args.manuscript_dir
    here = Path(__file__).resolve().parent
    commands = [
        ['validate_publication_runs.py', '--repo-root', repo, '--manifest', manifest, '--out-dir', out],
        ['publication_tables.py', '--repo-root', repo, '--index', out/'validated_runs.json', '--out-dir', out/'tables'],
        ['publication_figures.py', '--repo-root', repo, '--index', out/'validated_runs.json', '--out-dir', out/'figures', '--physical-data', args.physical_data],
        ['publication_values.py', '--index', out/'validated_runs.json', '--out-dir', out],
    ]
    for option, value in [('--campaign-root', args.campaign_root), ('--log-dir', args.log_dir)]:
        if value is not None:
            commands[0].extend([option, value])
    for script, *options in commands:
        print(f'Running {script}', flush=True)
        subprocess.run([sys.executable, str(here/script), *map(str, options)], check=True)
    if manuscript is not None:
        install_artifacts(out, manuscript)
    print(f'Regenerated 9 publication figures and numerical tables in {out}')
    print(f'Validation, source hashes and numerical results in {out}')


if __name__ == '__main__':
    main()

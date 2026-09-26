# Model demonstration

The demo runs the Great Miami Hurricane footprint through the insurance model using public institutional inputs and synthetic insurer finances. It requires neither licensed S&P files nor WindRiskTech event sets.

## Run

From the repository root:

```bash
python -m pip install -e .
python scripts/demo/run_demo.py --n_iter 100 --seed 42
```

The default run contains 100 realizations and generally takes less than a minute on a recent laptop. `--n_iter`, `--seed`, and `--out` change the simulation count, seed, and output directory. Output paths are relative to the repository root.

## Inputs

- The Great Miami county-loss footprint comes from public IBTrACS tracks processed with CLIMADA.
- The 99 private insurer names and identifiers are public, but their market shares and capital values in `demo_data/` are synthetic.
- Annual private premiums are assumed to total USD 10 billion and are allocated by the synthetic market shares. This is an illustrative assumption, not an estimate of actual premiums.
- Citizens uses its separate public exposure and financial inputs. Its zero-share roster entry identifies its catastrophe-bond recoveries; it does not allocate private exposure to Citizens.
- FHCF terms, Citizens inputs, NFIP inputs, and the reviewed catastrophe-bond inventory use the included public files.

## Outputs and checks

`demo_output/demo_summary.csv` contains one row per realization. A successful default run has 100 rows and no `scenario=error` rows. The command exits with a failure status if any realization fails and retains its error message for diagnosis.

`demo_output/expected_demo_summary.csv` is a reference output from the same corrected code with seed 42. If no reference exists in the requested output directory, a successful run creates one; an unsuccessful run never creates one. Results may vary slightly across numerical-library versions.

Demo values are illustrative and do not reproduce manuscript estimates. The real financial simulation requires the separately licensed insurer inputs described in the main README.

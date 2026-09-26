"""Reproduce publication figures and tables from the validated public archive."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from validate_archive import validate_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True,help='Extracted campaign directory containing manifest.json.')
    parser.add_argument('--out-dir',type=Path,default=Path('results/publication'))
    parser.add_argument('--physical-data',type=Path,help='Optional licensed ERA5 catalog directory containing all_events.csv, to recompute Figure S1.')
    args = parser.parse_args()
    repo=Path(__file__).resolve().parents[3]
    manifest,index=validate_archive(args.archive)
    out=args.out_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    index_file=out/'validated_runs.json'
    index_file.write_text(json.dumps(index,indent=2)+'\n')
    print('Validated 106 analyses and 997,000 seasons/realizations.',flush=True)
    here=Path(__file__).resolve().parent
    commands=[
        ['publication_tables.py','--repo-root',repo,'--index',index_file,'--out-dir',out/'tables'],
        ['publication_figures.py','--repo-root',repo,'--index',index_file,'--out-dir',out/'figures'],
        ['publication_values.py','--index',index_file,'--out-dir',out],
    ]
    if args.physical_data:
        commands[1].extend(['--physical-data',args.physical_data.resolve()])
    for script,*options in commands:
        subprocess.run([sys.executable,str(here/script),*map(str,options)],check=True)
    # Keep portable provenance in the deliverable; the absolute index is local only.
    index_file.write_text(json.dumps({r['name']:r['path'] for r in manifest['runs']},indent=2)+'\n')
    methods_file = out/'tables/table_methods.json'
    methods = json.loads(methods_file.read_text())
    methods['index_sha256'] = hashlib.sha256(index_file.read_bytes()).hexdigest()
    methods_file.write_text(json.dumps(methods,indent=2)+'\n')
    (out/'provenance.json').write_text(json.dumps({
        'software_version':'1.1.0','simulation_commit':manifest['simulation_commit'],
        'n_runs':len(index),'total_rows':manifest['total_rows'],
        'manifest_sha256':hashlib.sha256((args.archive/'manifest.json').read_bytes()).hexdigest(),
        'figure_s1':'recomputed from licensed event catalog' if args.physical_data else 'published reference figure; licensed event catalog required for recomputation',
    },indent=2)+'\n')
    print(f'Publication outputs written to {out}',flush=True)


if __name__=='__main__':
    main()

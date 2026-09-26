"""Checks for the publication inventory and fail-closed archive validation."""
import hashlib
import json
from pathlib import Path

import pytest

from scripts.cluster.campaign_inventory import build_job_list
from scripts.analysis.publication.validate_archive import validate_archive, SIMULATION_COMMIT


def test_campaign_inventory():
    jobs = build_job_list(Path('runs'),Path('inputs'))
    assert len(jobs) == len({j['name'] for j in jobs}) == 106
    assert sum(j['expected_seasons'] for j in jobs) == 997000
    assert sum(j['name'].startswith('buildingcode_') for j in jobs) == 65
    assert sum(j['name'].startswith('historical_') for j in jobs) == 7
    for job in jobs:
        assert job['cmd'][job['cmd'].index('--seed')+1] == '42'


@pytest.fixture
def archive(tmp_path):
    jobs=build_job_list(Path('runs'),Path('inputs'))
    path=tmp_path/'first.csv'
    path.write_text('scenario,year_id,total_damage_usd\nyear_1,1,0\n')
    manifest={'n_runs':106,'total_rows':997000,'software_version':'1.1.0','simulation_commit':SIMULATION_COMMIT,'runs':[
        {'name':j['name'],'path':'first.csv','expected_rows':j['expected_seasons'],
         'execution_commit':SIMULATION_COMMIT,'execution_exit_code':0,
         'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for j in jobs]}
    return tmp_path,manifest


@pytest.mark.parametrize('case,message',[
    ('duplicate','106 distinct'),('missing','106 distinct'),
    ('checksum','Checksum mismatch'),('path','escapes its root'),
    ('execution','Inconsistent source execution'),('rows','Incomplete run'),
    ('version','Archive version does not match'),
])
def test_archive_rejects_invalid_inputs(archive,case,message):
    root,manifest=archive
    first=manifest['runs'][0]
    if case=='duplicate':manifest['runs'][-1]=dict(first)
    elif case=='missing':manifest['runs'].pop()
    elif case=='checksum':first['sha256']='0'*64
    elif case=='path':first['path']='../outside.csv'
    elif case=='execution':first['execution_commit']='different'
    elif case=='version':manifest['simulation_commit']='earlier-campaign'
    (root/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match=message):
        validate_archive(root)

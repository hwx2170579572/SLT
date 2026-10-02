import copy
import pytest
from tools.analyze_phase2_reuse_v3 import read,OUT,CONTRACT,check_manifest,historical_result,ready_to_select


def test_real_manifest_and_real_reference():
    c=read(CONTRACT);m=read(OUT/'execution_manifest.json');check_manifest(m,c)
    d=historical_result(m['historical_controls'][0],c)
    assert d['summary']['episodes']==100 and d['kind']=='exact_final'


@pytest.mark.parametrize('field',['historical_controls','jobs'])
def test_missing_or_duplicated_cells_fail(field):
    c=read(CONTRACT);m=read(OUT/'execution_manifest.json');m[field][-1]=m[field][0]
    with pytest.raises(ValueError,match='coverage'):check_manifest(m,c)


def test_reference_hash_tampering_fails():
    r=read(OUT/'execution_manifest.json')['historical_controls'][0];r['result_sha256']='wrong'
    with pytest.raises(ValueError,match='hash'):historical_result(r,read(CONTRACT))


def test_relabeling_reference_is_not_accepted():
    r=read(OUT/'execution_manifest.json')['historical_controls'][0];r['method']='v4_13'
    with pytest.raises(ValueError,match='identity'):historical_result(r,read(CONTRACT))


def test_partial_extra_or_invalid_results_cannot_select():
    assert not ready_to_select({'a','b'},{'a'},[])
    assert not ready_to_select({'a'},{'a','extra'},[])
    assert not ready_to_select({'a'},{'a'},['bad'])
    assert ready_to_select({'a'},{'a'},[])

import pytest
from tools.prepare_phase2_confirmation_v1 import confirmation_jobs,require_complete,CONTRACT,read


def test_two_selected_candidates_give_60_paired_jobs():
    c=read(CONTRACT);jj=confirmation_jobs(c,{'v4_8':'lr_half','v4_13':'tau_half'})
    assert len(jj)==60
    assert {j['seed'] for j in jj}=={1,2}
    assert {j['candidate']['id'] for j in jj if j['method']=='mst_slt'}=={'control'}
    for m in c['optimization_methods']:
        for s in c['scenarios']:
            for seed in (1,2):
                assert len([j for j in jj if j['method']==m and j['scenario']==s and j['seed']==seed])==2


def test_default_selection_is_not_duplicated():
    jj=confirmation_jobs(read(CONTRACT),{'v4_8':'control','v4_13':'control'})
    assert len(jj)==36 and len({j['id'] for j in jj})==36


@pytest.mark.parametrize('selection',[{'v4_8':'control'},{'v4_8':'control','v4_13':'unknown'}, {'v4_8':'control','v4_13':'control','temporal_graph':'control'}])
def test_invalid_selection_is_rejected(selection):
    with pytest.raises(ValueError):confirmation_jobs(read(CONTRACT),selection)


@pytest.mark.parametrize('completion',[{'complete':False,'completed':65,'expected':66}, {'complete':True,'completed':66,'expected':66,'invalid':[{'id':'bad'}]}, {'complete':True,'completed':66,'expected':66,'missing':['missing']}])
def test_partial_or_invalid_screen_cannot_unlock(completion):
    with pytest.raises(ValueError,match='66 valid'):require_complete(completion)

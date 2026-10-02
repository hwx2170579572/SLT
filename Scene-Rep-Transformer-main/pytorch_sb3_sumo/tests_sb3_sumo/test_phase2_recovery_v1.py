import copy
import pytest
from tools.recover_phase2_evaluation_v1 import inspect,validate_block,ROOT


def fixture():
    directory=ROOT/'results_phase2_runtime_v2/smoke/v4_13__cross__control__seed0'
    return inspect(directory)


def test_completed_smoke_is_readable_without_retraining():
    a,c,r,blocks,missing=fixture()
    assert a['smoke'] and c['learner_updates']==25 and not missing and list(blocks)==[0]


@pytest.mark.parametrize('field,value',[('checkpoint_sha256','wrong'),('decoder','wrong'),('smoke',False)])
def test_wrong_block_identity_rejected(field,value):
    a,c,r,blocks,_=fixture();d=copy.deepcopy(blocks[0]);d[field]=value
    with pytest.raises(ValueError,match='mismatch'):
        validate_block(d,count=1,offset=0,checkpoint_hash=c['checkpoint_sha256'],decoder=a['decoder'],smoke=True,reference=r)


def test_wrong_traffic_rejected():
    a,c,r,blocks,_=fixture();d=copy.deepcopy(blocks[0]);d['episode_records'][0]['traffic_variant']='wrong'
    with pytest.raises(ValueError,match='traffic'):
        validate_block(d,count=1,offset=0,checkpoint_hash=c['checkpoint_sha256'],decoder=a['decoder'],smoke=True,reference=r)

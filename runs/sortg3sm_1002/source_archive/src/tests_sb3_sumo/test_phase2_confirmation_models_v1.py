from tools.audit_phase2_confirmation_models_v1 import differences,metadata_errors,TRAIN_FIELDS


def test_missing_and_changed_protocol_not_silently_accepted():
    ref={k:1 for k in TRAIN_FIELDS};old=dict(ref)
    del old['traffic_protocol'];old['action_repeat']=2
    assert set(differences(old,ref))=={'traffic_protocol','action_repeat'}


def test_later_checkpoint_and_wrong_seed_rejected():
    job=dict(seed=1,candidate=dict(learning_rate=1e-4,tau=.005))
    model=dict(_raw_steps_seen=50000,_n_updates=45001,seed=1,learning_rate=1e-4,tau=.005,
               batch_size=32,buffer_size=20000,gamma=.99,gradient_steps=3)
    assert not metadata_errors(model,job)
    model.update(_raw_steps_seen=100000,seed=2)
    assert set(metadata_errors(model,job))=={'_raw_steps_seen','seed'}

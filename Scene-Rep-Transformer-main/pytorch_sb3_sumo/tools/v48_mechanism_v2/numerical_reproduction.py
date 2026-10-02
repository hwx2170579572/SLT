"""Isolate encoding batch/mode/loading effects without training or simulation."""
from __future__ import annotations

import argparse

from .common import cpu_environment
cpu_environment()
import numpy as np
import torch

from .common import OUTPUT,configure_torch,probe_dataset_path,load_json,write_json,metadata,sha256
from .models import load_model,model_path,tensor_digest


def direct_encode(scene,data,batch_size,training):
    from algos.sb3_torch.evaluation import source_evaluation_augmentation
    model=load_model(scene,"v4_8_lr_half")
    model.policy.set_training_mode(training)
    before=tensor_digest(model)
    items={}
    with source_evaluation_augmentation(model),torch.no_grad():
        for start in range(0,len(data["trajectory"]),batch_size):
            obs={k:torch.tensor(data[k][start:start+batch_size]) for k in ("trajectory","map")}
            latent=model.critic.features_extractor.forward_tokens(obs)
            for key,value in dict(full=latent.tensor,ego=latent.z_ego,social=latent.z_social,route=latent.z_route).items():
                items.setdefault(key,[]).append(value.numpy())
    assert tensor_digest(model)==before
    return {key:np.concatenate(value) for key,value in items.items()}


def main(scene):
    from tools.run_latent_probes import _encode,_episode_split,_ridge_regression,_ridge_classification
    configure_torch()
    with np.load(probe_dataset_path(scene)) as archive:
        data={key:archive[key] for key in archive.files}
    reference=load_json(probe_dataset_path(scene).with_name("probe_result.json"))
    train,test,split=_episode_split(data["episode_id"],seed=73,train_fraction=.7)
    variants={}
    for batch_size,training in ((128,False),(256,False),(256,True)):
        name=f"direct_cpu_batch{batch_size}_{'train' if training else 'eval'}"
        variants[name]=direct_encode(scene,data,batch_size,training)
        print("encoded "+name,flush=True)
    # This legacy function is used strictly as an encoder numerical reference.
    # No predict/actor/decoder call and no env.reset/SUMO rollout occurs.
    original,legacy_meta=_encode("v4_8",model_path(scene,"v4_8_lr_half"),scene,
        data["trajectory"],data["map"],device="cpu",batch_size=256)
    variants["original_encode_cpu_batch256"]=original
    results={}
    base=variants["direct_cpu_batch128_eval"]
    for name,features in variants.items():
        rows=[]
        for slot,x in features.items():
            for target in ("ego_dynamics","minimum_distance","minimum_ttc"):
                actual=_ridge_regression(x,data[target],train,test,.001)["r2_mean"]
                expected=reference["results"]["v4_8"][slot][target]["r2_mean"]
                rows.append(dict(slot=slot,target=target,actual=actual,expected=expected,
                    absolute_difference=abs(actual-expected),passes_original_1e_5_tolerance=abs(actual-expected)<1e-5))
        results[name]=dict(metrics=rows,max_score_difference=max(row["absolute_difference"] for row in rows),
            latent_max_abs_from_cpu128={slot:float(np.max(np.abs(x-base[slot]))) for slot,x in features.items()},
            latent_rmse_from_cpu128={slot:float(np.sqrt(np.mean((x-base[slot])**2))) for slot,x in features.items()})
    write_json(OUTPUT / "stage2/reproduction" / f"{scene}.json",dict(**metadata(),scene=scene,
        dataset_sha256=sha256(probe_dataset_path(scene)),checkpoint_sha256=sha256(model_path(scene,"v4_8_lr_half")),
        split=split,results=results,legacy_encoder_metadata=legacy_meta,no_policy_prediction=True,no_training=True,no_simulation=True))
    print("numerical reproduction audit complete",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--scene",choices=("cross","carla"),default="cross")
    main(parser.parse_args().scene)

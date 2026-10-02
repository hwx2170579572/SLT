"""Action branches from verified reset-and-prefix replay, not partial state restore."""
from __future__ import annotations

import argparse
import time

from .common import cpu_environment
cpu_environment()
import numpy as np
import torch

from .common import OUTPUT, metadata, configure_torch, load_json, write_json, sha256
from .collect import make_environment, seal, identity_hash, ENCODERS
from .models import load_model, filter_observation, tensor_digest
from .features import encode
from .probes import grouped_split,fit_probe
from .control_analysis import action_features


def select_snapshots(data, smoke=False):
    nominal=10 if smoke else 30
    result=[]
    if len(data["action"])>nominal:
        result.append(("nominal",nominal))
    if not smoke:
        candidate=np.flatnonzero((np.arange(len(data["action"]))>=30) &
            (data["observed_distance"]<=10) & (data["closest_approach_time"]<=3))
        if len(candidate) and int(candidate[0])!=nominal:
            result.append(("interaction_proxy",int(candidate[0])))
    return result


def assert_observation_equal(actual,data,index,prefix=""):
    maximum=0.
    for key in ("trajectory","map","lane_action_mask"):
        expected=data[prefix+key][index]
        maximum=max(maximum,float(np.max(np.abs(actual[key]-expected))))
        np.testing.assert_allclose(actual[key],expected,rtol=0,atol=1e-6,err_msg=f"Replay {index}/{key}")
    return maximum


def replay_prefix(env,wrapped,data,receipt,index,seed,point):
    np.random.seed(seed+600000)
    torch.manual_seed(seed+600000)
    env._traffic_episode_index,env._traffic_roll=index,None
    obs,_=wrapped.reset(seed=seed)
    traffic=wrapped.traffic_identity()
    assert all(traffic[key]==receipt["traffic"][key] for key in ("seed","route_sha256","network_sha256"))
    error=0.
    for step in range(point):
        error=max(error,assert_observation_equal(obs,data,step))
        obs,reward,terminated,truncated,info=wrapped.step(data["action"][step])
        assert not terminated and not truncated
        assert abs(float(reward)-float(data["reward"][step]))<1e-8
        assert int(info["raw_steps_executed"])==int(data["raw_executed"][step])
    error=max(error,assert_observation_equal(obs,data,point))
    return obs,error


def run(smoke=False):
    from tools.v48_stability_v1.driving import DrivingTrace
    from algos.sb3_torch.evaluation import source_evaluation_augmentation
    configure_torch()
    control=OUTPUT / ("smoke/control" if smoke else "stage2/control")
    assert load_json(control / "complete.json")["traffic_pairing_verified"]
    folder=OUTPUT / ("smoke/branches" if smoke else "stage2/branches")
    protocol=dict(contract="v48-fixed-continuation-action-branches/v2",smoke=smoke,
        control_protocol_sha256=sha256(control / "protocol.json"),source_behavior="mst_slt",
        continuation="v4_8_lr_half final deterministic actor",first_traffic_seeds=1 if smoke else 12,
        nominal_decision=10 if smoke else 30,interaction="first decision >=30 with observed distance <=10m and closest-approach-time proxy <=3s; duplicates removed",
        candidate_actions=[[-.5,0],[0,0],[.5,0],[0,-1],[0,1]],include_observed_first_action=True,
        candidate_mask="actual physical lane mask at branch point",continuation_horizon=4 if smoke else 32,
        restoration="reset SUMO/entire environment from original seed, replay all original actions; compare every prefix observation/reward/raw-step count",
        prefix_atol=1e-6,ranking_targets=["discounted_environment_return","uncensored_observed_progress_m"],
        code_sha256=sha256(__file__),resource_policy="CPU1; SUMO1; no policy training; no CUDA")
    seal(folder / "protocol.json",protocol)
    fingerprint=identity_hash(protocol)
    cp=load_json(control / "protocol.json")
    started=time.time()
    for scene in cp["scenes"]:
        output=folder / scene
        output.mkdir(parents=True,exist_ok=True)
        env=make_environment(scene,("smoke_branch_" if smoke else "branch_")+scene)
        wrapped=DrivingTrace(env)
        model=load_model(scene,"v4_8_lr_half")
        original=tensor_digest(model)
        records,observations=[],[]
        try:
            with source_evaluation_augmentation(model):
                for index in range(protocol["first_traffic_seeds"]):
                    source=control / scene / "mst_slt" / f"e{index:03d}.json"
                    receipt=load_json(source)
                    path=source.with_name(receipt["data"])
                    assert sha256(path)==receipt["data_sha256"]
                    with np.load(path) as archive:
                        data={k:archive[k] for k in archive.files}
                    snapshots=select_snapshots(data,smoke)
                    for kind,point in snapshots:
                        mask=data["lane_action_mask"][point]
                        actions=[("observed_first",data["action"][point])]
                        for number,a in enumerate(protocol["candidate_actions"]):
                            if mask[int(a[1])+1]>.5:
                                actions.append((f"grid{number}",np.array(a,dtype=np.float32)))
                        for action_id,action in actions:
                            output_path=output / f"e{index:03d}_{kind}_{action_id}.json"
                            identity=dict(protocol_sha256=fingerprint,source_data_sha256=receipt["data_sha256"],
                                seed=receipt["record"]["seed"],point=point,snapshot=kind,action_id=action_id,action=action.tolist())
                            if output_path.exists():
                                record=load_json(output_path)
                                assert record["identity"]==identity
                            else:
                                obs,error=replay_prefix(env,wrapped,data,receipt,index,identity["seed"],point)
                                initial=wrapped.initial if not wrapped.samples else wrapped.samples[-1]
                                start_distance=initial["distance"]
                                rewards=[]
                                events=dict(collision=False,timeout=False,success=False,off_route=False)
                                elapsed=0
                                next_match=False
                                for horizon in range(protocol["continuation_horizon"]):
                                    selected=action if horizon==0 else model.predict(filter_observation(model,obs),deterministic=True)[0]
                                    obs,reward,terminated,truncated,info=wrapped.step(selected)
                                    if horizon==0 and action_id=="observed_first":
                                        error=max(error,assert_observation_equal(obs,data,point,prefix="next_"))
                                        assert abs(float(reward)-float(data["reward"][point]))<1e-8
                                        next_match=True
                                    rewards.append(float(reward))
                                    elapsed+=int(info["raw_steps_executed"])
                                    events.update(collision=bool(info.get("collision",False)),timeout=bool(info.get("max_time",False)),
                                                  success=bool(info.get("is_success",False)),off_route=bool(info.get("off_route",False)))
                                    if terminated or truncated:
                                        break
                                last=wrapped.samples[-1]
                                progress=float(last["distance"]-start_distance) if last is not None else None
                                record=dict(identity=identity,prefix_max_abs_error=error,prefix_verified=True,
                                    observed_first_next_state_verified=next_match if action_id=="observed_first" else None,
                                    discounted_environment_return=sum(.99**i*r for i,r in enumerate(rewards)),
                                    observed_progress_m=progress,progress_censored=last is None,
                                    raw_steps=elapsed,decisions=len(rewards),**events)
                                seal(output_path,record)
                            records.append(record)
                            observations.append({k:data[k][point] for k in ("trajectory","map","lane_action_mask")})
                        write_json(folder / "progress.json",dict(scene=scene,episode=index+1,snapshot=kind,
                            completed_branches=len(records),elapsed_seconds=time.time()-started))
                        print(f"branches {scene}/seed{index}/{kind}: {len(records)}",flush=True)
            assert original==tensor_digest(model)
        finally:
            env.close()
        arrays={k:np.stack([row[k] for row in observations]) for k in observations[0]}
        arrays["action"]=np.array([row["identity"]["action"] for row in records],dtype=np.float32)
        arrays["group_id"]=np.array([row["identity"]["seed"] for row in records])
        arrays["snapshot_id"]=np.array([str(row["identity"]["seed"])+"_"+row["identity"]["snapshot"] for row in records])
        arrays["return"]=np.array([row["discounted_environment_return"] for row in records])
        arrays["progress"]=np.array([row["observed_progress_m"] if row["observed_progress_m"] is not None else np.nan for row in records])
        arrays["collision"]=np.array([row["collision"] for row in records])
        arrays["timeout"]=np.array([row["timeout"] for row in records])
        np.savez_compressed(output / "dataset.npz",**arrays)
        write_json(output / "result.json",dict(**metadata(),protocol_sha256=fingerprint,records=records,
            dataset_sha256=sha256(output / "dataset.npz"),prefixes_all_verified=True,tensors_unchanged=True,
            missing_snapshot_coverage="Only available predeclared states; absent interaction strata are not silently replaced"))
    write_json(folder / "complete.json",dict(**metadata(),smoke=smoke,protocol_sha256=fingerprint,prefixes_all_verified=True,
        elapsed_seconds=time.time()-started))


def ranking_accuracy(y,prediction,snapshots,mask,min_difference):
    correct=total=0
    per_snapshot=[]
    for snapshot in np.unique(snapshots[mask]):
        indices=np.flatnonzero(mask & (snapshots==snapshot) & np.isfinite(y))
        c=n=0
        for left,index in enumerate(indices):
            for other in indices[left+1:]:
                truth=y[index]-y[other]
                if abs(truth)<=min_difference:
                    continue
                estimate=float(prediction[index,0]-prediction[other,0])
                c+=float((estimate*truth)>0) if abs(estimate)>1e-12 else .5
                n+=1
        correct+=c
        total+=n
        if n:
            per_snapshot.append(c/n)
    return dict(pairwise_accuracy=correct/total if total else None,eligible_pairs=total,
        mean_snapshot_accuracy=float(np.mean(per_snapshot)) if per_snapshot else None,
        identifiable_snapshots=len(per_snapshot),minimum_label_difference=min_difference)


def analyze():
    configure_torch()
    folder=OUTPUT / "stage2/branches"
    assert load_json(folder / "complete.json")["prefixes_all_verified"]
    for scene in ("cross","carla"):
        path=folder / scene / "dataset.npz"
        with np.load(path) as archive:
            data={k:archive[k] for k in archive.files}
        masks,split=grouped_split(data["group_id"])
        results={}
        for name in ENCODERS:
            feature,_=encode(scene,name,{k:data[k] for k in ("trajectory","map","lane_action_mask")},sha256(path),"branch")
            x=np.concatenate([feature["full"],action_features(data["action"])],axis=1)
            for target,margin in (("return",.01),("progress",.5)):
                for head in ("linear","mlp"):
                    prediction,result=fit_probe(x,data[target],masks,head=head)
                    if prediction is not None:
                        eligible=masks[2] & (~data["collision"] if target=="progress" else True)
                        result["action_ranking"]=ranking_accuracy(data[target],prediction,data["snapshot_id"],eligible,margin)
                    results[f"{head}__{target}__{name}"]=result
        write_json(folder / scene / "probe_result.json",dict(**metadata(),dataset_sha256=sha256(path),split=split,results=results,
            limitations=["Small predeclared 12-seed branch diagnostic; many return ties may make reward ranking unidentifiable",
                "Fixed full-policy continuation, never optimal Q; no model-specific label generation",
                "Progress ranking excludes collision branches and censored positions; safety outcomes reported separately",
                "Every action prefix was replayed and verified including wrapper history; no partial snapshot restoration"] ))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--analyze",action="store_true")
    args=parser.parse_args()
    analyze() if args.analyze else run(args.smoke)

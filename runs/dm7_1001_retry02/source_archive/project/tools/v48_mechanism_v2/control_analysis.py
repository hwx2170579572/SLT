"""Action-conditioned future labels on a shared mixture of frozen behavior policies."""
from __future__ import annotations

from .common import cpu_environment
cpu_environment()
import numpy as np
from sklearn.decomposition import PCA

from .common import OUTPUT, SCENES, metadata, configure_torch, write_json
from .collect import BEHAVIORS, ENCODERS
from .control_data import load_control
from .features import encode, geometry, linear_cka
from .probes import grouped_split, fit_probe, paired_error_interval, standardize, regression_metrics, binary_metrics


def action_features(actions):
    lanes=np.where(actions[:,1]<-1/3,0,np.where(actions[:,1]>1/3,2,1))
    return np.concatenate([actions[:,:1],np.eye(3)[lanes]],axis=1)


def physical_reference(data):
    keys=("speed","acceleration","lane","route_index","raw_start","observed_distance","closest_approach_time",
          "following_gap","following_ttc")
    columns=np.column_stack([data[k].astype(float) for k in keys])
    missing=~np.isfinite(columns)
    columns[missing]=0
    return np.concatenate([columns,missing.astype(float),data["lane_action_mask"]],axis=1),keys


def main():
    from tools.v48_stability_v1.driving import aggregate_driving
    configure_torch()
    for scene in SCENES:
        data,labels,episodes,digest=load_control(scene)
        masks,split=grouped_split(data["group_id"])
        features={name:encode(scene,name,{k:data[k] for k in ("trajectory","map","lane_action_mask")},digest,"control")[0]
                  for name in ENCODERS}
        x={name:values["full"] for name,values in features.items()}
        raw=np.concatenate([data[k].reshape(len(data[k]),-1) for k in ("trajectory","map")],axis=1)
        raw_scaled=standardize(raw,masks[0])
        pca=PCA(n_components=128,svd_solver="randomized",random_state=73).fit(raw_scaled[masks[0]])
        x["raw_pca128"]=pca.transform(raw_scaled)
        x["physical_reference"],physical_keys=physical_reference(data)
        behavior=np.eye(len(BEHAVIORS))[data["behavior_id"]]
        action=action_features(data["action"])
        targets=[f"{kind}_h{h}" for kind in ("progress_m","collision","timeout") for h in (1,4,16,32)]
        results,predictions,comparisons={},{},{}
        for target in targets:
            kind="regression" if target.startswith("progress") else "binary"
            # All four horizons have linear diagnostics. Nonlinear capacity and
            # action-omission controls concentrate on the predeclared 16 horizon.
            heads=("linear","mlp") if target.endswith("_h16") else ("linear",)
            for head in heads:
                for name,values in x.items():
                    conditions=(True,False) if name in ("mst_slt","v4_8_lr_half") and target.endswith("_h16") else (True,)
                    for with_action in conditions:
                        inputs=np.concatenate([values,behavior,*([action] if with_action else [])],axis=1)
                        key=f"{head}__{target}__{name}__{'action' if with_action else 'state'}"
                        prediction,result=fit_probe(inputs,labels[target],masks,kind=kind,head=head)
                        results[key]=result
                        if prediction is not None:
                            predictions[key]=prediction
                print(f"control probes {scene}/{head}/{target}",flush=True)
        for target in targets:
            for head in ("linear","mlp"):
                firstkey=f"{head}__{target}__mst_slt__action"
                if firstkey not in predictions:
                    continue
                for name in ENCODERS[1:]:
                    key=f"{head}__{target}__{name}__action"
                    if key not in predictions:
                        continue
                    comparison=paired_error_interval(labels[target],predictions[firstkey],predictions[key],
                        data["group_id"],masks[2],np.asarray(results[firstkey]["fit_only_target_scale"]))
                    comparisons[key+"_vs_mst"]=comparison
        strata={"near_observed_vehicle_le10m":data["observed_distance"]<=10,
                "far_observed_vehicle_gt10m":data["observed_distance"]>10,
                "non_keep_feasible":data["lane_action_mask"][:,[0,2]].any(1),
                "only_keep_feasible":~data["lane_action_mask"][:,[0,2]].any(1),
                "route_change_intent":data["route_intent_valid"] & (data["route_intent"]!=0),
                "other_route_intent":~(data["route_intent_valid"] & (data["route_intent"]!=0))}
        stratified={}
        for key,prediction in predictions.items():
            head,target,name,condition=key.split("__")
            if not target.endswith("_h16") or condition!="action":
                continue
            y=labels[target].reshape(-1,1)
            for stratum,member in strata.items():
                take=masks[2] & member & np.isfinite(y[:,0])
                if take.sum()<2:
                    continue
                score=regression_metrics(y[take],prediction[take]) if target.startswith("progress") else binary_metrics(y[take],prediction[take])
                stratified[key+"__"+stratum]=dict(samples=int(take.sum()),traffic_seeds=len(np.unique(data["group_id"][take])),scores=score)
        behavior_summary={}
        for name in BEHAVIORS:
            selected=[episode for episode in episodes if episode["behavior"]==name]
            behavior_summary[name]=dict(episodes=len(selected),
                **{outcome+"_rate":float(np.mean([episode["record"][outcome] for episode in selected]))
                   for outcome in ("success","collision","timeout","off_route")},
                driving=aggregate_driving(selected))
        coverage={}
        for target in targets:
            y=labels[target]
            coverage[target]=[dict(samples=int((mask & np.isfinite(y)).sum()),
                positive_rows=int(np.sum(y[mask & np.isfinite(y)]>0)),
                event_seed_count=int(len(np.unique(data["group_id"][mask & (y>0)])))) for mask in masks]
        report=dict(**metadata(),dataset_digest=digest,split=split,coverage=coverage,probes=results,
            paired_intervals=comparisons,stratified=stratified,behavior_summary=behavior_summary,
            geometry={name:geometry(item) for name,item in features.items()},
            cka_to_mst={name:linear_cka(features["mst_slt"]["full"],item["full"]) for name,item in features.items()},
            references=dict(raw_pca128="fit-only transform; same history as learned encoders",physical_reference=list(physical_keys)),
            limitations=["behavior ID is supplied to every probe; association is conditional on continuation behavior",
                "h16 has matched linear/MLP and action-omission controls; h1/h4/h32 are linear horizon screening",
                "missing terminal positions are censored for progress regression; report event outcomes jointly",
                "few event seeds limit risk evidence even when many adjacent frames have positive labels",
                "class frequencies are natural in this fixed equal-episode behavior mixture, not calibrated deployment risk",
                "current models differ beyond representation; conditional evidence only"])
        folder=OUTPUT / "stage2/future" / scene
        folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder / "predictions.npz",**predictions)
        write_json(folder / "result.json",report)
        print(f"complete future probes: {scene}",flush=True)


if __name__=="__main__":
    main()

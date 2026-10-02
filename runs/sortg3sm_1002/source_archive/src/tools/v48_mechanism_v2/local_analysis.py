"""Reproduce old local probes, then compare matched linear/nonlinear readability."""
from __future__ import annotations

from .common import cpu_environment
cpu_environment()
import numpy as np
from sklearn.decomposition import PCA

from .common import ROOT, OUTPUT, SCENES, metadata, configure_torch, sha256, probe_dataset_path, load_json, write_json
from .models import model_path
from .collect import ENCODERS
from .features import encode, geometry, linear_cka
from .probes import grouped_split, fit_probe, paired_error_interval, standardize


def main():
    from tools.run_latent_probes import _episode_split, _ridge_regression, _ridge_classification
    configure_torch()
    assert load_json(OUTPUT / "stage0/audit.json")["historical_diagnostics_gate"]=="PASS_WITH_DECLARED_LIMITATIONS"
    for scene in SCENES:
        for name in ENCODERS:
            audited=load_json(OUTPUT / "stage0/models" / f"{scene}__{name}.json")
            assert sha256(model_path(scene,name))==audited["checkpoint"]["sha256"], "Checkpoint changed after stage0"
        completed=OUTPUT / "stage2/local" / scene / "result.json"
        if completed.exists():
            previous=load_json(completed)
            assert previous["dataset_sha256"]==sha256(probe_dataset_path(scene)), "Completed diagnostic data changed"
            print(f"reuse completed local probes: {scene}",flush=True)
            continue
        with np.load(probe_dataset_path(scene)) as archive:
            data={key:archive[key] for key in archive.files}
        dataset_id=sha256(probe_dataset_path(scene))
        features={name:encode(scene,name,{key:data[key] for key in ("trajectory","map")},dataset_id,"legacy")[0]
                  for name in ENCODERS}
        reference=load_json(probe_dataset_path(scene).with_name("probe_result.json"))
        assert reference["dataset_sha256"]==dataset_id, "Historical reference dataset differs"
        train,test,legacy_split=_episode_split(data["episode_id"],seed=73,train_fraction=.7)
        legacy={}
        reproduction=[]
        for name,item in features.items():
            legacy[name]={}
            for slot,values in item.items():
                if slot=="target":
                    continue
                scores={label:_ridge_regression(values,data[label],train,test,.001) for label in
                        ("ego_dynamics","minimum_distance","minimum_ttc")}
                scores["route_progress_class"]=_ridge_classification(values,data["route_progress"],train,test,.001)
                legacy[name][slot]=scores
                old_name="v4_8" if name=="v4_8_lr_half" else name
                if old_name in reference["results"]:
                    for label,score in scores.items():
                        key="accuracy" if label=="route_progress_class" else "r2_mean"
                        if key in score:
                            expected=reference["results"][old_name][slot][label][key]
                            delta=abs(score[key]-expected)
                            reproduction.append(dict(model=name,slot=slot,label=label,metric=key,expected=expected,
                                                     actual=score[key],abs_difference=delta,passed=delta<1e-5))
        # User explicitly requested that the observed ~1.7e-4 numeric score
        # differences be ignored. Preserve the strict comparison and the real
        # values; accept small differences without claiming bit-exact parity.
        disposition=dict(original_tolerance=1e-5,numeric_reference_comparison_is_blocking=False,
            user_requested_ignore_minor_numeric_differences=True,
            maximum_absolute_difference=max((row["abs_difference"] for row in reproduction),default=None),
            accepted=bool(reproduction) and all(np.isfinite(row["actual"]) for row in reproduction),
            identity_gate="Stage0 checkpoint hashes and archived dataset hash must match; reference scores are descriptive only")
        write_json(OUTPUT / "stage2/local" / scene / "legacy_reproduction_audit.json",
            dict(dataset_sha256=dataset_id,comparisons=reproduction,all_passed=bool(reproduction) and all(row["passed"] for row in reproduction),
                 disposition=disposition))
        assert disposition["accepted"], "Missing or non-finite historical comparison"
        masks,split=grouped_split(data["episode_id"])
        x={name:item["full"] for name,item in features.items()}
        raw=np.concatenate([data[k].reshape(len(data[k]),-1) for k in ("trajectory","map")],axis=1)
        raw_scaled=standardize(raw,masks[0])
        pca=PCA(n_components=128,svd_solver="randomized",random_state=73).fit(raw_scaled[masks[0]])
        x["raw_pca128"]=pca.transform(raw_scaled)
        results,predictions={},{}
        for head in ("linear","mlp"):
            for label in ("ego_dynamics","minimum_distance","minimum_ttc"):
                for name,values in x.items():
                    key=f"{head}__{label}__{name}"
                    prediction,result=fit_probe(values,data[label],masks,head=head)
                    results[key]=result
                    if prediction is not None:
                        predictions[key]=prediction
                print(f"local probe {scene}/{head}/{label}",flush=True)
        comparisons={}
        for head in ("linear","mlp"):
            for label in ("ego_dynamics","minimum_distance","minimum_ttc"):
                first=predictions[f"{head}__{label}__mst_slt"]
                scale=np.asarray(results[f"{head}__{label}__mst_slt"]["fit_only_target_scale"])
                for name in ENCODERS[1:]:
                    second=predictions[f"{head}__{label}__{name}"]
                    comparisons[f"{head}__{label}__{name}_vs_mst"]=paired_error_interval(
                        data[label],first,second,data["episode_id"],masks[2],scale)
        folder=OUTPUT / "stage2/local" / scene
        folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder / "predictions.npz",**predictions)
        report=dict(**metadata(),dataset_sha256=dataset_id,legacy_split=legacy_split,legacy_results=legacy,
            legacy_reproduction=reproduction,legacy_reproduction_disposition=disposition,split=split,probes=results,paired_intervals=comparisons,
            geometry={name:geometry(item) for name,item in features.items()},
            cka_to_mst={name:linear_cka(features["mst_slt"]["full"],item["full"]) for name,item in features.items()},
            raw_reference=dict(method="fit-only standardized PCA-128 + same probe",fit_only_explained_variance_ratio=float(pca.explained_variance_ratio_.sum()),
                not_a_theoretical_upper_bound=True),
            limitations=["minimum_ttc is historical closest-approach time, not collision TTC",
                "new fit22/val6/test12 probes differ from historical fit28/test12 probes; compare within protocol",
                "behavior distribution is MST only; cross-policy control diagnostics are separate",
                "whole-method, one-RL-training-seed comparison, not TTG/Graph-SLT causal effects"])
        write_json(folder / "result.json",report)
        print(f"complete local probes: {scene}",flush=True)


if __name__=="__main__":
    main()

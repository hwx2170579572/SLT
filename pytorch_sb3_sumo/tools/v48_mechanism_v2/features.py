"""Shared-observation encoding and descriptive representation geometry."""
from __future__ import annotations

import numpy as np
import torch

from .common import ROOT, OUTPUT, sha256, load_json, write_json
from .models import load_model, model_path, tensor_digest


def encode(scene, name, observations, dataset_identity, tag, batch_size=128):
    from algos.sb3_torch.evaluation import source_evaluation_augmentation
    folder = OUTPUT / "stage2/features" / tag / scene
    path = folder / f"{name}.npz"
    receipt_path = folder / f"{name}.json"
    identity = dict(dataset=dataset_identity, checkpoint_sha256=sha256(model_path(scene, name)),
                    encoding_sha256=sha256(__file__), loader_sha256=sha256(ROOT / "tools/v48_mechanism_v2/models.py"),
                    device="cpu", source_augmentation=False, policy_training=False)
    if receipt_path.exists():
        receipt = load_json(receipt_path)
        if receipt["identity"] != identity or receipt["data_sha256"] != sha256(path):
            raise ValueError("Feature cache identity mismatch")
        with np.load(path) as archive:
            return {k: archive[k] for k in archive.files}, receipt
    model = load_model(scene, name)
    before = tensor_digest(model)
    online, target = model.critic.features_extractor, model.critic_target.features_extractor
    items = {}
    with source_evaluation_augmentation(model), torch.no_grad():
        for start in range(0, len(observations["trajectory"]), batch_size):
            obs = {k: torch.as_tensor(v[start:start+batch_size], device="cpu") for k,v in observations.items()}
            if hasattr(online, "forward_tokens"):
                latent = online.forward_tokens(obs)
                result = dict(full=latent.tensor, ego=latent.z_ego, social=latent.z_social, route=latent.z_route)
            else:
                result = dict(full=online(obs))
            result["target"] = target(obs)
            for key, value in result.items():
                assert torch.isfinite(value).all(), (name, key, start)
                items.setdefault(key, []).append(value.cpu().numpy())
    assert tensor_digest(model) == before
    data = {k: np.concatenate(v) for k,v in items.items()}
    folder.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".tmp").open("wb") as stream:
        np.savez_compressed(stream, **data)
    path.with_suffix(".tmp").replace(path)
    receipt = dict(identity=identity, data_sha256=sha256(path), shapes={k:list(v.shape) for k,v in data.items()},
                   tensors_unchanged=True, auxiliary_target_uses_online=bool(model.representation_online_target_encoder))
    write_json(receipt_path, receipt)
    print(f"encoded {tag}/{scene}/{name}: {data['full'].shape}", flush=True)
    return data, receipt


def geometry(features):
    x = features["full"].astype(np.float64)
    centered = x - x.mean(0)
    cov = centered.T @ centered / max(1, len(x)-1)
    eigen = np.maximum(np.linalg.eigvalsh(cov), 0)
    probability = eigen / max(eigen.sum(), 1e-30)
    positive = probability[probability > 0]
    lag = features["target"].astype(np.float64) - x
    result = dict(effective_rank=float(np.exp(-np.sum(positive*np.log(positive)))),
        participation_ratio=float(eigen.sum()**2 / max(np.square(eigen).sum(), 1e-30)),
        feature_std_mean=float(x.std(0).mean()), dimensions_std_below_1e_5=int((x.std(0)<1e-5).sum()),
        online_target_relative_rmse=float(np.sqrt(np.mean(lag**2) / max(np.mean(x**2),1e-30))),
        online_target_mean_cosine=float(np.mean(np.sum(x*features["target"],axis=1) /
            np.maximum(np.linalg.norm(x,axis=1)*np.linalg.norm(features["target"],axis=1),1e-12))),
        interpretation="Final online/EMA-target discrepancy, not temporal drift; rank/CKA are not quality scores.")
    if "ego" in features:
        boundaries = {"ego":slice(0,32), "social":slice(32,96), "route":slice(96,128)}
        result["slot_cross_covariance"] = {}
        for left,right in (("ego","social"),("ego","route"),("social","route")):
            a,b = boundaries[left],boundaries[right]
            denominator = np.linalg.norm(cov[a,a])*np.linalg.norm(cov[b,b])
            result["slot_cross_covariance"][left+"_"+right] = float(np.linalg.norm(cov[a,b])**2/max(denominator,1e-30))
    return result


def linear_cka(x, y):
    x,y = x.astype(np.float64),y.astype(np.float64)
    x,y = x-x.mean(0),y-y.mean(0)
    return float(np.linalg.norm(x.T@y)**2 / max(np.linalg.norm(x.T@x)*np.linalg.norm(y.T@y),1e-30))

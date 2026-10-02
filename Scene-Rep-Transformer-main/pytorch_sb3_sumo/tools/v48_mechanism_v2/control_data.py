"""Future labels obey episode boundaries and observed-time censoring."""
from __future__ import annotations

import hashlib
import numpy as np

from .common import OUTPUT, sha256, load_json
from .collect import BEHAVIORS


def future_labels(episode, horizons=(1,4,16,32)):
    length=len(episode["reward"])
    labels={}
    for h in horizons:
        collision,timeout,success,progress,seconds,actual=[],[],[],[],[],[]
        for start in range(length):
            end=min(start+h,length)
            collision.append(bool(episode["collision"][start:end].any()))
            timeout.append(bool(episode["timeout"][start:end].any()))
            success.append(bool(episode["success"][start:end].any()))
            # Never treat a cached terminal observation / last visible position
            # as a measured final distance. The regression excludes censoring.
            progress.append(float(episode["distance_end"][end-1]-episode["distance_start"][start])
                            if episode["distance_end_observed"][end-1] else np.nan)
            seconds.append(float(episode["raw_executed"][start:end].sum()*.1))
            actual.append(end-start)
        for name,values in (("collision",collision),("timeout",timeout),("success",success),
                             ("progress_m",progress),("seconds",seconds),("actual_h",actual)):
            labels[f"{name}_h{h}"]=np.asarray(values)
    return labels


def load_control(scene):
    folder=OUTPUT / "stage2/control"
    complete=load_json(folder / "complete.json")
    assert complete["traffic_pairing_verified"] and complete["tensors_unchanged"]
    protocol=load_json(folder / "protocol.json")
    columns,labels,episodes={}, {}, []
    digest=hashlib.sha256()
    for behavior_index,name in enumerate(BEHAVIORS):
        for index in range(protocol["episodes_per_behavior"]):
            receipt=load_json(folder / scene / name / f"e{index:03d}.json")
            path=folder / scene / name / receipt["data"]
            if sha256(path)!=receipt["data_sha256"]:
                raise ValueError("Control dataset changed")
            digest.update(receipt["data_sha256"].encode())
            with np.load(path) as archive:
                data={k:archive[k] for k in archive.files}
            n=len(data["action"])
            data["group_id"]=np.full(n,receipt["record"]["seed"],dtype=np.int64)
            data["behavior_id"]=np.full(n,behavior_index,dtype=np.int64)
            data["episode_id"]=np.full(n,behavior_index*protocol["episodes_per_behavior"]+index,dtype=np.int64)
            for k,v in data.items():
                columns.setdefault(k,[]).append(v)
            for k,v in future_labels(data).items():
                labels.setdefault(k,[]).append(v)
            episodes.append(dict(behavior=name,**receipt))
    return {k:np.concatenate(v) for k,v in columns.items()}, {k:np.concatenate(v) for k,v in labels.items()}, episodes,digest.hexdigest()

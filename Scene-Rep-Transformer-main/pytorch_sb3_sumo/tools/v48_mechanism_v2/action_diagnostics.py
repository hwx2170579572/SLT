"""Compare each actor and its own critic on identical, physically masked states."""
from __future__ import annotations

from .common import cpu_environment
cpu_environment()
import numpy as np
import torch

from .common import OUTPUT,SCENES,configure_torch,metadata,write_json
from .models import load_model,observation_tensors,tensor_digest
from .collect import ENCODERS
from .control_data import load_control


def q_from_features(critic,features,actions):
    if hasattr(critic,"forward_from_features"):
        return critic.forward_from_features(features,actions)
    return tuple(network(torch.cat([features,encoder(actions)],dim=1))
                 for network,encoder in zip(critic.q_networks,critic.action_encoders))


def summarize_arrays(arrays, mask):
    if not mask.any():
        return dict(samples=0)
    result=dict(samples=int(mask.sum()))
    for name in ("normalized_speed","lane_confidence","lane_entropy","own_grid_best_minus_actor_q","own_grid_q_spread","infeasible_lane"):
        values=arrays[name][mask]
        result[name]=dict(mean=float(np.mean(values)),std=float(np.std(values)),p10=float(np.quantile(values,.1)),p90=float(np.quantile(values,.9)))
    result["lane_fraction"]={str(lane):float((arrays["lane"][mask]==lane).mean()) for lane in (-1,0,1)}
    return result


def main():
    from algos.sb3_torch.evaluation import source_evaluation_augmentation
    configure_torch()
    candidates=np.array([[-.5,0],[0,0],[.5,0],[0,-1],[0,1]],dtype=np.float32)
    for scene in SCENES:
        data,_,_,digest=load_control(scene)
        outputs={}
        for name in ENCODERS:
            model=load_model(scene,name)
            before=tensor_digest(model)
            pieces={}
            with source_evaluation_augmentation(model),torch.no_grad():
                for start in range(0,len(data["action"]),128):
                    obs=observation_tensors(model,{k:data[k][start:start+128] for k in model.observation_space.spaces})
                    action=model.actor(obs,deterministic=True)
                    size=len(action)
                    mask=torch.tensor(data["lane_action_mask"][start:start+128]>0.5)
                    if hasattr(model.actor,"distribution_parameters"):
                        probabilities=model.actor.distribution_parameters(obs)[0]
                    else:
                        mean,logstd,_=model.actor.get_action_dist_params(obs)
                        dist=torch.distributions.Normal(mean[:,1],logstd[:,1].exp())
                        threshold=float(np.arctanh(1/3))
                        low=dist.cdf(torch.full((size,),-threshold))
                        high=dist.cdf(torch.full((size,),threshold))
                        probabilities=torch.stack([low,high-low,1-high],dim=1)
                    lane=torch.where(action[:,1]<-1/3,-1,torch.where(action[:,1]>1/3,1,0))
                    feasible=mask.gather(1,(lane+1)[:,None]).squeeze(1)
                    feature=model.critic.extract_features(obs,model.critic.features_extractor)
                    cached_q=q_from_features(model.critic,feature,action)
                    if start==0:
                        for expected,actual in zip(model.critic(obs,action),cached_q):
                            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
                    q_actor=torch.cat(cached_q,dim=1).min(1).values
                    grid=[]
                    for candidate in candidates:
                        a=torch.tensor(candidate).reshape(1,2).expand(size,2)
                        grid.append(torch.cat(q_from_features(model.critic,feature,a),dim=1).min(1).values)
                    values=torch.stack(grid,dim=1)
                    grid_mask=torch.stack([mask[:,1],mask[:,1],mask[:,1],mask[:,0],mask[:,2]],dim=1)
                    top=values.masked_fill(~grid_mask,-float("inf")).max(1).values
                    bottom=values.masked_fill(~grid_mask,float("inf")).min(1).values
                    batch=dict(normalized_speed=action[:,0],lane=lane,lane_confidence=probabilities.max(1).values,
                        lane_entropy=-(probabilities*probabilities.clamp_min(1e-12).log()).sum(1),
                        own_grid_best_minus_actor_q=top-q_actor,own_grid_q_spread=top-bottom,infeasible_lane=(~feasible).float(),
                        lane_probabilities=probabilities,grid_q=values,actor_q=q_actor)
                    for key,value in batch.items():
                        pieces.setdefault(key,[]).append(value.numpy())
            assert tensor_digest(model)==before
            outputs[name]={k:np.concatenate(v) for k,v in pieces.items()}
            print(f"common-state actor/critic {scene}/{name}",flush=True)
        strata=dict(all=np.ones(len(data["action"]),dtype=bool),
            non_keep_feasible=data["lane_action_mask"][:,[0,2]].any(1),
            route_change_intent=data["route_intent_valid"] & (data["route_intent"]!=0),
            near_vehicle=data["observed_distance"]<=10)
        results={name:{stratum:summarize_arrays(arrays,mask) for stratum,mask in strata.items()} for name,arrays in outputs.items()}
        paired={}
        for name in ENCODERS[1:]:
            paired[name]=dict(lane_disagreement_to_mst=float(np.mean(outputs[name]["lane"]!=outputs["mst_slt"]["lane"])),
                mean_abs_speed_difference_to_mst_mps=float(np.mean(np.abs(outputs[name]["normalized_speed"]-outputs["mst_slt"]["normalized_speed"]))*5))
        folder=OUTPUT / "stage2/actions" / scene
        folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder / "arrays.npz",**{name+"__"+key:v for name,arrays in outputs.items() for key,v in arrays.items()})
        write_json(folder / "result.json",dict(**metadata(),dataset_digest=digest,results=results,paired=paired,tensors_unchanged=True,
            candidates=candidates.tolist(),limitations=["Q gaps compare each critic to its own finite action grid, never to a no-entropy return",
                "MST lane probabilities integrate its continuous lateral Gaussian over the environment thresholds; it has no categorical lane logits",
                "Common-state actions describe policy use of representations, not closed-loop causal component effects",
                "Critic evaluations are diagnostic only; deployed actions always come directly from the actor"]))


if __name__=="__main__":
    main()

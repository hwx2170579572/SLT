"""Frozen-checkpoint gradient compatibility; never take an optimizer step."""
from __future__ import annotations

from .common import cpu_environment
cpu_environment()
import numpy as np
import torch

from .common import OUTPUT, SCENES, metadata, configure_torch, write_json
from .models import load_model, observation_tensors, tensor_digest
from .collect import ENCODERS
from .control_data import load_control


def nstep_targets(data, indices, horizon, gamma=.99, corrected=True):
    episode=data["episode_id"]
    ends=np.flatnonzero(np.r_[episode[1:]!=episode[:-1],True])+1
    boundary=ends[np.searchsorted(ends,indices,side="right")]
    actual=np.minimum(horizon,boundary-indices)
    last=indices+actual-1
    reward=np.array([sum(gamma**offset*float(data["reward"][i+offset]) for offset in range(int(h)))
                     for i,h in zip(indices,actual)],dtype=np.float32)
    raw_seconds=np.array([data["raw_executed"][i:i+h].sum()*.1 for i,h in zip(indices,actual)])
    return dict(last=last,actual_h=actual,reward=reward,
        done=data["terminated"][last].astype(np.float32),
        discount=(gamma**actual if corrected else np.full(len(indices),gamma)).astype(np.float32),seconds=raw_seconds)


def vector_gradient(loss,parameters,retain_graph=False):
    gradients=torch.autograd.grad(loss,parameters,allow_unused=True,retain_graph=retain_graph)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gradients,parameters)])


def auxiliary_gradient(model,obs,next_obs,actions,parameters):
    from algos.sb3_torch.features import _nonzero_mask
    from algos.sb3_torch.topo_temporal_features_v2 import soft_slot_balance_loss
    target=model.critic if model.representation_online_target_encoder else model.critic_target
    mask=_nonzero_mask(next_obs["trajectory"][:,0,0])
    encoder=model.critic.features_extractor
    if model.structured_representation:
        online=encoder.forward_tokens(obs)
        with torch.no_grad():
            future=target.features_extractor.forward_tokens(next_obs)
        graph=model.representation(online,actions,future,sample_mask=mask).total
        balance,_=soft_slot_balance_loss(online,epsilon=model.slot_balance_epsilon)
        loss=model.representation_coef*(graph+model.slot_balance_coef*balance)
        graph_grad=vector_gradient(model.representation_coef*graph,parameters,retain_graph=True)
        full_grad=vector_gradient(loss,parameters)
        details=dict(graph_slt_loss=float(graph.detach()),slot_balance_loss=float(balance.detach()),
            graph_gradient_norm=float(graph_grad.norm()),weighted_balance_gradient_norm=float((full_grad-graph_grad).norm()))
    else:
        online=encoder(obs)
        with torch.no_grad():
            future=target.features_extractor(next_obs)
        loss=model.representation_coef*model.representation(online,actions,future,sample_mask=mask)
        full_grad=vector_gradient(loss,parameters)
        details=dict(slt_loss=float(loss.detach()))
    return full_grad,dict(**details,valid_auxiliary_fraction=float(mask.float().mean()))


def soft_bootstrap_value(model,next_obs):
    with torch.no_grad():
        entropy=torch.exp(model.log_ent_coef.detach()) if model.log_ent_coef is not None else model.ent_coef_tensor
        if hasattr(model.actor,"all_action_samples"):
            batch=model.actor.all_action_samples(next_obs,deterministic_speed=False)
            feature=model.critic_target.features_extractor(next_obs)
            heads=model.critic_target.all_q_from_features(feature,batch.actions)
            minimum=torch.stack(heads,dim=-1).min(dim=-1).values
            # v4.8 inherits the v4.2 factorized entropy objective. Its default
            # lane_entropy_scale is zero; using joint log-probabilities here
            # would silently diagnose a different TD objective.
            logprob=model.entropy_log_probabilities(batch) if hasattr(model,"entropy_log_probabilities") else batch.joint_log_probabilities
            values=minimum-entropy*logprob.unsqueeze(-1)
            return (batch.lane_probabilities.unsqueeze(-1)*values).sum(1)
        action,logprob=model.actor.action_log_prob(next_obs)
        q=torch.cat(model.critic_target(next_obs,action),dim=1).min(1,keepdim=True).values
        return q-entropy*logprob.reshape(-1,1)


def main():
    from algos.sb3_torch.evaluation import source_evaluation_augmentation
    configure_torch()
    for scene in SCENES:
        data,_,_,digest=load_control(scene)
        indices=np.random.default_rng(731).choice(len(data["action"]),size=min(512,len(data["action"])),replace=False)
        conditions=((4,False,"n4_legacy_gamma"),(4,True,"n4_correct_gamma_h"),(16,True,"n16_correct_gamma_h"))
        results={}
        for name in ENCODERS:
            model=load_model(scene,name)
            before=tensor_digest(model)
            parameters=[p for p in model.critic.features_extractor.parameters() if p.requires_grad]
            results[name]={}
            with source_evaluation_augmentation(model):
                for horizon,corrected,label in conditions:
                    rows=[]
                    for batch_index,start in enumerate(range(0,len(indices),64)):
                        selection=indices[start:start+64]
                        targets=nstep_targets(data,selection,horizon,model.gamma,corrected)
                        obs=observation_tensors(model,{k:data[k][selection] for k in model.observation_space.spaces})
                        one=observation_tensors(model,{k:data["next_"+k][selection] for k in model.observation_space.spaces})
                        future=observation_tensors(model,{k:data["next_"+k][targets["last"]] for k in model.observation_space.spaces})
                        action=torch.tensor(data["action"][selection],dtype=torch.float32)
                        representation,details=auxiliary_gradient(model,obs,one,action,parameters)
                        # Fixed diagnostic RNG; changing n changes only the
                        # observation endpoint and return formula for this model.
                        torch.manual_seed(731+batch_index)
                        value=soft_bootstrap_value(model,future)
                        reward=torch.tensor(targets["reward"]).reshape(-1,1)
                        bootstrap=(1-torch.tensor(targets["done"]).reshape(-1,1))*torch.tensor(targets["discount"]).reshape(-1,1)*value
                        td_target=reward+bootstrap
                        q=model.critic(obs,action)
                        loss=.5*sum(torch.nn.functional.mse_loss(head,td_target) for head in q)
                        td=vector_gradient(loss,parameters)
                        norm_rep,norm_td=float(representation.norm()),float(td.norm())
                        assert torch.isfinite(representation).all() and torch.isfinite(td).all()
                        denominator=norm_rep*norm_td
                        cosine=float(torch.dot(representation,td)/denominator) if denominator>1e-20 else None
                        rows.append(dict(batch=batch_index,norm_representation=norm_rep,norm_td=norm_td,cosine=cosine,
                            representation_to_td_norm=norm_rep/max(norm_td,1e-20),td_mse=float(loss.detach()),
                            target_variance=float(td_target.var(unbiased=False)),reward_nonzero_fraction=float((reward!=0).float().mean()),
                            bootstrap_abs_mean=float(bootstrap.abs().mean()),reward_abs_mean=float(reward.abs().mean()),
                            actual_h_mean=float(targets["actual_h"].mean()),actual_seconds_mean=float(targets["seconds"].mean()),
                            terminal_fraction=float(targets["done"].mean()),**details))
                    cosines=[row["cosine"] for row in rows if row["cosine"] is not None]
                    results[name][label]=dict(batches=rows,mean_cosine=float(np.mean(cosines)) if cosines else None,
                        negative_cosine_batches=sum(c<0 for c in cosines),batch_count=len(rows),
                        mean_norm_ratio=float(np.mean([row["representation_to_td_norm"] for row in rows])))
            assert before==tensor_digest(model)
            print(f"frozen gradients {scene}/{name}",flush=True)
        coverage={}
        for horizon in (1,4,16,32):
            target=nstep_targets(data,np.arange(len(data["action"])),horizon)
            coverage[str(horizon)]=dict(nonzero_return_fraction=float((target["reward"]!=0).mean()),
                mean_actual_h=float(target["actual_h"].mean()),mean_actual_seconds=float(target["seconds"].mean()),
                terminal_fraction=float(target["done"].mean()),discount_mean=float(target["discount"].mean()))
        write_json(OUTPUT / "stage2/gradients" / f"{scene}.json",dict(**metadata(),dataset_digest=digest,
            results=results,horizon_coverage=coverage,tensors_unchanged=True,optimizer_steps=0,
            limitations=["Gradients on a fixed shared behavior mixture, not the unavailable historical training replay",
                "No source rotation augmentation; unclipped gradients, same frozen weights, no sequential auxiliary update",
                "SAC target form reproduced including final entropy term; intermediate entropy and importance correction remain absent",
                "v4.8 uses its native factorized entropy helper (lane scale zero); MST uses its native continuous entropy",
                "Each actual collected decision appears once; diagnostic mixture does not duplicate terminal replay samples",
                "n4/n16 changes here are diagnostic target perturbations, not retrained-policy ablations",
                "Final gradient disagreement is a mechanism clue, not evidence of interference throughout training"]))


if __name__=="__main__":
    main()

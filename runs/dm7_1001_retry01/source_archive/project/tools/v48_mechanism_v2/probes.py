"""Grouped supervised probes. Fitting these heads never updates an RL encoder."""
from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss


def grouped_split(groups, seed=73):
    unique = np.unique(groups)
    np.random.default_rng(seed).shuffle(unique)
    fit_count, validation_count = int(.55*len(unique)), max(1,int(.15*len(unique)))
    assert fit_count > 1 and len(unique) > fit_count+validation_count
    chosen = (unique[:fit_count], unique[fit_count:fit_count+validation_count], unique[fit_count+validation_count:])
    masks = tuple(np.isin(groups, ids) for ids in chosen)
    assert np.all(sum(mask.astype(int) for mask in masks)==1)
    return masks, dict(seed=seed, grouping="traffic seed; paired policies/branches remain together",
        fit_groups=sorted(chosen[0].tolist()), validation_groups=sorted(chosen[1].tolist()), test_groups=sorted(chosen[2].tolist()),
        samples=[int(mask.sum()) for mask in masks])


def standardize(x, fit):
    mean, std = x[fit].mean(0), x[fit].std(0)
    std[std<1e-6] = 1.
    return ((x-mean)/std).astype(np.float32)


def regression_metrics(y, p):
    residual = p-y
    denominator = np.square(y-y.mean(0)).sum(0)
    scores = [float(1-error/denom) if denom>1e-12 else None
              for error,denom in zip(np.square(residual).sum(0),denominator)]
    defined=[score for score in scores if score is not None]
    return dict(rmse=float(np.sqrt(np.mean(residual**2))), r2_mean=float(np.mean(defined)) if defined else None,
                r2_per_dimension=scores, mae=float(np.abs(residual).mean()))


def binary_metrics(y, p):
    y,p = y.reshape(-1),p.reshape(-1)
    both = len(np.unique(y))==2
    return dict(samples=len(y), positives=int(y.sum()), prevalence=float(y.mean()),
        roc_auc=float(roc_auc_score(y,p)) if both else None,
        average_precision=float(average_precision_score(y,p)) if both else None,
        brier=float(np.mean((p-y)**2)), identifiable_discrimination=both)


def fit_probe(x, y, masks, *, kind="regression", head="linear", seed=73):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim==1:
        y=y[:,None]
    valid = np.isfinite(y).all(1) & np.isfinite(x).all(1)
    fit,val,test = tuple(mask & valid for mask in masks)
    if min(fit.sum(),val.sum(),test.sum()) < 2:
        return None, dict(identifiable=False, reason="insufficient complete samples", samples=[int(m.sum()) for m in (fit,val,test)])
    if kind=="binary" and len(np.unique(y[fit])) < 2:
        return None, dict(identifiable=False, reason="single event class in fit groups", fit_positives=int(y[fit].sum()))
    xs = standardize(x, fit)
    if kind=="regression":
        ym,ys = y[fit].mean(0), y[fit].std(0)
        ys[ys<1e-6] = 1.
        target = (y-ym)/ys
    else:
        ym,ys = np.zeros(y.shape[1]),np.ones(y.shape[1])
        target = y
    if head=="linear" and kind=="regression":
        design = np.concatenate([xs.astype(np.float64),np.ones((len(xs),1))],axis=1)
        gram,rhs = design[fit].T@design[fit],design[fit].T@target[fit]
        candidates=[]
        for alpha in (.001,.1,1.,10.):
            reg = np.eye(design.shape[1])*alpha
            reg[-1,-1]=0
            weights = np.linalg.solve(gram+reg,rhs)
            candidates.append((float(np.mean((design[val]@weights-target[val])**2)),alpha,weights))
        val_loss,alpha,weights = min(candidates,key=lambda row:row[0])
        prediction = (design@weights)*ys+ym
        settings = dict(ridge=alpha, validation_standardized_mse=val_loss)
    elif head=="linear" and kind=="binary":
        assert y.shape[1]==1
        candidates=[]
        for c in (.1,1.,10.,100.):
            classifier=LogisticRegression(C=c,max_iter=1000,random_state=seed)
            classifier.fit(xs[fit],y[fit,0])
            candidates.append((log_loss(y[val,0],classifier.predict_proba(xs[val])[:,1],labels=[0,1]),c,classifier))
        val_loss,c,classifier=min(candidates,key=lambda row:row[0])
        prediction=classifier.predict_proba(xs)[:,1:2]
        settings=dict(C=c,validation_log_loss=float(val_loss))
    elif head=="mlp":
        torch.manual_seed(seed)
        net=torch.nn.Sequential(torch.nn.Linear(x.shape[1],128),torch.nn.ReLU(),
            torch.nn.Linear(128,128),torch.nn.ReLU(),torch.nn.Linear(128,y.shape[1]))
        optimizer=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=.0001)
        tx,ty=torch.from_numpy(xs),torch.tensor(target,dtype=torch.float32)
        indices=np.flatnonzero(fit)
        random=np.random.default_rng(seed)
        criterion=torch.nn.functional.mse_loss if kind=="regression" else torch.nn.functional.binary_cross_entropy_with_logits
        best_loss,best_epoch,best_state,wait=float("inf"),0,None,0
        for epoch in range(1,101):
            net.train()
            order=random.permutation(indices)
            for start in range(0,len(order),256):
                batch=order[start:start+256]
                optimizer.zero_grad(set_to_none=True)
                loss=criterion(net(tx[batch]),ty[batch])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite probe loss")
                loss.backward()
                optimizer.step()
            net.eval()
            with torch.no_grad():
                score=float(criterion(net(tx[val]),ty[val]))
            if score < best_loss-1e-6:
                best_loss,best_epoch,best_state,wait=score,epoch,copy.deepcopy(net.state_dict()),0
            else:
                wait+=1
            if wait>=12:
                break
        net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            out=net(tx)
            prediction=(out.numpy()*ys+ym) if kind=="regression" else out.sigmoid().numpy()
        settings=dict(best_epoch=best_epoch,epochs_run=epoch,validation_loss=best_loss,
            hidden=[128,128],lr=.001,weight_decay=.0001,seed=seed,parameter_count=sum(p.numel() for p in net.parameters()))
    else:
        raise ValueError((kind,head))
    assert np.isfinite(prediction[valid]).all()
    scores = regression_metrics(y[test],prediction[test]) if kind=="regression" else binary_metrics(y[test],prediction[test])
    baseline = regression_metrics(y[test],np.broadcast_to(y[fit].mean(0),y[test].shape)) if kind=="regression" else binary_metrics(y[test],np.full_like(y[test],y[fit].mean()))
    return prediction, dict(identifiable=True,kind=kind,head=head,samples=[int(m.sum()) for m in (fit,val,test)],
        fit_only_target_scale=ys.tolist(),test=scores,constant_fit_baseline=baseline,settings=settings,
        valid_test_indices=np.flatnonzero(test).tolist())


def paired_error_interval(y, first, second, groups, test, scale, *, repetitions=1000):
    """Equal-group mean normalized squared-error difference; negative favors second."""
    y=np.asarray(y).reshape(len(y),-1)
    usable=test & np.isfinite(y).all(1) & np.isfinite(first).all(1) & np.isfinite(second).all(1)
    values=np.mean(((second-y)/scale)**2-((first-y)/scale)**2,axis=1)
    ids=np.unique(groups[usable])
    errors=np.asarray([values[usable & (groups==group)].mean() for group in ids])
    if len(ids)<2:
        return dict(identifiable=False,groups=len(ids))
    rng=np.random.default_rng(731)
    boot=errors[rng.integers(0,len(ids),(repetitions,len(ids)))].mean(1)
    return dict(identifiable=True,groups=len(ids),second_minus_first=float(errors.mean()),
        ci95=np.quantile(boot,[.025,.975]).tolist(),repetitions=repetitions,
        metric="equal-traffic-seed normalized MSE difference; negative favors second",
        uncertainty="conditional on these fixed training-seed-0 models and fitted probes")

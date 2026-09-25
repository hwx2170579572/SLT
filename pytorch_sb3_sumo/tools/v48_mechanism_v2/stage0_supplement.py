"""Read restored optimizer state without modifying any existing training source."""
from .common import cpu_environment
cpu_environment()
from .common import OUTPUT,SCENES,configure_torch,metadata,write_json
from .models import load_model,tensor_digest
from .collect import ENCODERS


def main():
    configure_torch()
    models=[]
    for scene in SCENES:
        for name in ENCODERS:
            model=load_model(scene,name)
            models.append(dict(scene=scene,name=name,tensor_sha256=tensor_digest(model),
                restored_optimizer_rates={key:[float(group["lr"]) for group in optimizer.param_groups]
                    for key,optimizer in {"actor":model.actor.optimizer,"critic":model.critic.optimizer,
                        "representation":model.representation_optimizer,"entropy":model.ent_coef_optimizer}.items()
                    if optimizer is not None},
                separate_target_projector=bool(getattr(model.representation,"separate_target_projector",False)),
                auxiliary_target_encoder="online_critic_stop_gradient" if model.representation_online_target_encoder else "ema_critic_target_stop_gradient",
                replay_class=type(model.replay_buffer).__name__,
                actual_training_implementation=type(model).train.__module__,
                lane_entropy_scale=getattr(model,"lane_entropy_scale",None),
                replay_n_steps=model.replay_buffer.n_steps,raw_training_steps=int(model._raw_steps_seen),
                updates=int(model._n_updates),entropy_coefficient=float(model.log_ent_coef.detach().exp()) if model.log_ent_coef is not None else float(model.ent_coef_tensor)))
    write_json(OUTPUT / "stage0/restored_training_state.json",dict(**metadata(),models=models,
        interpretation="Restored optimizer rates, which may differ from saved initial LR under late decay. No optimizer steps executed."))


if __name__=="__main__":
    main()

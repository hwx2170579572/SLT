"""Source-backed figures and a conservative stage-0/stage-2 evidence report."""
from __future__ import annotations

import csv
import os
from pathlib import Path

from .common import cpu_environment, OUTPUT
cpu_environment()
os.environ["MPLCONFIGDIR"]=str(OUTPUT / ".matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import ROOT, SCENES, metadata, load_json, write_json, describe_file
from .collect import ENCODERS,BEHAVIORS

NAMES=dict(mst_slt="MST+SLT",v4_8_lr_half="v4.8/lr_half",late_decay="late_decay",batch64="batch64")
COLORS=dict(mst_slt="#33658A",v4_8_lr_half="#D55E00",late_decay="#009E73",batch64="#AA66AA")


def fmt(value,digits=3):
    return "不可识别" if value is None else f"{value:.{digits}f}"


def save_fig(fig,name):
    fig.savefig(OUTPUT / "figures" / (name+".png"),dpi=180,bbox_inches="tight",facecolor="white")
    fig.savefig(OUTPUT / "figures" / (name+".pdf"),bbox_inches="tight",facecolor="white")
    plt.close(fig)


def local_figure(local):
    targets=("ego_dynamics","minimum_distance","minimum_ttc")
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
    for row,scene in enumerate(SCENES):
        for column,head in enumerate(("linear","mlp")):
            ax=axes[row,column]
            for i,name in enumerate(ENCODERS):
                values=[local[scene]["probes"][f"{head}__{target}__{name}"]["test"]["r2_mean"] for target in targets]
                ax.bar(np.arange(3)+(i-1.5)*.18,values,.17,label=NAMES[name],color=COLORS[name])
            raw=[local[scene]["probes"][f"{head}__{target}__raw_pca128"]["test"]["r2_mean"] for target in targets]
            ax.scatter(np.arange(3),raw,marker="x",s=55,color="#333333",label="Raw PCA-128 reference",zorder=4)
            ax.set_title(f"{scene.upper()} | {head}")
            ax.set_xticks(range(3),["Next ego dynamics","Current distance","Closest-approach time"])
            ax.set_ylabel("Test R-squared (higher is better)")
            ax.axhline(0,color="gray",linewidth=.7)
            ax.grid(axis="y",alpha=.15)
    axes[0,0].legend(fontsize=8,loc="lower left")
    fig.suptitle("Shared historical observations | grouped 22 fit / 6 validation / 12 test episodes\nFrozen seed-0 policies; closest-approach proxy is not collision TTC",fontsize=12)
    fig.tight_layout()
    save_fig(fig,"local_readability")


def future_figure(future):
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for row,scene in enumerate(SCENES):
        for column,(target,metric,title) in enumerate((("progress_m_h16","r2_mean","16-decision progress R-squared"),
                                                      ("collision_h16","roc_auc","16-decision collision AUROC"))):
            ax=axes[row,column]
            for j,head in enumerate(("linear","mlp")):
                values=[]
                for name in ENCODERS:
                    result=future[scene]["probes"][f"{head}__{target}__{name}__action"]
                    value=result.get("test",{}).get(metric)
                    values.append(np.nan if value is None else value)
                ax.bar(np.arange(4)+(j-.5)*.32,values,.30,label=head,color=("#33658A","#D55E00")[j])
            ax.set_xticks(range(4),[NAMES[n] for n in ENCODERS],rotation=12)
            ax.set_title(f"{scene.upper()} | {title}")
            if metric=="roc_auc":
                positive=future[scene]["coverage"][target][2]["event_seed_count"]
                ax.text(.02,.04,f"Positive-event test seeds: {positive}/12",transform=ax.transAxes,fontsize=9)
                ax.axhline(.5,color="gray",linestyle="--",linewidth=.7)
                ax.set_ylim(0,1.05)
            ax.grid(axis="y",alpha=.15)
            ax.legend(fontsize=9)
    fig.suptitle("Action-conditioned future control diagnostics | shared three-policy mixture\nConditional episode uncertainty; not independent RL-training seeds",fontsize=12)
    fig.tight_layout()
    save_fig(fig,"future_alignment")


def gradient_figure(gradients):
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),sharey=True)
    labels=("n4_legacy_gamma","n4_correct_gamma_h","n16_correct_gamma_h")
    names=("4 / legacy discount","4 / correct discount","16 / correct discount")
    for ax,scene in zip(axes,SCENES):
        for j,(label,name) in enumerate(zip(labels,names)):
            for i,model in enumerate(ENCODERS):
                rows=gradients[scene]["results"][model][label]["batches"]
                values=[r["cosine"] for r in rows if r["cosine"] is not None]
                position=i+(j-1)*.23
                ax.scatter(np.full(len(values),position),values,s=12,alpha=.35,color=("#999999","#33658A","#D55E00")[j])
                if values:
                    ax.scatter(position,np.mean(values),s=65,marker="_",color=("#555555","#33658A","#D55E00")[j],label=name if i==0 else None)
        ax.axhline(0,color="black",linewidth=.8)
        ax.set_xticks(range(4),[NAMES[n] for n in ENCODERS],rotation=12)
        ax.set_title(scene.upper())
        ax.set_ylim(-1,1)
        ax.set_ylabel("Cosine: representation vs TD encoder gradients")
        ax.legend(fontsize=8)
    fig.suptitle("Frozen final checkpoints, 8 shared diagnostic batches | no optimizer steps\nTarget perturbations are not retrained-policy ablations",fontsize=12)
    fig.tight_layout()
    save_fig(fig,"frozen_gradient_cosines")


def horizon_figure(gradients):
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    h=np.array([1,4,16,32])
    for scene,color in zip(SCENES,("#33658A","#D55E00")):
        coverage=gradients[scene]["horizon_coverage"]
        axes[0].plot(h,[coverage[str(n)]["nonzero_return_fraction"] for n in h],"o-",label=scene.upper(),color=color)
        axes[1].plot(h,[coverage[str(n)]["mean_actual_seconds"] for n in h],"o-",label=scene.upper(),color=color)
    for ax in axes:
        ax.set_xlabel("Maximum stored-decision horizon")
        ax.set_xticks(h)
        ax.grid(alpha=.2)
        ax.legend()
    axes[0].set_ylabel("Fraction with a nonzero aggregated reward")
    axes[1].set_ylabel("Mean actual observed duration (seconds)")
    fig.suptitle("Empirical target coverage on the same fixed behavior mixture\nTerminal-event coverage is not causal decision-to-outcome delay",fontsize=11)
    fig.tight_layout()
    save_fig(fig,"horizon_coverage")


def main():
    figure=OUTPUT / "figures"
    figure.mkdir(parents=True,exist_ok=True)
    local={s:load_json(OUTPUT / "stage2/local" / s / "result.json") for s in SCENES}
    future={s:load_json(OUTPUT / "stage2/future" / s / "result.json") for s in SCENES}
    gradients={s:load_json(OUTPUT / "stage2/gradients" / f"{s}.json") for s in SCENES}
    branches={s:load_json(OUTPUT / "stage2/branches" / s / "probe_result.json") for s in SCENES}
    actions={s:load_json(OUTPUT / "stage2/actions" / s / "result.json") for s in SCENES}
    audit=load_json(OUTPUT / "stage0/audit.json")
    restored=load_json(OUTPUT / "stage0/restored_training_state.json")
    local_figure(local)
    future_figure(future)
    gradient_figure(gradients)
    horizon_figure(gradients)
    lines=["# v4.8/lr_half：阶段0与阶段2执行结果", "",
        "本报告使用真实历史文件和新增诊断数据。新策略训练数为0；检查点均为预定训练结束时的最终模型，动作直接来自actor。阶段1的配置、源码和运行进程未被修改。", "",
        "## 阶段0：已核验的事实与适用范围", "",
        f"历史模型诊断门槛：{audit['historical_diagnostics_gate']}。现有测试12项、回报数值检查7项，以及8个最终模型的CPU加载/actor/梯度检查通过。新诊断单元测试另见 stage2/pytest.xml。", "",
        "未来受控训练尚须冻结公共U、构建并验证A/B/C桥接接口。历史MST的四步gamma bootstrap与full的16步gamma^h不同，不能把旧minus_horizon当成纯长度消融。正确折扣修复也不能当成表示创新。", "",
        "| 场景/模型 | 编码器参数 | 辅助模块参数 | 回报 n | 保存的初始 LR | 实际恢复的 actor LR | 辅助 target |",
        "|---|---:|---:|---:|---:|---:|---|"]
    for item in audit["models"]:
        state=next(v for v in restored["models"] if v["scene"]==item["scene"] and v["name"]==item["model"])
        lines.append(f"| {item['scene']}/{NAMES[item['model']]} | {item['encoder_parameters']} | {item['auxiliary_parameters']} | {state['replay_n_steps']} | {item['saved_config']['learning_rate']} | {state['restored_optimizer_rates']['actor'][0]} | {state['auxiliary_target_encoder']} |")
    lines += ["", "actor不向共享编码器传梯度；critic和辅助目标都会更新编码器。full还包含0.01的slot平衡项；关闭structured_representation会同时绕开该正则。Graph-SLT不是整个TTG编码器，整方法比较不能独立识别二者贡献。", "",
        "Cross与CARLA都是终端奖励为主；Cross超时保留bootstrap，CARLA超时终止。当前实现不含完整的中间熵累积或off-policy概率比校正。旧minimum_ttc只是恒速最近接近时间代理，并非已验证碰撞TTC。", "",
        "## 阶段2A：复用旧数据与局部可读性", "",
        "旧数据为MST行为的每场景40回合。先用原28/12划分和Ridge 0.001逐项核对；历史数字偏差依用户指示只记录、不阻塞，不宣称逐字精确复现（Cross最大约0.000167；CARLA最大约0.006625，后者来自social slot，full latent最大约0.001185）。模型哈希和数据身份仍严格核验。下表新比较另用共同22 fit / 6 validation / 12 test分组与统一CPU协议。两套分数不能混算。", "",
        "| 场景 | 读出 | 标签 | MST R² | full R² | full−MST 的配对标准化误差差 [95%区间] |",
        "|---|---|---|---:|---:|---|"]
    local_rows=[]
    for scene in SCENES:
        assert local[scene]["legacy_reproduction_disposition"]["accepted"]
        for head in ("linear","mlp"):
            for target in ("ego_dynamics","minimum_distance","minimum_ttc"):
                first=local[scene]["probes"][f"{head}__{target}__mst_slt"]["test"]["r2_mean"]
                second=local[scene]["probes"][f"{head}__{target}__v4_8_lr_half"]["test"]["r2_mean"]
                interval=local[scene]["paired_intervals"][f"{head}__{target}__v4_8_lr_half_vs_mst"]
                lo,hi=interval["ci95"]
                lines.append(f"| {scene} | {head} | {target} | {fmt(first)} | {fmt(second)} | {fmt(interval['second_minus_first'])} [{fmt(lo)}, {fmt(hi)}] |")
                local_rows.append(dict(scene=scene,head=head,target=target,mst_r2=first,full_r2=second,normalized_error_difference=interval["second_minus_first"],ci_low=lo,ci_high=hi))
    lines += ["", "误差差为full减MST，负值有利于full；按测试episode等权统计，和逐帧总体R²是不同聚合指标。区间以交通/episode为单位重采样，条件于固定模型与probe，不估计训练seed方差。多项探索性比较未做确认性多重检验校正。", "",
        "![局部表示读出](figures/local_readability.png)", "", "## 阶段2B：控制相关性与新数据覆盖", "",
        "新数据固定三种行为（MST/full/late_decay），每场景各40回合，共240回合。40个交通seed跨行为配对，按seed整体分入22/6/12组。late_decay是预先规定的首个稳定性候选，不是本报告选择的赢家。训练模型均为seed0。", "",
        "未来1/4/16/32决策做线性读出；16决策另做MLP和不输入动作的对照。所有probe共同获得behavior ID，动作条件使用速度+车道one-hot。目标是给定继续策略的实际未来，尚非反事实最优动作价值。", "",
        "| 场景 | 模型行为 | 成功 | 碰撞 | 超时 | 回合数 |", "|---|---|---:|---:|---:|---:|"]
    for scene in SCENES:
        for name in BEHAVIORS:
            result=future[scene]["behavior_summary"][name]
            lines.append(f"| {scene} | {NAMES[name]} | {fmt(result['success_rate'])} | {fmt(result['collision_rate'])} | {fmt(result['timeout_rate'])} | {result['episodes']} |")
    lines += ["", "这批新增诊断环境与种子不能混入历史100回合得分；详细驾驶数据含速度、停止、加速度、jerk、急制动、跟车TTC适用覆盖、变道与不可行动作比例。跟车TTC不能覆盖所有侧向/交叉冲突。缺失终端位置不用于未来距离回归。", "",
        "| 场景 | 16步目标 | 读出 | MST | full | 测试正事件seed数 |", "|---|---|---|---:|---:|---:|"]
    for scene in SCENES:
        for target,metric in (("progress_m_h16","r2_mean"),("collision_h16","roc_auc"),("timeout_h16","roc_auc")):
            for head in ("linear","mlp"):
                score=[]
                for name in ("mst_slt","v4_8_lr_half"):
                    score.append(future[scene]["probes"][f"{head}__{target}__{name}__action"].get("test",{}).get(metric))
                n="—" if target.startswith("progress") else future[scene]["coverage"][target][2]["event_seed_count"]
                lines.append(f"| {scene} | {target}/{metric} | {head} | {fmt(score[0])} | {fmt(score[1])} | {n} |")
    lines += ["", "AUROC不可识别表示测试缺少正负类别或训练缺少事件，不能写成风险不存在。AP、Brier、常数基线、分层分数和完整类别覆盖均保存在future结果文件。罕见事件的很多连续正帧不能替代多个独立事件seed。", "",
        "![控制相关读出](figures/future_alignment.png)", "", "## 阶段2C：回报覆盖与冻结梯度", "",
        "同一观测混合数据分别构造legacy n4、正确n4和正确n16目标。保持每个最终模型权重不变，计算共享编码器上的TD/表示梯度；没有执行optimizer.step，也不是重新训练后的消融结果。", "",
        "| 场景 | 模型 | legacy4夹角余弦 | 正确4夹角余弦 | 正确16夹角余弦 |", "|---|---|---:|---:|---:|"]
    for scene in SCENES:
        for name in ENCODERS:
            rows=gradients[scene]["results"][name]
            values=[rows[key]["mean_cosine"] for key in ("n4_legacy_gamma","n4_correct_gamma_h","n16_correct_gamma_h")]
            lines.append(f"| {scene} | {NAMES[name]} | {' | '.join(fmt(value) for value in values)} |")
    lines += ["", "负夹角只是最终权重附近的局部梯度冲突线索；本诊断未读取历史训练replay，未重现整条优化轨迹，不能声称已证明训练震荡的因果来源。有效秩、CKA和最终在线/target差距同样是描述量，不是质量评分或完整target drift时间序列。", "",
        "![梯度诊断](figures/frozen_gradient_cosines.png)", "", "![回报覆盖](figures/horizon_coverage.png)", "",
        "## 阶段2D：动作分支与闭环使用", "",
        "分支从首12个预定交通seed的MST轨迹出发，逐步核对重放的历史、奖励和交通文件，所有候选动作均接full最终actor作为公共继续策略。报告的是该继续策略下的32决策后果，不是最优Q。动作有相同回报时不硬造排序标签。", "",
        "| 场景 | 标签 | 读出 | 模型 | 可排序测试动作对 | 排序准确率 |", "|---|---|---|---|---:|---:|"]
    for scene in SCENES:
        for key,result in branches[scene]["results"].items():
            head,target,name=key.split("__")
            rank=result.get("action_ranking",{})
            lines.append(f"| {scene} | {target} | {head} | {NAMES[name]} | {rank.get('eligible_pairs',0)} | {fmt(rank.get('pairwise_accuracy'))} |")
    lines += ["", "分支规模小且回报可能大量持平，不能单靠排序分数下强结论。进展排序排除碰撞分支及缺失位置；其条件性必须与安全结果一起解释。actions结果另给同状态actor车道分布、速度及各自critic内部Q-gap；部署仍只运行actor。", "",
        "## 当前能与不能支持的判断", "",
        "1. 局部线性读出、非线性读出、未来控制和闭环表现已分别测量。它们解决不同问题：线性更好只说明更易线性读出；未来事件不足则结论是未判明。", "",
        "2. 本阶段可以找出与目标不对齐、梯度干扰或回报敏感性相符的证据，但不能把整方法差异归因为TTG或Graph-SLT。训练种子、优化协议、动作头、回报和容量差异仍需受控实验。", "",
        "3. 下一步等待阶段1按原协议冻结公共U，然后阶段3开展A16/B16/C16及正确C4；阶段4在固定n下干预预测目标/时距；阶段5做正确折扣下A4/A16/C4/C16。只有这些训练干预才能区分设计问题与当前实验读出/覆盖不足。", "",
        "4. 没有通过新数据选择历史检查点，没有增加部署选择器，也没有启动阶段3训练。本阶段不对CCF-A或期刊录用作保证。", "",
        "## 可复核文件", "",
        "- `stage0/audit.json`：模型、数据、代码指纹与发现；`stage0/restored_training_state.json`：恢复后的真实optimizer LR。",
        "- `stage2/control`：预定协议、240个回合逐步数组、真实遥测与配对收据。",
        "- `stage2/local` / `stage2/future`：完整读出结果、分组及预测；`stage2/features`：哈希绑定的冻结表示。",
        "- `stage2/gradients` / `stage2/actions` / `stage2/branches`：目标、梯度、行为与验证后的动作分支。",
        "- `figures`：由本地真实数值生成的PNG和PDF；`evidence_manifest.json`：结果输入指纹。", ""]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    with (OUTPUT / "local_comparison.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(local_rows[0]))
        writer.writeheader();writer.writerows(local_rows)
    inputs=[OUTPUT / "stage0/audit.json",OUTPUT / "stage0/restored_training_state.json"]
    inputs += [OUTPUT / f"stage2/{axis}/{scene}/result.json" for axis in ("local","future","actions") for scene in SCENES]
    inputs += [OUTPUT / f"stage2/gradients/{scene}.json" for scene in SCENES]
    inputs += [OUTPUT / f"stage2/branches/{scene}/probe_result.json" for scene in SCENES]
    write_json(OUTPUT / "evidence_manifest.json",dict(**metadata(),inputs=[describe_file(p) for p in inputs],
        code=[describe_file(p) for p in sorted((ROOT / "tools/v48_mechanism_v2").glob("*.py"))],
        figures=[describe_file(p) for p in sorted(figure.glob("*"))]))
    print("Source-backed report and figures complete",flush=True)


if __name__=="__main__":
    main()

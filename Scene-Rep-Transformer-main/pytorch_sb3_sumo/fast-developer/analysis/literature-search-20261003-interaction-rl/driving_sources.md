# Driving interaction, safety, world-model, and closed-loop RL papers

Search date: 2026-10-03. Publication window: 2024-10-03 through 2026-10-03, using formal proceedings/first-online publication date rather than arXiv posting date. Venue names below are taken from proceedings or publisher pages; no CCF or journal-quartile labels are assigned.

## Selected six

### CaRL: Learning Scalable Planning Policies with Simple Rewards

CoRL 2025, PMLR 305, pp. 5301–5338. The paper studies privileged planning with PPO and reports a concrete optimization interaction: PPO struggles with a popular multi-term shaped reward as minibatches grow, while a route-completion-centered reward with infractions handled by termination or multiplicative penalties scales better. The authors report 300M CARLA samples and 500M nuPlan samples on one 8-GPU node; the policy reaches 64 driving score on CARLA Longest6 v2 and 91.3 / 90.6 on nuPlan Val14 non-reactive / reactive evaluation. These are paper-reported results, not numbers from this project. The actionable idea is to treat reward form and optimizer batch regime as a coupled hypothesis. It does not establish that PPO findings transfer unchanged to SAC or to a structured-state intersection task. Official code is available at [autonomousvision/CaRL](https://github.com/autonomousvision/CaRL); its Civil-M license includes additional use conditions.

Evidence: [PMLR paper page and abstract](https://proceedings.mlr.press/v305/jaeger25a.html); [official code and license](https://github.com/autonomousvision/CaRL).

Screening scores (insight / fit to interaction decisions / evidence, 1–5): **5 / 5 / 5**.

### Raw2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving (in CARLA v2)

NeurIPS 2025 Main Conference. Raw2Drive uses a privileged world-model/planner stream to guide alignment of a raw-sensor world model, then uses privileged-model reward and continuation signals to guide raw-policy training. The official experiments use CARLA 0.9.15.1, train over 1,000 routes, and evaluate without privileged observations. On Bench2Drive, the paper reports 71.36 driving score and 50.24% success rate; this is its raw-sensor RL row, while the same table’s privileged Think2Drive expert is 91.85 / 85.41, so the two rows are not an input-matched comparison. The ablation shows strong sensitivity to alignment: the reported Dev10 score changes from 0.0 with no spatial/temporal alignment to 83.5 with both. The transferable point is to measure representation/rollout alignment before attributing failures to the policy. The method is image-based CARLA E2E, so it does not establish benefits for a small structured-state policy. The official repository currently describes itself as inference code, not a complete training release: [Thinklab-SJTU/Raw2Drive](https://github.com/Thinklab-SJTU/Raw2Drive).

Evidence: [official NeurIPS paper and Tables 3–8](https://papers.neurips.cc/paper_files/paper/2025/hash/c2915bc5961edb04e209a524ec167522-Paper-Conference.pdf); [official NeurIPS record](https://papers.neurips.cc/paper_files/paper/2025/hash/c2915bc5961edb04e209a524ec167522-Abstract-Conference.html); [repository README](https://github.com/Thinklab-SJTU/Raw2Drive).

Screening scores: **4 / 4 / 4**.

### SafeDrive: Fine-Grained Safety Reasoning for End-to-End Driving in a Sparse World

CVPR 2026, pp. 24854–24864. SafeDrive conditions a sparse future world on candidate trajectories, models critical agents and road entities, and predicts pair-wise collision risk per agent and temporal drivable-area compliance. Its reported NAVSIM result is PDMS 91.6 / EPDMS 87.5 with 61 collisions among 12,146 scenarios (0.5%); its Bench2Drive driving score is 66.8. This is useful as a design reference for actor- and time-specific failure evidence, not as proof that an online RL safety module improves success. The paper’s training is planning/safety-head learning with PDM-based supervision rather than a matched policy-gradient RL experiment. Official code and checkpoints were released in August 2026: [SPA-junghokim/SafeDrive](https://github.com/SPA-junghokim/SafeDrive).

Evidence: [CVF official paper page and abstract](https://openaccess.thecvf.com/content/CVPR2026/html/Kim_SafeDrive_Fine-Grained_Safety_Reasoning_for_End-to-End_Driving_in_a_Sparse_CVPR_2026_paper.html); [paper PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Kim_SafeDrive_Fine-Grained_Safety_Reasoning_for_End-to-End_Driving_in_a_Sparse_CVPR_2026_paper.pdf); [author code/release notes and reported metrics](https://github.com/SPA-junghokim/SafeDrive).

Screening scores: **4 / 4 / 5**.

### Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-To-End Autonomous Driving

NeurIPS 2024 Datasets and Benchmarks Track. Its paper describes 2 million annotated frames from 13,638 clips across 44 interactive scenarios, 23 weather conditions, and 12 CARLA towns; evaluation comprises 220 short routes, each centered on one scenario. This scenario-disentangled design directly supports separating ability-level outcomes from a single long-route score. The scope is camera/sensor E2E CARLA evaluation; its traffic and scoring assumptions are not interchangeable with a state-based SUMO intersection environment. The paper states that data, code, and checkpoints are available from the [official repository](https://github.com/Thinklab-SJTU/Bench2Drive) and [Hugging Face dataset](https://huggingface.co/datasets/rethinklab/Bench2Drive).

Evidence: [official NeurIPS paper page](https://proceedings.neurips.cc/paper_files/paper/2024/hash/017761f94a1cd66d01c041aff85492c4-Abstract-Datasets_and_Benchmarks_Track.html); [paper PDF, abstract and evaluation protocol](https://proceedings.neurips.cc/paper_files/paper/2024/file/017761f94a1cd66d01c041aff85492c4-Paper-Datasets_and_Benchmarks_Track.pdf); [official repository](https://github.com/Thinklab-SJTU/Bench2Drive).

Benchmark screening scores (method insight / protocol fit / benchmark evidence): **N/A / 5 / 5**.

### NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking

NeurIPS 2024 Datasets and Benchmarks Track. NAVSIM rolls a fixed plan for a four-second horizon from one queried scene frame and uses a non-reactive environment; the policy receives no feedback and does not affect other agents. It computes collision and drivable-area penalties along with progress, TTC, and comfort subscores. On navtest, the paper reports 84.0 PDMS for TransFuser and 94.8 for the human reference. This is a practical source for balanced safety/progress/comfort metrics and a cheaper alignment screen. Its decoupled actors cannot validate how a learned ego policy changes other drivers’ responses, so it cannot replace interactive closed-loop tests. Official code: [autonomousvision/navsim](https://github.com/autonomousvision/navsim).

Evidence: [official NeurIPS paper PDF, simulation assumptions and Table 1](https://proceedings.neurips.cc/paper_files/paper/2024/file/32768f7faf1995026ef9821c696f3404-Paper-Datasets_and_Benchmarks_Track.pdf); [official repository](https://github.com/autonomousvision/navsim).

Benchmark screening scores: **N/A / 4 / 4**.

### AdaWM: Adaptive World Model based Planning for Autonomous Driving

ICLR 2025. AdaWM diagnoses whether finetuning failure is dominated by policy mismatch or dynamics-model mismatch under distribution shift, then selectively updates the relevant part using low-rank adaptation. In its CARLA Table 2, for the ROM03 task, AdaWM reports TTC 2.05 and success rate 0.82; the listed DreamerV3 success rate is 0.40. The paper describes 12 hours of pretraining and one hour of finetuning on a single V100 GPU. This supports separating policy and model mismatch when diagnosing transfer failures. It assumes a pretrained model and policy and does not demonstrate a remedy for scratch training; some baseline comparisons use existing offline checkpoints without finetuning. No official code repository was linked from the ICLR record or paper during this search.

Evidence: [official ICLR record](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d4c745dcbaf8ef0d7e145754e31b1516-Abstract-Conference.html); [official ICLR paper PDF, Table 2 and experimental protocol](https://proceedings.iclr.cc/paper_files/paper/2025/file/d4c745dcbaf8ef0d7e145754e31b1516-Paper-Conference.pdf).

Screening scores: **4 / 3 / 4**.

## Date-boundary and screened candidates

**Think2Drive** is retained as a date-boundary/history anchor, not part of the six-paper shortlist. Springer lists the ECCV 2024 conference paper as first online on **24 November 2024**, inside the requested window, even though the conference meeting took place in September–October and its arXiv version predates the window. On the paper’s CornerCaseRepo table (mean over 3 runs), it reports 99.6±0.1 route completion; this is not a claim of 100% official Leaderboard completion. The method trains a planner in a learned latent world model and adds scenario generation and termination-priority replay. The official author page links the paper and demos, but no standalone original training-code release was verified; the later CaRL repository says it contains a Think2Drive reproduction. Sources: [Springer publication record](https://link.springer.com/chapter/10.1007/978-3-031-72995-9_9), [official ECCV paper PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06129.pdf), [CornerCaseRepo author page](https://thinklab-sjtu.github.io/CornerCaseRepo/), [CaRL repository description](https://github.com/autonomousvision/CaRL).

**RAD** remains a candidate but is outside the six selected entries to keep this slice compact and because its 3DGS/log-replay setting is distant from structured-state interaction RL. It is NeurIPS 2025; the authors report collision ratio 0.089 for RL+IL versus 0.143 for pure RL and 0.229 for pure IL on their 337-scene closed-loop 3DGS evaluation. Crucially, other traffic participants are controlled by real-world log replay in the experiments, so the result does not establish ego-conditioned reactive behavior of other drivers. Official code: [hustvl/RAD](https://github.com/hustvl/RAD). Sources: [NeurIPS record](https://papers.neurips.cc/paper_files/paper/2025/hash/2ed3a566a0af6dcec424b988f1880ecc-Abstract-Conference.html), [paper PDF, interaction setup and Table 1/Table 4](https://papers.neurips.cc/paper_files/paper/2025/file/2ed3a566a0af6dcec424b988f1880ecc-Paper-Conference.pdf).

PlannerRFT (CVPR 2026) and DRS-RL (IROS 2025) were screened. PlannerRFT’s full author manuscript describes scenario-adaptive diffusion exploration and nuPlan closed-loop results, but its pretrained diffusion-planner/structured-input setting is outside the compact shortlist; its project page did not expose a code repository at search time. DRS-RL’s IEEE record verifies IROS 2025 and an abstract-level claim of up to 92.17% collision-rate reduction, but the full-text mechanism and evaluation denominators were not independently verified here, so that number is not used as evidence in the selected set. Sources: [CVPR official PlannerRFT record](https://openaccess.thecvf.com/content/CVPR2026/html/Li_PlannerRFT_Reinforcing_Diffusion_Planners_through_Closed-Loop_and_Sample-Efficient_Fine-Tuning_CVPR_2026_paper.html), [author manuscript](https://arxiv.org/html/2601.12901), [IEEE DRS-RL record](https://ieeexplore.ieee.org/abstract/document/11247055).

These screening scores prioritize fit and evidence traceability; they are not estimates of venue acceptance or the probability that a method will improve another system.

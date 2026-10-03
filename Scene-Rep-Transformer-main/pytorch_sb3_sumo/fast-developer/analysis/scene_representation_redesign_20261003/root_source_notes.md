# Root literature notes — scene representation redesign

Date: 2026-10-03. Research window: 2024-10-03 to 2026-10-03. These notes supplement the two literature agents' primary-source reviews. They are not project experiment results. No new training or simulator evaluation was run for this task.

## Scope and exclusions

Searches covered topology/agent-map graphs, future interaction topology, multimodal joint prediction, conflict-zone resource graphs, graph RL, and 2026 updates. Generic graph-attention papers, map perception without a decision interface, traffic-signal control, unrelated human motion, unverified venue claims, and MDPI sources are not used to support the proposed method. Search-result timestamps are not publication dates. An accepted/issued paper is distinguished from a preprint and a conference workshop from its main track.

Representative public queries (no private paths, results or proposed method text were sent):
- `CVPR 2025 autonomous driving topology interaction graph motion prediction attention conflict`
- `ICCV 2025 lane topology interaction graph motion prediction`
- `NeurIPS 2025 motion prediction graph topological interaction uncertainty`
- `intersection conflict graph representation 2025 autonomous driving reinforcement learning`
- `motion prediction conflict hypergraph 2025 2026`
- `trajectory prediction homotopy 2025 topology`
- `joint occupancy motion prediction CVPR 2025`
- `motion forecasting factor graph 2025`
- `site.openaccess.thecvf.com/content/CVPR2026 trajectory interaction prediction`
- `site.proceedings.neurips.cc/paper_files/paper/2025 joint motion prediction`

## Additional primary sources

### CfDCA — published; full text inspected

Alireza Soltani, David M. Levinson, Mohsen Ramezani. *Communication-free Distributed Control Algorithm for autonomous vehicles at intersections*. Transportation Research Part C 180, 105309 (2025); online 2025-08-28. DOI: 10.1016/j.trc.2025.105309.

Source: https://transportlab.sydney.edu.au/wp-content/uploads/2025/09/AS-DL-MR-2025.pdf

Vehicle/conflict-zone resource-acquisition graphs, priority ordering, and entry/exit clearance already exist. Its safety reasoning assumes prescribed lane-based paths, coverage of conflict regions, vehicle dynamics limits, and all AVs following the protocol. Those assumptions do not describe a single SAC ego among SUMO background controllers. Resource temporal overlap is therefore a useful representation primitive, not a safety theorem for our proposed encoder. The article compares delay/throughput to signal control and FCFS; it is not evidence for learned SAC representation gains. No CAS partition is asserted.

### RAP — published; full text inspected

Xiaolong Tang, Meina Kan, Shiguang Shan, Xilin Chen. *RAP: Role-Aware Joint Prediction and Planning in Autonomous Driving*. IEEE Robotics and Automation Letters 11(2), February 2026. Author PDF filename says 2025, but the journal issue is 2026; do not label it a 2025 issue.

Source: https://vipl-epp.github.io/pdf/2025RAL-RAP.pdf

RAP distinguishes ego and surrounding actors in information, objectives and feedback, while decoding multimodal joint futures. Its ablations warn that ego history shortcuts can improve open-loop scores while harming closed-loop behavior; applying ego-oriented safety objectives to predicted neighbors can unrealistically make them yield. This is a direct reason not to train our background predictor to satisfy the ego's desired safety outcome. Joint futures and role tokens are existing ideas, not proposed novelty. The paper evaluates nuPlan, not SAC/SUMO. No CAS partition is asserted.

### ModeSeq — published; official paper retrieval partially available

Zikang Zhou et al. *ModeSeq: Taming Sparse Multimodal Motion Prediction with Sequential Mode Modeling*. CVPR 2025, pp. 1612 onward.

Source: https://openaccess.thecvf.com/content/CVPR2025/papers/Zhou_ModeSeq_Taming_Sparse_Multimodal_Motion_Prediction_with_Sequential_Mode_Modeling_CVPR_2025_paper.pdf

Official indexed paper text describes sequential mode decoding and Early-Match-Take-All training, addressing diversity and mode-confidence problems of winner-take-all prediction. Direct subsequent PDF/HTML retrieval failed. This supports only the caution that multiple output modes do not automatically yield calibrated probabilities. Do not claim a detailed numerical comparison or closed-loop RL improvement from this limited read.

### Conflict-aware HGRL — published; publisher abstract only

*Cooperative decision-making in mixed traffic via conflict-aware Heterogeneous Graph Reinforcement Learning*. Simulation Modelling Practice and Theory 148 (April 2026), 103268. DOI: 10.1016/j.simpat.2026.103268.

Source: https://www.sciencedirect.com/science/article/abs/pii/S1569190X26000171

Publisher search text describes conflict-centric heterogeneous graph modeling, sparse attention and execution constraints for mixed-traffic intersection control. Direct page retrieval returned 403. Treat it as a novelty warning against claiming conflict-aware graph RL as new; its exact equations, baselines and effect sizes remain unverified. Do not give it a guessed CCF/CAS rank.

### RoC-GRL — discovery only; full metadata not verified

*RoC-GRL: A right-of-way communication graph reinforcement learning multi-agent decision-making method for mixed-traffic unsignalized intersection*. DOI: 10.1016/j.eswa.2026.134435.

Source: https://www.sciencedirect.com/science/article/pii/S0957417426033397

Publisher-indexed abstract mentions local right-of-way graphs, global CAV communication graphs and attention with TD3. Exact online date was not recovered and direct page access returned 403. Do not count this as a verified within-window core paper or as a basis for claims about a single noncommunicating ego.

### RuleNet — published; publisher abstract inspected

Ruolin Shi, Xuesong Wang, Yang Zhou, Meixin Zhu. *RuleNet: rule-priority-aware multi-agent trajectory prediction in ambiguous traffic scenarios*. Transportation Research Part C, November 2025, 105339. DOI: 10.1016/j.trc.2025.105339.

Source: https://www.sciencedirect.com/science/article/pii/S0968090X25003432

Combines graph trajectory/map representations and attention with rule-priority refinement quantified using signal temporal logic. Its evidence concerns INTERACTION prediction, not closed-loop SAC success. Rule-aware graph attention is thus also an established category. Detailed causal attribution and safety claims cannot be inferred from the abstract.

### Recent supplementary candidates

- UniMotion, NeurIPS 2025, DOI 10.52202/085713-4063: https://proceedings.neurips.cc/paper_files/paper/2025/hash/b0499a1aecf036d42074d03f621d7864-Abstract-Conference.html . Official abstract confirms shared Transformer representations for simulation/prediction/planning; large unified model is not the minimal proposed route. Abstract only.
- JointMotion, CoRL 2024 conference / PMLR 270 publication 2025: https://proceedings.mlr.press/v270/wagner25a.html . Scene/instance self-supervision is existing practice; distinguish conference year from proceedings publication year. Abstract only.
- *Trajectory Prediction Considering Asymmetric Interactions Among Heterogeneous Traffic Participants*, IEEE TVT early access 2026-05-21, DOI 10.1109/TVT.2026.3695709: https://doi.org/10.1109/TVT.2026.3695709 . Publisher-indexed abstract describes asymmetric interaction intensity and auxiliary supervision. No journal partition or closed-loop result asserted.
- EdgeVTP, CVPR **Workshops** 2026, pp. 3712–3723: https://openaccess.thecvf.com/content/CVPR2026W/EVW/html/Kim_EdgeVTP_Exploration_of_Latency-efficient_Trajectory_Prediction_for_Edge-based_Embedded_Vision_CVPRW_2026_paper.html . Graph/Transformer with bounded latency; highway prediction, not a main-track CCF-A acceptance or intersection decision baseline.

### Older novelty anchor, outside the recent window

Vinicius Trentin, Antonio Artuñedo, Jorge Godoy, Jorge Villagra. *Multi-Modal Interaction-Aware Motion Prediction at Unsignalized Intersections*. IEEE T-IV 8(5):3349–3365, 2023.

Author manuscript: https://autopia.car.upm-csic.es/wp-content/papercite-data/pdf/trentin2023_multimodalinteraction.pdf

The author-hosted indexed text already uses navigable corridors, geometric conflict regions, Bayesian interaction/intent inference and Markov motion prediction. Consequently, even probabilistic corridor/conflict modeling is not inherently new. Direct follow-up PDF retrieval failed; do not claim a complete reproduction audit.

## Corrections made during review

GraphAD Eq. (1) uses **min over synchronized future times** of inter-trajectory distance, not all cross-time pairs and not only current distance. It also represents an actor's different motion modalities as different dynamic nodes. Any proposed difference must acknowledge both. Source: https://www.ijcai.org/proceedings/2025/0270.pdf . Its use of mode-indexed nodes does not by itself establish a globally aligned joint probability distribution across actors; that narrower distinction still requires an empirical control rather than a novelty assertion.

## Confidence boundary

The search identifies close prior art; it does not establish that the final candidate is the first of its kind. Before a paper claim, compare equations/code and follow the citation graph of GraphAD, BeTop, QCNeXt, RAP, CfDCA and the older interaction-aware corridor work. Publication level cannot be inferred from a combination of their components.

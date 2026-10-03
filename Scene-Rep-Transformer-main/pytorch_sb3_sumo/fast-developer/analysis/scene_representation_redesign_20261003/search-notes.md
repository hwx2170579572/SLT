# Literature search and screening notes

Search date: 2026-10-03. Target window: 2024-10-03 through 2026-10-03. Purpose: identify prior art for a structured scene representation over lane topology, interacting vehicles, uncertain futures, and conflict-resource events, while keeping publication date, task, and evidence type attached. These notes combine `literature_topology.md`, `literature_interaction.md`, and `root_source_notes.md`; the selected 12 entries and their read depths are indexed in `papers.md` and `papers.csv`.

## Query log

Only public, generic queries were used; no project paths, private experiment values, or unpublished method descriptions were sent to search services.

The topology/scene-representation search used:

- `traffic topology scene graph lane connectivity traffic signal`
- `interaction scene graph agent map future trajectory`
- `lane topology graph generation autonomous driving ICCV CVPR 2025`
- `multi-agent behavioral topology interactive autonomous driving`
- `future-aware motion forecasting agent map temporal`
- `scene graph conflict zone vehicle intersection resource acquisition`
- `scene-agent-goal autonomous driving 2026`

The interaction and joint-future search used:

- `autonomous driving vehicle interaction graph joint prediction planning NeurIPS 2024 2025`
- `vehicle interaction hypergraph trajectory prediction AAAI 2025`
- `small-world interaction flow-aware trajectory prediction TPAMI 2026`
- `causal interaction representations sim-to-real motion forecasting CVPR 2025`
- `joint prediction planning role-aware autonomous driving RAL 2026`
- `sequential mode modeling sparse multimodal motion prediction CVPR 2025`
- `causal logic trajectory prediction autonomous driving IJCAI 2025`
- `joint multi-agent trajectory prediction QCNeXt`
- `intersection conflict graph representation 2025 autonomous driving reinforcement learning`
- `motion prediction conflict hypergraph 2025 2026`
- `trajectory prediction conflict region interaction event autonomous driving`
- official-domain searches for CVPR/ICCV 2025–2026, NeurIPS 2024–2025, IJCAI 2025, AAAI 2025, and PMLR/ICML 2025 records.

## Source and selection rules

1. Formal venue and publication metadata come from proceedings, publisher records, or journal issue metadata. arXiv dates are listed separately and are not treated as conference publication dates.
2. Mechanism and numbers are taken from papers/PDFs. An abstract or repository-only screen is marked as such; it is not upgraded to a full-paper read. No quantitative result is filled in if the result table was not read.
3. The selected table prioritizes the direct representation blockers: GraphAD, BeTop, T²SG, SeqGrowGraph, FINet, SWIFT, NEST, ModeSeq, RAP, and CfDCA. MomAD and Flow Planner provide temporal/closed-loop planning context. This is a topical shortlist, not a venue-ranking list.
4. Code links are included only where an author/project repository was identified. A link is not a replication audit; no code was run.
5. Read depth is descriptive: full paper and selected tables; full text with tables but partial official-PDF retrieval; methods/selected ablations only; or abstract/repository only. Confidence refers to confidence in the cited metadata/mechanism/result as summarized, not to external validity.
6. No CCF or CAS classification is stated. Conference acceptance or journal publication does not make a motion-forecasting score comparable to the project’s SAC result.

## Date and source cautions

- **SWIFT**: arXiv v1 is dated 2026-07-03, within the search window; arXiv comments and the author page confirm TPAMI acceptance. The exact formal issue/publication date was not independently established by the 2026-10-03 cutoff, so this index does not force an issue date.
- **BeTop**: arXiv v1 predates the window (2024-09-26), but the formal NeurIPS 2024 proceedings paper is in-window. The screening uses the formal conference record, not the preprint timestamp.
- **RAP**: the author PDF filename contains “2025”; the formal venue metadata is IEEE Robotics and Automation Letters 11(2), February 2026. Use the journal issue date in references.
- **NEST**: use the official AAAI 2025 article/PDF metadata; no author repository was verified.
- **ModeSeq**: the official CVF entry identifies the CVPR 2025 paper, while the arXiv full text was used for the table-level read after direct proceedings-PDF retrieval was partial.
- **Flow Planner**: the NeurIPS page/repository confirms the formal 2025 venue and code; this index only screened its abstract/repository, so it carries no detailed numerical claim.
- **SeqGrowGraph**: official ICCV page/repository and formulation were checked, but the main benchmark values were not transcribed; no score is inferred from “state of the art.”

## Screened but not part of the 12-paper table

The companion notes retain additional relevant material without treating it as equivalent to the core evidence: InteractionMap (CVPR 2025) and TopoStreamer (arXiv 2025) focus on map perception; GoIRL (ICML 2025) uses lane graph features for IRL-based prediction; Foresight in Motion (ICCV 2025) is reward-guided prediction; SGDrive (CVPR 2026) is a VLM scene-agent-goal hierarchy; RuleNet (Transportation Research Part C, 2025) combines graph trajectory/map representation and rule-priority refinement; conflict-aware HGRL (2026) was available only at publisher abstract level; RoC-GRL metadata/date were not fully verified. QCNeXt (2023 workshop/preprint), LaneGCN, MTR, QCNet and GameFormer are outside the time window and retained only as older novelty anchors. Further details and source links are in the companion topology/interaction/root notes.

## Evidence boundary for the proposed representation

The shortlist does not establish novelty or a closed-loop SAC gain. Existing work already covers multimodal agent graph nodes and future geometric agent/map edges (GraphAD), future behavioral topology and topology-guided attention (BeTop), explicit directed lane topology (T²SG/SeqGrowGraph), learned small-world/hypergraph interactions (SWIFT/NEST), per-agent mode dependence (ModeSeq), historical-query planning (MomAD), and vehicle–conflict-zone resource acquisition with entry/exit/clearance (CfDCA). RAP and BeTop report simulator closed-loop planning, but neither isolates an off-policy SAC critic representation. The key unknown is whether a scene-level correlated mode over path-ordered resource occupancy intervals gives a more useful online critic state/action ranking than these existing structures or their simple combinations. This is a hypothesis requiring controlled evidence, not a literature fact.

No source code, training, simulation, evaluation, or shared research-record edits were performed for this index.

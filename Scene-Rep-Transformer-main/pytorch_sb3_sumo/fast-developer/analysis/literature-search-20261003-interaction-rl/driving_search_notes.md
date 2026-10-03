# Search notes

Access date: 2026-10-03. Scope: formal publications from 2024-10-03 through 2026-10-03 on autonomous-driving interaction decisions, risk/safety, world models, and closed-loop reinforcement learning. The user asked that formal publication dates be distinguished from preprints; Think2Drive is the boundary case because Springer lists its ECCV 2024 paper as first online 2024-11-24, while the arXiv preprint appeared earlier.

## Queries used

- `autonomous driving reinforcement learning interaction decision world model risk planning NeurIPS ICML AAAI 2025 closed-loop`
- `site:openaccess.thecvf.com 2025 autonomous driving world model planning interaction risk CVPR`
- `site:proceedings.neurips.cc autonomous driving closed-loop benchmark 2024 driving reinforcement learning`
- `site:openreview.net autonomous driving reinforcement learning world model 2025 planning interaction`
- `2025 autonomous driving RL safety-aware collision risk planner paper IEEE RA-L`
- `2025 2026 end-to-end autonomous driving closed-loop RL interaction planning paper conference`
- Exact-title searches for CaRL, Raw2Drive, SafeDrive, Bench2Drive, NAVSIM, AdaWM, Think2Drive, RAD, PlannerRFT, and DRS-RL were used to verify official records, full papers, and author repositories.

Searches used broad public research topics and public paper titles. No unpublished project method names, results, or code were sent to external services.

## Source and selection procedure

I prioritized official conference proceedings, publisher records, primary paper PDFs, author project pages, and author repositories. Venue/year status comes from publisher or official proceedings pages. Quantitative claims are transcribed with the paper’s own benchmark, metric, and sample scope. A paper’s reported benchmark result is not treated as a prediction for another simulator or policy class. No CCF or CAS journal-tier labels were assigned because no official classification source was checked for this note.

The six selected works are CaRL, Raw2Drive, SafeDrive, Bench2Drive, NAVSIM, and AdaWM. The first three method papers cover reward/optimization, model alignment, and fine-grained safety reasoning; the two benchmarks cover scenario-separated closed-loop assessment and a cheaper non-reactive metric screen; AdaWM separates model and policy mismatch under pretrained finetuning. Think2Drive is retained only as a formal-date and historical anchor. RAD is retained as a candidate because it provides a useful RL/IL and reward ablation, but its log-replayed traffic differs from ego-conditioned interactive agents. PlannerRFT was screened but left out of this six-paper compact slice because it is an IL-pretrained diffusion planner with structured inputs; DRS-RL was left out because only the IEEE abstract claim and metadata were verified here, not its full mechanism or evaluation denominators.

## Exclusions and unknowns

- Pure visual perception papers without planning, interaction, risk, world-model, or closed-loop decision evidence were not selected.
- LLM/VLM-only planning papers were not selected because they do not directly address the structured decision and closed-loop RL scope of this slice.
- Preprint-only results without a formal venue record were not presented as accepted conference evidence.
- No MDPI source was used.
- Official code status is stated conservatively: CaRL, SafeDrive, Bench2Drive, NAVSIM, and RAD have author repositories; Raw2Drive’s README describes official inference code; no standalone original Think2Drive training release or AdaWM repository was verified; PlannerRFT’s author project page did not link a training repository.
- Results with different input modalities, traffic reactivity, route splits, training budgets, or metrics are not directly rankable. The entries are evidence leads and design references, not a fair benchmark comparison.

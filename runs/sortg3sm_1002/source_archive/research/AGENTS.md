# Research context for fast-developer

These instructions concern research work under
`Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer` and its dependencies.

## Read before continuing this research

1. `Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer/RESEARCH_CONTEXT.md`
2. `Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer/analysis/full_mst_slt_implementation.md`
3. `Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer/analysis/experiment_history.md`

The user has explicitly defined the research roles:
- **full** is the complete method being investigated.
- **MST+SLT** is the strong baseline.
- **SAC+MLP** is the pure reinforcement-learning baseline.
- The current study starts from SAC+MLP and adds components of full incrementally to identify their effects and failure causes.

For basic file search, listing, and reading, the user requests a **Luna sub-agent with max reasoning** (`gpt-6-luna`, `max`). The coordinating agent integrates evidence and research conclusions. Preserve this preference when supported by the available tools.

Treat the research records as source-backed snapshots, not as a substitute for fresh evidence. Distinguish user-confirmed objectives, static implementation facts, recorded experimental observations, hypotheses, and unknowns. Keep scenario, density, training and evaluation seeds, checkpoint choice, raw/decision steps, and continuation conditions attached to results. Do not interpret a named ablation as a single-variable comparison until its actual configuration and gradient paths have been checked.

Update these research records when this research materially changes; retain historical results and their limitations rather than replacing them with only the latest or best numbers.

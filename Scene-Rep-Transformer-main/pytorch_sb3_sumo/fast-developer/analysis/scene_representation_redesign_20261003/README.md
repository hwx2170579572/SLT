# Scene representation research — 2026-10-03

Start with [implementation_roadmap.md](implementation_roadmap.md) for the current stage-by-stage build plan, and [technical_route.md](technical_route.md) for the research derivation. They describe a candidate topology-constrained passage-event encoder for SAC, not an implemented method or a successful experiment.

- [environment_evidence.md](environment_evidence.md): current scenario, observability and source-code boundaries.
- [papers.md](papers.md), [papers.csv](papers.csv), [search-notes.md](search-notes.md): selected literature and search/screening provenance.
- [literature_topology.md](literature_topology.md), [literature_interaction.md](literature_interaction.md), [root_source_notes.md](root_source_notes.md): supporting literature notes, including close prior art and read-depth limitations.
- [review_notes.md](review_notes.md): independent review of the design's observation, joint-mode, determinism and factual-supervision contracts.
- [check_event_representation_counterexample.py](check_event_representation_counterexample.py): exact synthetic marginal-versus-joint availability example; passed with the project Python on 2026-10-03. It does not invoke SUMO or train a model.

Scope: user prioritizes scene representation over learning-mechanism changes. The previous alternative remains in [reward_literature_technical_route_20261003.md](../reward_literature_technical_route_20261003.md). No new training/evaluation was launched, no historical experiment values were replaced, and no publication-level novelty or success-rate gain is established by these documents.

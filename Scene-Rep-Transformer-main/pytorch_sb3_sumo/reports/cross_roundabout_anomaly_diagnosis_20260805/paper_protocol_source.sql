-- DuckDB-compatible access to the transcribed paper tables and v14 protocol.
-- The source PDF is tmp/pdfs/scene_rep_transformer_arxiv_v3.pdf; its Tables
-- I, II and VI were also visually checked on rendered pages 11 and 16.

SELECT *
FROM read_json_auto(
  'pytorch_sb3_sumo/experiments/sb3_sumo_paper/protocol_baselines_v14.json'
);

SELECT *
FROM read_json_auto(
  'pytorch_sb3_sumo/experiments/sb3_sumo_paper/paper_reference_tables.json'
);

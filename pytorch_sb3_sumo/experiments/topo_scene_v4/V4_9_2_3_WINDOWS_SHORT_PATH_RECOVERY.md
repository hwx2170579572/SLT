# v4.9.2.3 Windows short-path recovery

The first immutable recovery attempt was correctly isolated but failed before
training because its descriptive TensorBoard event path exceeded the Windows
path budget. The failed directory and log remain preserved.

v4.9.2.3 changes only fresh recovery output identity:

- run files use the repo-local `r_v4923/runs/rN_<logical-job-hash>` path;
- launcher logs use `r_v4923/logs/`;
- full logical job identity, command, source/recovery mapping, hashes, and gate
  results remain in the long-form engineering report directory.

No scientific command argument changes except `--output-dir` and
`--model-name`. The model, optimizer, data partitions, seeds, 50,000-step
budget, selector, validation, and frozen gates are identical. The adapter adds
no safety rule or action rewrite. It preserves attempt-all behavior and never
starts formal testing.


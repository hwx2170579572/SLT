# v4.8.3 promotion automation runbook v2

## Iteration trigger

The first automation version stopped the whole batch after the first promotion
job subprocess or acceptance failure.  The user changed the required operating
policy: every promotion cell must be attempted before the automation stops and
produces its statistics.

This v2 runbook and `tools/run_remaining_v4_8_3_promotion_v2.py` are new files.
The v1 script, runbook, state, and events remain unchanged as historical
evidence.  No scientific implementation or frozen v4.8.3 acceptance code is
modified.

## v2 execution policy

1. Derive all 12 jobs and their order from the frozen v4.8 contract.
2. If the first unaccepted job is already running, wait for its subprocess to
   finish.  If it is accepted, record acceptance.  If its launcher records a
   completed failure, record that failure and continue.
3. Invoke the existing v4.8.3 runner separately for every later unaccepted job
   with its exact job name and `--workers 1`.
4. After a job subprocess returns, validate its immutable run.  Record either
   the accepted result or the complete failure reason, then continue to the
   next frozen job.
5. After all jobs have been preaccepted or attempted, regenerate the partial or
   complete promotion summary and write one joint attempt summary.
6. Compute the unchanged scientific promotion gate only if all 12 jobs are
   accepted.  An incomplete matrix is reported but is never passed to the
   scientific gate.
7. Never launch formal.  Formal accepted count must remain exactly zero.

## Safety exceptions

“Continue after failure” applies to a job whose subprocess has returned or
whose launcher has emitted a failure record.  It does not authorize concurrent
GPU/SUMO runs when the current job may still be alive.  Therefore the following
remain hard automation stops:

- unknown current-job state with no filesystem/log activity for one hour;
- multiple immutable incomplete run directories before takeover;
- an incomplete directory appearing out of frozen order;
- failure to validate contracts, freezes, engineering receipts, or formal-zero
  status;
- orchestration I/O or lock corruption.

These stops protect evidence integrity; they are not scientific job failures.

## Runtime evidence

v2 writes distinct files under `results_topo_v4_8_promotion/automation/`:

- `remaining_promotion_v2_state.json`;
- `remaining_promotion_v2_events.jsonl`;
- `remaining_promotion_v2_child.log`;
- `remaining_promotion_v2.lock.json` while live;
- `remaining_promotion_v2_attempt_summary.json` after all jobs are attempted.

The attempt summary contains the frozen order, preexisting accepted jobs, every
v2 attempt and return code, accepted and failed job lists, acceptance reasons,
available per-run metrics, aggregate statistics over accepted real runs, gate
status, and confirmation that formal was not launched.

## Exit codes

- `0`: all 12 accepted and promotion gate passed;
- `2`: all 12 accepted but promotion gate failed;
- `3`: all jobs were attempted, one or more remained unaccepted, partial real
  results were summarized, and no scientific gate was computed;
- other nonzero/traceback: infrastructure safety stop before all attempts.

## Invocation

```powershell
& 'D:\Programs\Anaconda\envs\llm_pipeline\python.exe' `
  'tools\run_remaining_v4_8_3_promotion_v2.py' `
  --device cuda `
  --poll-seconds 30 `
  --stall-timeout-seconds 3600
```

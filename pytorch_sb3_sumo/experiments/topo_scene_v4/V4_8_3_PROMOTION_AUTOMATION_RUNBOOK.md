# v4.8.3 promotion automation runbook

## Scope

This is an operations-only addition around the frozen v4.8/v4.8.3 experiment
stack.  It does not change the scientific implementation, training command,
checkpoint selector, metrics, seeds, traffic, thresholds, acceptance checks,
result roots, or promotion order.

The automation script is `tools/run_remaining_v4_8_3_promotion.py`.  It may be
started while the first incomplete promotion job is already running.  It waits
for that immutable run to pass the existing v4.8.3 acceptance function instead
of launching a duplicate or overwriting its directory.  It then delegates all
remaining work to the existing v4.8.3 runner with `--job all --workers 1`.

## Frozen order

1. TemporalGraph: Cross seeds 0, 1
2. TemporalGraph: roundabout_medium seeds 0, 1
3. TemporalGraph: CARLA seeds 0, 1
4. selected v4 candidate: Cross seeds 0, 1
5. selected v4 candidate: roundabout_medium seeds 0, 1
6. selected v4 candidate: CARLA seeds 0, 1

Accepted immutable runs are skipped by the existing runner.  An incomplete run
directory is never overwritten.  Takeover is permitted only when exactly one
incomplete directory exists and it is the first unaccepted job in the frozen
order.  Multiple or out-of-order incomplete directories stop the automation.

## Stop and evidence rules

- One worker is mandatory; promotion jobs run serially.
- A currently running job is polled until accepted.  A recorded launcher
  failure or one hour without any run-tree/log activity stops the automation.
- Any remaining-job subprocess or acceptance failure stops the batch at that
  job.  Existing runner evidence and `last_execution.json` are preserved.
- After 12/12 accepted jobs, the script regenerates the promotion summary and
  computes the unchanged promotion gate.
- A failed gate exits with code 2 so deep attribution can begin.
- A passed gate exits with code 0.
- The script never issues a formal-stage command.  It additionally requires
  formal accepted count to remain exactly 0 before and after gate computation.

Runtime evidence is written under
`results_topo_v4_8_promotion/automation/`:

- `remaining_promotion_state.json`: latest atomic state or final gate result;
- `remaining_promotion_events.jsonl`: timestamped orchestration events;
- `remaining_promotion_child.log`: relayed runner, summary, gate, and status
  output;
- `remaining_promotion.lock.json`: exclusive live-orchestrator lock, removed on
  a clean exit.

## Invocation

Use the frozen project environment:

```powershell
& 'D:\Programs\Anaconda\envs\llm_pipeline\python.exe' `
  'tools\run_remaining_v4_8_3_promotion.py' `
  --device cuda `
  --poll-seconds 30 `
  --stall-timeout-seconds 3600
```

If the lock remains after an abnormal orchestrator termination, first verify
that its recorded PID is no longer the live automation process.  Remove only
that exact lock file; never delete or rewrite a promotion run directory.

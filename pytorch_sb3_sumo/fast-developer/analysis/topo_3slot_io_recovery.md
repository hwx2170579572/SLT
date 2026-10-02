# Topo / three-slot training I/O recovery — 2026-09-30

## Confirmed failure and retained evidence

Original run root:
`D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/runs/t0930_0023_topo3_100k`.

The three-slot worker exited with code 1 at about 00:37:05 Asia/Shanghai.
Its traceback originated in the behavior recorder while publishing
`diagnostics/train/summary.json`: Windows rejected
`summary.json.tmp.replace(summary.json)` with `PermissionError`, WinError 5.
The exception escaped the terminal environment step and stopped learning.
It did not originate in a model forward, gradient update or checkpoint save.

The worker status recorded approximately 5,651 raw steps. Its last periodic
progress snapshot was 5,382 raw steps and 382 updates. The retained behavior
summary subsequently contained 21 episodes and 5,653 raw records; cleanup
may have successfully refreshed it. These are different recording points,
not a completed 100k run or a method-performance result.

The target was not read-only, its ACL allowed modification, and its path
length was 173 characters. The particular handle/component responsible
for denying replacement is unknown. No permissions were weakened and no
external file handles were forcibly unlocked.

The ordinary Topo worker was still alive at inspection (PID 17496; original
start time 00:31:18), with a verified snapshot of 14,374 raw steps and 9,374
updates. Supervisor PID 56072 also matched its original start time. Both
methods share the affected recorder; the user requested a normal restart.
The original run and its outputs are retained. A restart uses fresh models
and a new directory so that both methods use the corrected I/O code.

## Correction and verification

The behavior recorder now retries summary publication after PermissionError
up to four attempts, with waits of 0.05, 0.10 and 0.20 seconds. If replacement
is still denied, normal training continues with the flushed JSONL streams.
The existing canonical summary remains intact and, when writable, the latest
complete snapshot remains in `summary.json.tmp` with `summary_write.pending=true`.
Subsequent normal flushes retry publication; successful recovery removes the
temporary file and records recovery counters. A pending temporary summary
must be considered when reading a stale canonical summary.

Critical metadata failures still propagate. Disk-full and unrelated I/O
errors are not swallowed. This change does not alter SAC losses, module
switches, evaluation cases or gradient updates.

Four injected-failure regression tests passed, covering transient denial,
persistent denial during episode termination, recovery, pending output at
close and strict metadata errors. Together with the existing behavior,
gradient-event and paired-comparison tests, the specified suite passed
25 tests in `D:/Programs/Anaconda/envs/pytorch/python.exe`.

The base progress/status JSON publisher also had an unguarded `os.replace`.
It now uses the same four-attempt bounded retry. Only derived `progress.json`
may defer publication: its last canonical file and complete PID-specific
temporary JSON remain available, and a warning is logged. Critical
status/configuration writes still fail explicitly if all retries are denied.
Four additional mock tests passed for this helper, including propagation of
unrelated OSError rather than treating it as a transient lock. The two
relevant test groups therefore passed 29 tests in total; compilation passed.

## Restart status

The original supervisor and worker identities were rechecked by PID and
process start time. The restricted execution environment initially denied
termination; the same specifically scoped command succeeded after execution
approval. Only the verified original supervisor tree was terminated. The
original files and checkpoints remain, with `restart_note.txt` distinguishing
the failed three-slot run from the deliberately interrupted Topo run.

The repaired hidden supervisor started at **2026-09-30 01:24:05 Asia/Shanghai**,
PID **50304**, under the new root:
`D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/runs/t0930_topo3_retry01`.
The original training/evaluation settings are retained: two CUDA workers,
seed 0, fresh continuous 100,000 raw steps, 5,000 raw-step warmup, checkpoint
every 10,000 raw steps and 100 final evaluation episodes per method.
The three-slot stage keeps Graph-SLT and SBS disabled. No previous
checkpoint is resumed and no smoke settings are used.

Verification now includes actual optimization beyond both the 5,000-step
warmup and the previous failure point, rather than startup alone:

| Method | Worker PID | Verified raw steps | Verified updates | State |
| --- | --- | --- | --- | --- |
| `sac_mlp_d1_st_rt_topo` | 8364 | 6,578 | 1,578 | training |
| `sac_mlp_d1_st_rt_topo_3slot` | 62860 | 6,577 | 1,577 | training |

Both workers were observed advancing through 2,691/0, 5,082/82,
5,680/680 and 6,279/1,279 raw-step/update snapshots before the final
verification above. Both Python PIDs appeared in the NVIDIA compute-process
listing. Their training summaries reported zero diagnostic errors,
`summary_write.pending=false` and zero permission denials/deferred flushes
in the replacement run. Each had 33 representation samples and all expected
behavior/optimization/representation streams. No traceback, PermissionError
or WinError was found in the new worker logs at this check.

`source_snapshot.json` records 16 source SHA256 entries and
`launcher_invocation.json` records the exact command and worker mapping.
The run is continuing toward 100k; these are verification snapshots, not
completion or scientific-performance results. Current progress is available
in each worker's `progress.json` (or its complete PID-specific temporary
snapshot if publication is pending).

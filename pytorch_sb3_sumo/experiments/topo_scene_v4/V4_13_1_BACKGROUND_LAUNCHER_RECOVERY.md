# v4.13.1 background-launcher recovery

This is an engineering-only recovery.  It does not change the frozen v4.13
model, loss, inference score, experiment matrix, seeds, gates, or formal lock.

The first hidden `Start-Process` launcher correctly detached the process, but
passed the absolute dispatcher path as an unquoted `ArgumentList` element.
Windows split the workspace path at `D:\Program Files (x86)`, so Python exited
before importing the dispatcher and before creating any experiment run.

The v4.13.1 launcher uses a fixed working directory and quote-safe relative
arguments for both the dispatcher and aggregate report.  It retains hidden,
terminal-independent execution, existing-process detection, automatic
discovery of current/future versioned pipelines, attempt-all semantics, and
the formal-test lock.  The failed v4.13 launcher log is preserved under
`results_promotion_automation/detached_logs_v4_13`.

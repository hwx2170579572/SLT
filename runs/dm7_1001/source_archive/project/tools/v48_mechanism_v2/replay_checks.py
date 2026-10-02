"""Small analytic checks of replay semantics, with historical deviations explicit."""
from __future__ import annotations
from .common import OUTPUT, cpu_environment, configure_torch, metadata, write_json

def run_checks():
    cpu_environment()
    configure_torch()
    import numpy as np
    from gymnasium import spaces
    from algos.sb3_torch.replay_buffer import DictNStepReplayBuffer
    from algos.sb3_torch.replay_buffer_v4_7 import HorizonCorrectDictNStepReplayBufferV47

    gamma = 0.99
    observations = spaces.Dict({
        "trajectory": spaces.Box(-100.0, 100.0, shape=(1,), dtype=np.float32),
        "map": spaces.Box(-100.0, 100.0, shape=(1,), dtype=np.float32),
    })
    actions = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    def make(cls, n, repeat=1, duplicate=False):
        return cls(32, observations, actions, device="cpu", n_envs=1,
                   n_steps=n, gamma=gamma, source_action_repeat=repeat,
                   duplicate_episode_end_transition=duplicate,
                   handle_timeout_termination=True)
    def add(buf, t, reward, done=False, timeout=False, raw=1):
        obs = {key: np.array([[t]], dtype=np.float32) for key in observations.spaces}
        nxt = {key: np.array([[t+1]], dtype=np.float32) for key in observations.spaces}
        buf.add(obs, nxt, np.array([[0.25]], dtype=np.float32),
                np.array([reward], dtype=np.float32),
                np.array([done], dtype=bool),
                [{"TimeLimit.truncated": timeout, "raw_steps_executed": raw}])
    def sample(buf, slot=0):
        return buf._get_samples(np.array([slot], dtype=np.int64), env=None)
    def scalar(tensor):
        return float(tensor.detach().cpu().reshape(-1)[0])
    def close(actual, expected, label):
        if not np.isclose(actual, expected, atol=1e-6, rtol=1e-6):
            raise AssertionError(f"{label}: {actual} != {expected}")
    def nonterminal(n):
        corrected = make(HorizonCorrectDictNStepReplayBufferV47, n)
        for i in range(n):
            add(corrected, i, i+1)
        s = sample(corrected)
        expected = sum(gamma**i * (i+1) for i in range(n))
        close(scalar(s.rewards), expected, "discounted decision reward")
        close(scalar(s.discounts), gamma**n, "actual-horizon bootstrap")
        close(scalar(s.dones), 0, "nonterminal")
        close(scalar(s.next_observations["trajectory"]), n, "bootstrap next state")
        close(scalar(s.one_step_next_observations["trajectory"]), 1, "one-step auxiliary next state")
        return {"n": n, "discount": scalar(s.discounts), "expected_discount": gamma**n,
                "reward": scalar(s.rewards), "expected_reward": expected,
                "actual_horizons": corrected.last_sample_actual_horizons.tolist()}
    def boundary(timeout):
        corrected = make(HorizonCorrectDictNStepReplayBufferV47, 16)
        for i in range(3):
            add(corrected, i, i+1, done=i==2, timeout=timeout and i==2)
        s = sample(corrected)
        close(scalar(s.rewards), sum(gamma**i*(i+1) for i in range(3)), "short return")
        close(scalar(s.discounts), gamma**3, "short actual horizon")
        close(scalar(s.dones), 0 if timeout else 1, "bootstrap terminal mask")
        close(scalar(s.one_step_next_observations["trajectory"]), 1, "short one-step target")
        return {"timeout": timeout, "discount": scalar(s.discounts),
                "bootstrap_done_mask": scalar(s.dones),
                "actual_horizons": corrected.last_sample_actual_horizons.tolist()}
    def legacy_difference():
        old = make(DictNStepReplayBuffer, 4)
        corrected = make(HorizonCorrectDictNStepReplayBufferV47, 4)
        for i in range(4):
            add(old, i, i+1)
            add(corrected, i, i+1)
        a, b = sample(old), sample(corrected)
        close(scalar(a.rewards), scalar(b.rewards), "same accumulated reward")
        close(scalar(a.discounts), gamma, "historical discount contract")
        close(scalar(b.discounts), gamma**4, "corrected discount contract")
        if np.isclose(scalar(a.discounts), scalar(b.discounts)):
            raise AssertionError("Expected to detect the historical compound-ablation difference")
        return {"legacy_discount": scalar(a.discounts),
                "corrected_discount": scalar(b.discounts),
                "scientific_finding": "Legacy four-step bootstrap uses gamma, not gamma**actual_horizon.",
                "new_controlled_training_gate": "Requires the chosen return protocol to be matched across methods."}
    def decision_storage():
        buf = make(HorizonCorrectDictNStepReplayBufferV47, 4, repeat=3)
        add(buf, 0, 1, raw=1)
        if buf.pos != 0:
            raise AssertionError("Nonboundary incomplete raw action hold was stored")
        add(buf, 1, 1, raw=3)
        if buf.pos != 1:
            raise AssertionError("Completed decision was not stored")
        add(buf, 2, 1, done=True, raw=1)
        if buf.pos != 2:
            raise AssertionError("Short boundary decision was not stored")
        return {"stored_decisions": int(buf.pos), "raw_partial_nonboundary_skipped": True}
    def duplicate_boundary():
        buf = make(HorizonCorrectDictNStepReplayBufferV47, 4, duplicate=True)
        for i in range(3):
            add(buf, i, 1, done=i==2)
        if buf.pos != 4:
            raise AssertionError(f"Expected explicit inherited boundary duplication, got {buf.pos}")
        s = sample(buf)
        close(scalar(s.rewards), 1+gamma+gamma**2, "return stops at first true terminal")
        close(scalar(s.dones), 1, "terminal mask")
        return {"insertions_for_three_decisions": int(buf.pos),
                "first_return_actual_horizon": buf.last_sample_actual_horizons.tolist(),
                "scientific_finding": "Terminal duplication is inherited behavior; this test documents it, not a proof it is optimal."}
    checks = []
    cases = [("n4_correct_discount_and_aux_target", lambda: nonterminal(4)),
             ("n16_correct_discount_and_aux_target", lambda: nonterminal(16)),
             ("short_true_terminal", lambda: boundary(False)),
             ("short_timeout_bootstrap", lambda: boundary(True)),
             ("legacy_discount_deviation_detected", legacy_difference),
             ("decision_vs_raw_step_storage", decision_storage),
             ("inherited_terminal_duplication", duplicate_boundary)]
    for name, fn in cases:
        try:
            result = fn()
            checks.append({"name": name, "status": "pass", "result": result})
        except Exception as exc:
            checks.append({"name": name, "status": "fail",
                           "error": f"{type(exc).__name__}: {exc}"})
    payload = {**metadata(), "checks": checks, "all_checks_passed": all(c["status"]=="pass" for c in checks),
               "interpretation": "Checks validate observations about the code. Detected historical deviations remain findings, not repaired behavior."}
    write_json(OUTPUT / "stage0" / "replay_checks.json", payload)
    return payload

if __name__ == "__main__":
    result = run_checks()
    print(__import__("json").dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["all_checks_passed"] else 1)


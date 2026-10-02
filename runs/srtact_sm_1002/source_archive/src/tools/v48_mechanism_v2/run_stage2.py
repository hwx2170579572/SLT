"""One serial diagnostic worker; dependency wait does not use a compute thread."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

from .common import cpu_environment
cpu_environment()
from .common import ROOT,OUTPUT,metadata,load_json,write_json


def run(wait_for_control=False):
    start=time.time()
    status=OUTPUT / "stage2/pipeline.json"
    while not (OUTPUT / "stage2/control/complete.json").exists():
        if not wait_for_control:
            raise RuntimeError("Complete the serial control collector first")
        write_json(status,dict(**metadata(),state="waiting_for_control_dependency",elapsed_seconds=time.time()-start))
        if time.time()-start>7200:
            raise TimeoutError("Control collection dependency did not complete in two hours")
        time.sleep(15)
    assert load_json(OUTPUT / "stage2/control/complete.json")["traffic_pairing_verified"]
    stages=[
        ("tests",["pytest","tests_sb3_sumo/test_v48_mechanism_v2.py","-q","--disable-warnings","--junitxml=r48m2/stage2/pytest.xml"],"stage2/pytest.xml"),
        ("restored_state",["tools.v48_mechanism_v2.stage0_supplement"],"stage0/restored_training_state.json"),
        ("local",["tools.v48_mechanism_v2.local_analysis"],"stage2/local/carla/result.json"),
        ("gradients",["tools.v48_mechanism_v2.gradient_diagnostics"],"stage2/gradients/carla.json"),
        ("actions",["tools.v48_mechanism_v2.action_diagnostics"],"stage2/actions/carla/result.json"),
        ("branch_smoke",["tools.v48_mechanism_v2.branches","--smoke"],"smoke/branches/complete.json"),
        ("branches",["tools.v48_mechanism_v2.branches"],"stage2/branches/complete.json"),
        ("branch_probes",["tools.v48_mechanism_v2.branches","--analyze"],"stage2/branches/carla/probe_result.json"),
        ("future",["tools.v48_mechanism_v2.control_analysis"],"stage2/future/carla/result.json"),
        ("report",["tools.v48_mechanism_v2.report"],"evidence_manifest.json"),
    ]
    completed=[]
    for name,args,artifact in stages:
        path=OUTPUT / artifact
        # Completed identities are checked inside the collection/encoding code.
        # Analysis jobs only rerun after an explicit restart, and never retrain policies.
        if path.exists() and name not in ("tests","report"):
            completed.append(dict(stage=name,status="reused_completed_artifact"))
            continue
        write_json(status,dict(**metadata(),state="running",stage=name,completed=completed,elapsed_seconds=time.time()-start))
        print("START "+name,flush=True)
        log=OUTPUT / "stage2/logs" / (name+".log")
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open("w",encoding="utf-8") as stream:
            process=subprocess.run([sys.executable,"-m",*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        if process.returncode!=0:
            write_json(status,dict(**metadata(),state="failed",stage=name,exit_code=process.returncode,
                log=str(log),completed=completed,elapsed_seconds=time.time()-start))
            raise RuntimeError(f"Diagnostic stage failed: {name}; inspect {log}")
        if not path.exists():
            raise RuntimeError(f"Stage {name} did not produce its completion artifact")
        completed.append(dict(stage=name,status="completed"))
        print("DONE "+name,flush=True)
    write_json(status,dict(**metadata(),state="complete",completed=completed,elapsed_seconds=time.time()-start))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--wait-for-control",action="store_true")
    run(parser.parse_args().wait_for_control)

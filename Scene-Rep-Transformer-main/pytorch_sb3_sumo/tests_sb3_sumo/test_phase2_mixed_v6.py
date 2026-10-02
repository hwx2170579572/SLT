from tools.control_phase2_mixed_v6 import available_device,adopt_running


def test_third_gpu_replaces_interrupted_cpu_slot():
    w=[{'device':'cuda'},{'device':'cuda'},{'device':'cpu'}]
    assert available_device(w)=='cuda'
    assert available_device(w+[{'device':'cuda'}]) is None


def test_only_one_cpu_slot_remains():
    w=[{'device':'cuda'}]*3
    assert available_device(w)=='cpu'
    assert available_device(w+[{'device':'cpu'}]) is None


def test_active_job_not_rescheduled():
    p,a=adopt_running([{'id':'existing'},{'id':'new'}],[{'jobid':'existing','pid':42}])
    assert p==[{'id':'new'}] and set(a)=={'existing'}

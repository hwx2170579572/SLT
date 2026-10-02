from tools.control_phase2_mixed_v5 import available_device,adopt_running


def test_two_gpu_two_cpu_limit():
    w=[{'device':'cuda'},{'device':'cuda'},{'device':'cpu'}]
    assert available_device(w)=='cpu'
    assert available_device(w+[{'device':'cpu'}]) is None


def test_gpu_can_refill_with_two_cpu():
    assert available_device([{'device':'cuda'},{'device':'cpu'},{'device':'cpu'}])=='cuda'


def test_existing_tasks_adopted_not_rescheduled():
    p,a=adopt_running([{'id':'a'},{'id':'b'}],[{'jobid':'a','pid':1},{'jobid':'external','pid':2}])
    assert p==[{'id':'b'}] and set(a)=={'a'}

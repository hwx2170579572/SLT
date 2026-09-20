from tools.control_phase2_gpu_v7 import available_device,adopt_running


def test_cpu_completion_does_not_create_replacement_cpu():
    g=[{'device':'cuda'}]*3
    assert available_device(g+[{'device':'cpu'}]) is None
    assert available_device(g) is None


def test_only_gpu_slots_are_refilled():
    assert available_device([{'device':'cuda'}]*2+[{'device':'cpu'}])=='cuda'
    assert available_device([])=='cuda'


def test_current_cpu_task_is_adopted_until_finished():
    task={'jobid':'cpu_job','pid':13608,'device':'cpu'}
    pending,adopted=adopt_running([{'id':'cpu_job'},{'id':'new'}],[task])
    assert adopted=={'cpu_job':task} and pending==[{'id':'new'}]

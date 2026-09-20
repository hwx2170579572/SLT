from tools.control_phase2_mixed_v4 import available_device


def test_adds_cpu_without_displacing_two_existing_cuda_workers():
    workers=[{'device':'cuda'},{'device':'cuda'}]
    assert available_device(workers)=='cpu'
    assert available_device(workers+[{'device':'cpu'}]) is None


def test_gpu_slot_can_refill_while_cpu_runs():
    assert available_device([{'device':'cuda'},{'device':'cpu'}])=='cuda'


def test_one_cpu_slot_only_and_cuda_index_counts():
    assert available_device([{'device':'cuda:0'},{'device':'cuda'},{'device':'cpu'}]) is None

import time
import torch
import numpy as np

def diff(dataA, dataB):
    print(dataA)
    print(dataB)
    if isinstance(dataA, torch.Tensor):
        dataA = dataA.cpu().numpy()
    if isinstance(dataB, torch.Tensor):
        dataB = dataB.cpu().numpy()

    print("max:", np.max(np.abs(dataA - dataB)))
    print("mean:", np.mean(np.abs(dataA - dataB)))
    print("min:", np.min(np.abs(dataA - dataB)))
    return np.max(np.abs(dataA - dataB))


def timeit(func,):
    def run_func(*args, **kwargs):
        st = time.time()
        result = func(*args, **kwargs)
        run_time = time.time() - st
        print(f"{func.__name__} run time is {run_time*1000} ms")
        return result
    return run_func
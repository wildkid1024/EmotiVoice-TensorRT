import os
import re
import time
import torch
import numpy as np
import onnxruntime as ort
from typing import Optional


def get_sess(model_path: Optional[os.PathLike],
             device: str='cpu',
             cpu_threads: int=1,
             use_trt: bool=False):
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if 'cuda' in device.lower():
        device_id = int(device.split(':')[1]) if len(
            device.split(':')) == 2 else 0
        # fastspeech2/mb_melgan can't use trt now!
        if use_trt:
            provider_name = 'TensorrtExecutionProvider'
        else:
            provider_name = 'CUDAExecutionProvider'
        providers = [(provider_name, {'device_id': device_id})]
    elif device.lower() == 'cpu':
        providers = ['CPUExecutionProvider']
    sess_options.intra_op_num_threads = cpu_threads
    sess = ort.InferenceSession(
        model_path, providers=providers, sess_options=sess_options)
    return sess
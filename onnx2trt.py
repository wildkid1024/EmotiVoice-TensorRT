import numpy as np
import torch
from functools import lru_cache, cache
from polygraphy import cuda

from engines.trt_engine import Engine, device_view
from utils.profile import diff

stream = cuda.Stream()

@lru_cache(maxsize=10)
def get_trt_engine(trt_path="am.trt"):
    engine = Engine(trt_path)
    engine.load()
    engine.activate()
    return engine

def run_trt(onnx_path="am.onnx", trt_path="am.trt", build=False, input_profile=None, inputs_shape=None, inputs_feed=None, output_names=None):
    engine = get_trt_engine(trt_path=trt_path)
    # if build: engine.build(onnx_path, fp16=True, input_profile=input_profile)
    engine.allocate_buffers(shape_dict=inputs_shape, batch_size=1)
    outputs = engine.infer(feed_dict=inputs_feed, stream=stream)
    return_pt = [outputs[output_name] for output_name in output_names]
    return return_pt if len(return_pt) > 1 else return_pt[0]


def run_am_encoder(build=False, inputs=None, outputs=None):
    onnx_path="outputs/onnx/am_encoder.onnx"
    trt_path="outputs/trt/am_encoder.trt"

    input_profile = {
        key: [val.size(), val.size(), val.size()]
        for key, val in inputs.items() if key != 'input_lengths'
    }
    input_profile.update({
        'inputs_ling': [[1, 1], [1, 256], [1, 512]]
    })

    inputs_shape = {
        key: list(val.size())
        for key, val in inputs.items() if key != 'input_lengths'
    }
    inputs_shape.update({
        'x': [1, inputs_shape['inputs_ling'][1], 384],
        'd_outs': [1, inputs_shape['inputs_ling'][1]]
    })

    inputs_feed = {
        key: device_view(val.cuda())
        for key,val in inputs.items() if key != 'input_lengths'
    }

    x, d_outs = run_trt(build=build, onnx_path=onnx_path, trt_path=trt_path, input_profile=input_profile, inputs_shape=inputs_shape, inputs_feed=inputs_feed, output_names=['x', 'd_outs'])
    # print(x.size())
    # print(d_outs.size())
    return x, d_outs


def run_am_decoder(build=False, inputs=None, outputs=None):
    onnx_path="outputs/onnx/am_decoder.onnx"
    trt_path="outputs/trt/am_decoder.trt"

    input_profile = {
        key: [val.size(), val.size(), val.size()]
        for key, val in inputs.items() if key != 'input_lengths'
    }
    input_profile.update({
        'x': [[1, 1, 384], [1, 256, 384], [1, 512, 384]],
        'd_outs': [[1, 1], [1, 256], [1, 512]]
    })

    inputs_shape = {
        key: list(val.size())
        for key, val in inputs.items() if key != 'input_lengths'
    }
    mel_lenghs = torch.sum(inputs['d_outs'], dim=-1).int().max().item()
    inputs_shape.update({
        'logmel': [1, mel_lenghs, 80],
    })

    inputs_feed = {
        key: device_view(val.cuda())
        for key,val in inputs.items() if key != 'input_lengths'
    }

    logmel = run_trt(build=build, onnx_path=onnx_path, trt_path=trt_path, input_profile=input_profile, inputs_shape=inputs_shape, inputs_feed=inputs_feed, output_names=['logmel', ])
    # print(logmel.size())
    return logmel

def run_voc(build=False, inputs=None, outputs=None):
    onnx_path="outputs/onnx/voc.onnx"
    trt_path="outputs/trt/voc.trt"
    # print("输入的inputs:", inputs)

    input_profile = {
        'logmel': [[1, 1, 80], [1, 256, 80], [1, 512, 80]]
    }

    inputs_shape = {
        key: list(val.size())
        for key, val in inputs.items()
    }
    inputs_shape.update({
        'wav': [1, 1, inputs['logmel'].size(1) * 256]
        # 'wav': [1, 512, 241]
        # 1,256,1928
    })

    inputs_feed = {
        key: device_view(val.cuda())
        for key,val in inputs.items()
    }

    wav = run_trt(onnx_path=onnx_path, trt_path=trt_path,build=build, 
                  input_profile=input_profile, inputs_shape=inputs_shape, inputs_feed=inputs_feed, output_names=['wav', ])
    return wav


def run_embedding(build=False, inputs=None):
    onnx_path="outputs/onnx/embedding.onnx"
    trt_path="outputs/trt/embedding.trt"

    input_profile = {
        'input_ids': [[1, 1], [1, 256], [1, 512]]
    }

    inputs_shape = {
        key: list(val.size())
        for key, val in inputs.items() if key == 'input_ids'
    }
    inputs_shape.update({
        'style_embedding': [1, 768]
        # 'wav': [1, 512, 241]
        # 1,256,1928
    })

    inputs_feed = {
        key: device_view(val.cuda())
        for key,val in inputs.items() if key == 'input_ids'
    }

    style_embedding = run_trt(onnx_path=onnx_path, trt_path=trt_path, build=build, 
                  input_profile=input_profile, inputs_shape=inputs_shape, inputs_feed=inputs_feed, output_names=['style_embedding', ])
    return style_embedding
    
def run_embbeding_test():
    inputs = torch.load("outputs/embedding_inputs.pt", map_location='cuda')
    outputs = torch.load("outputs/embedding_outputs.pt", map_location='cuda')
    style_embedding = run_embedding(build=False, inputs=inputs)
    diff(style_embedding, outputs['style_embedding'])

def get_style_embedding(prompt, tokenizer):
    prompt = tokenizer([prompt], return_tensors="pt")
    input_ids = prompt["input_ids"]
    output = run_embedding(inputs=prompt)
    style_embedding = output.cpu().squeeze().numpy()
    return style_embedding


def onnx2trt():
    embed_inputs = torch.load("outputs/embedding_inputs.pt", map_location='cuda')
    embed_outputs = torch.load("outputs/embedding_outputs.pt", map_location='cuda')
    am_inputs = torch.load("outputs/inputs_2.pt", map_location='cuda')
    voc_inputs = torch.load("outputs/logmel_2.pt", map_location='cuda')

    style_embedding = run_embedding(build=True, inputs=embed_inputs)
    x, d_outs = run_am_encoder(build=True, inputs=am_inputs)
    logmel = run_am_decoder(build=True, inputs={"x": x, "d_outs": d_outs})
    wav = run_voc(build=True, inputs=voc_inputs)


if __name__ == "__main__":
    onnx2trt()






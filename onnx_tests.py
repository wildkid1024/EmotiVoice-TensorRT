import torch
from engines.onnx_engine import get_sess
from utils.profile import diff


def test_voc_ort(inputs=None, outputs=None):
    onnx_path = "generator.onnx"
    inputs_path = "inputs.pt"
    outputs_path = "outputs.pt"
    inputs = torch.load(inputs_path, map_location='cuda:0')
    outputs = torch.load(outputs_path, map_location='cuda:0')

    sess = get_sess(onnx_path, device='cpu', cpu_threads=8)
    inputs = {
        k: v.cpu().numpy() 
        for k, v in inputs.items() if k != 'input_lengths'
    }
    wav = sess.run(output_names=['wav'], input_feed=inputs)
    diff(wav[0], outputs['wav'])
    return wav[0]


def test_am_split_ort(inputs=None, outputs=None):
    am_onnx_path = "am.onnx"

    inputs_path = "inputs_2.pt"
    outputs_path = "outputs_2.pt"
    inputs = torch.load(inputs_path, map_location='cuda:0')
    outputs = torch.load(outputs_path, map_location='cuda:0')

    sess = get_sess(am_onnx_path, device='cpu', cpu_threads=8)
    inputs = {
        k: v.cpu().numpy() 
        for k, v in inputs.items() if k != 'input_lengths'
    }
    logmel = sess.run(output_names=['logmel'], input_feed=inputs)
    diff(logmel[0], outputs['logmel'])
    return logmel[0]
    


def test_embedding_ort(inputs=None, outputs=None):
    inputs_path = "embedding_inputs.pt"
    outputs_path = "embedding_outputs.pt"

    inputs = torch.load(inputs_path, map_location='cuda:0')
    outputs = torch.load(outputs_path, map_location='cuda:0')

    embed_onnx_path = "embedding.onnx"
    inputs = torch.load("embedding_inputs.pt", map_location='cpu')

    inputs = {
        'input_ids': inputs['input_ids'].cpu().numpy()
    }
    sess = get_sess(embed_onnx_path, device='cpu', cpu_threads=8)
    style_embedding = sess.run(input_feed=inputs, output_names=['style_embedding'])
    diff(style_embedding[0], outputs['style_embedding'])
    return style_embedding[0]

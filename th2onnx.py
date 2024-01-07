
import os
import torch

from th_models import TorchHiFiGAN, TorchPromptTTS, TorchSimBert
from th_models import get_models


def export_voc_model(voc_model, voc_onnx_path='outputs/onnx/voc.onnx', inputs_dict=None):
    print(inputs_dict.keys())
    for param in inputs_dict.values():
        print(param.size())
    
    torch.onnx.export(
        model = voc_model,
        args = inputs_dict,
        f = voc_onnx_path, 
        export_params = True,
        opset_version=19,
        verbose = False, 
        input_names = ["logmel"], 
        output_names = ["wav"],
        do_constant_folding=True, 
        dynamic_axes = {
            'logmel': {1: 'seq_len'}
        }, 
    )

def export_am_split_model(
            am_mdoel,
            am_encoder_onnx_path = "outputs/onnx/am_encoder.onnx", 
            am_decoder_onnx_path = "outputs/onnx/am_decoder.onnx",
            inputs_dict = None,
        ):
    print(inputs_dict.keys())
    for param in inputs_dict.values():
        print(param.size())

    encoder_outputs = am_mdoel.encoder_forward(**inputs_dict)
    print("am_outputs:", encoder_outputs[0].size(), encoder_outputs[1].size())
    am_mdoel.forward = am_mdoel.encoder_forward
    # torch.save({"x": encoder_outputs[0], "d_outs": encoder_outputs[1]}, "outputs/inputs_2_am_encoder_out.pt")

    dynamic_axes = {
        "inputs_ling": {1: "seq_len"}, 
        # "inputs_style_embedding": {0: "bs"},
        # "inputs_content_embedding": {0: "bs"}, 
    }

    torch.onnx.export(
        model = am_mdoel,
        args = inputs_dict,
        f = am_encoder_onnx_path, 
        export_params = True,
        opset_version=19,
        verbose = False, 
        input_names = ["inputs_ling", "inputs_speaker", "inputs_style_embedding", "inputs_content_embedding"], 
        output_names = ["x", "d_outs"],
        do_constant_folding=True, 
        dynamic_axes = dynamic_axes, 
    )
    # export_am_decoder:
    # param_dict = torch.load("inputs_2_am_encoder_out.pt", map_location='cpu')
    # for name, param in param_dict.items():
        # print(name, ":",  param.size())

    am_outputs = am_mdoel.decoder_forward(encoder_outputs[0], encoder_outputs[1])
    print("am_decoder_outputs:", am_outputs.size())
    am_mdoel.forward = am_mdoel.decoder_forward

    dynamic_axes = {
        "x": {1: "seq_len"}, 
        "d_outs": {1: "seq_len"}
    }

    torch.onnx.export(
        model = am_mdoel,
        args = encoder_outputs,
        f = am_decoder_onnx_path, 
        export_params = True,
        opset_version=19,
        verbose = False, 
        input_names = ["x", "d_outs"],
        output_names = ["logmel"],
        do_constant_folding=True, 
        dynamic_axes = dynamic_axes, 
    )
    
    return 0


def get_style_embedding(prompt, tokenizer, style_encoder):
    prompt = tokenizer([prompt], return_tensors="pt")
    input_ids = prompt["input_ids"]
    token_type_ids = prompt["token_type_ids"]
    attention_mask = prompt["attention_mask"]
    with torch.no_grad():
        output = style_encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
    # print(output)
    torch.save(prompt, "outputs/embedding_inputs.pt")
    torch.save({"style_embedding": output}, "outputs/embedding_outputs.pt")
    # style_embedding = output["pooled_output"].cpu().squeeze().numpy()
    style_embedding = output.cpu().squeeze().numpy()
    return style_embedding


def export_embedding(style_encoder, tokenizer, embed_onnx_path = "outputs/onnx/embedding.onnx", ):
    # (style_encoder, generator, tokenizer, token2id, speaker2id) = get_models()
    prompt = "你好,明天天气怎么样？"
    outputs = get_style_embedding(prompt=prompt, tokenizer=tokenizer, style_encoder=style_encoder)

    inputs = torch.load("outputs/embedding_inputs.pt", map_location='cpu')

    torch.onnx.export(
        model = style_encoder,
        args = inputs['input_ids'],
        f = embed_onnx_path, 
        export_params = True,
        opset_version=19,
        verbose = False, 
        input_names = ["input_ids", ], 
        output_names = ["style_embedding"],
        do_constant_folding=True, 
        dynamic_axes = {
            'input_ids': {1: 'seq_len'}
        }, 
    )


def export_models():
    models = get_models()
    (style_encoder, generator, tokenizer, token2id, speaker2id) = models
    style_encoder = style_encoder.cpu().eval()
    generator = generator.cpu().eval()
    
    am_mdoel = generator.am
    voc_model = generator.generator    
    am_mdoel = TorchPromptTTS(am_mdoel)
    voc_model = TorchHiFiGAN(voc_model)
    style_encoder = TorchSimBert(style_encoder)

    am_inputs = torch.load("outputs/inputs_2.pt", map_location='cpu')
    voc_inputs = torch.load("outputs/logmel_2.pt", map_location='cpu')

    export_embedding(style_encoder, tokenizer)
    export_am_split_model(am_mdoel, inputs_dict=am_inputs)
    export_voc_model(voc_model, inputs_dict=voc_inputs)



if __name__ == "__main__":
    export_models()

import os
import sys
import numpy as np
import torch
import soundfile as sf

from th_models import get_models, run_frontend
from onnx2trt import get_style_embedding, run_am_decoder, run_am_encoder, run_voc
from utils.profile import timeit

DEVICE = 'cuda:0'
MAX_WAV_VALUE = 32768.0

@timeit
def emotivoice_tts(text, prompt, content, speaker):
    (style_encoder, generator, tokenizer, token2id, speaker2id) = get_models()

    style_embedding = get_style_embedding(prompt, tokenizer)
    content_embedding = get_style_embedding(content, tokenizer)

    speaker = speaker2id[speaker]
    text_int = [token2id[ph] for ph in text.split()]
    sequence = torch.from_numpy(np.array(text_int)).to(DEVICE).long().unsqueeze(0)
    sequence_len = torch.from_numpy(np.array([len(text_int)])).to(DEVICE)
    style_embedding = torch.from_numpy(style_embedding).to(DEVICE).unsqueeze(0)
    content_embedding = torch.from_numpy(
        content_embedding).to(DEVICE).unsqueeze(0)
    speaker = torch.from_numpy(np.array([speaker])).to(DEVICE)

    params = {
        "inputs_ling": sequence.cuda(),
        "inputs_style_embedding": style_embedding.cuda(),
        "inputs_content_embedding": content_embedding.cuda(),
        "inputs_speaker": speaker.cuda()
    }

    # am_out = run_am(inputs=params)
    am_encoder_x, am_encoder_d_outs = run_am_encoder(build=False, inputs=params)
    am_out = run_am_decoder(build=False, inputs={'x': am_encoder_x, 'd_outs': am_encoder_d_outs})
    wav_out = run_voc(inputs={'logmel': am_out})
    audio = wav_out.squeeze() * MAX_WAV_VALUE
    audio = audio.cpu().numpy().astype('int16')

    return audio


def get_style_embedding_th(prompt, tokenizer, style_encoder):
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
    style_embedding = output["pooled_output"].cpu().squeeze().numpy()
    return style_embedding

@timeit
def emotivoice_tts_th(text, prompt, content, speaker):
    (style_encoder, generator, tokenizer, token2id, speaker2id) = get_models()

    style_embedding = get_style_embedding_th(prompt, tokenizer, style_encoder)
    content_embedding = get_style_embedding_th(content, tokenizer, style_encoder)

    speaker = speaker2id[speaker]

    text_int = [token2id[ph] for ph in text.split()]

    sequence = torch.from_numpy(np.array(text_int)).to(
        DEVICE).long().unsqueeze(0)
    sequence_len = torch.from_numpy(np.array([len(text_int)])).to(DEVICE)
    style_embedding = torch.from_numpy(style_embedding).to(DEVICE).unsqueeze(0)
    content_embedding = torch.from_numpy(
        content_embedding).to(DEVICE).unsqueeze(0)
    speaker = torch.from_numpy(np.array([speaker])).to(DEVICE)

    with torch.no_grad():
        infer_output = generator(
            inputs_ling=sequence,
            inputs_style_embedding=style_embedding,
            input_lengths=sequence_len,
            inputs_content_embedding=content_embedding,
            inputs_speaker=speaker,
            alpha=1.0
        )

    audio = infer_output["wav_predictions"].squeeze() * MAX_WAV_VALUE
    audio = audio.cpu().numpy().astype('int16')

    return audio

@timeit
def run():
    inputs = "你好呀,明天天气怎么样?"
    prompt = "兴奋的"
    voice = '8051'
    wav_file = "test.wav"
    # warm up

    text = run_frontend(inputs)
    # np_audio = emotivoice_tts(text, prompt, inputs, voice) # 11 ms
    np_audio = emotivoice_tts_th(text, prompt, inputs, voice) # 85 ms
    sf.write(file=wav_file, data=np_audio, samplerate=16000, format='WAV')
    

if __name__ == "__main__":
    # run_all_model()
    for i in range(10):
        run()

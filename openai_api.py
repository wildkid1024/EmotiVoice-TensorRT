import logging
import os
import sys
import io
import glob
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import soundfile as sf
from pydub import AudioSegment
import pyrubberband as pyrb
import uvicorn
from fastapi import Response
from pydantic import BaseModel
# import fastapi_offline 
# from fastapi_offline import FastAPIOffline as FastAPI

# fastapi_offline.core._STATIC_PATH = Path(__file__).parent / "assets/js"

from fastapi import FastAPI

from th_models import run_frontend, get_models
from onnx2trt import get_style_embedding, run_am_decoder, run_am_encoder, run_voc

LOGGER = logging.getLogger(__name__)

DEVICE = 'cuda:0'
MAX_WAV_VALUE = 32768.0
WAV_SR = 14000


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

    am_encoder_x, am_encoder_d_outs = run_am_encoder(inputs=params)
    am_out = run_am_decoder(inputs={'x': am_encoder_x.cuda(), 'd_outs': am_encoder_d_outs.cuda()})
    wav_out = run_voc(inputs={'logmel': am_out.cuda()})
    audio = wav_out.squeeze() * MAX_WAV_VALUE
    audio = audio.cpu().numpy().astype('int16')
    
    return audio


app = FastAPI()

class SpeechRequest(BaseModel):
    input: str
    voice: str = '8051'
    prompt: Optional[str] = ''
    language: Optional[str] = 'zh_us'
    model: Optional[str] = 'emoti-voice'
    response_format: Optional[str] = 'mp3'
    speed: Optional[float] = 1.0


@app.post("/v1/audio/speech")
def text_to_speech(speechRequest: SpeechRequest):
    text = run_frontend(inputs=speechRequest.input)
    np_audio = emotivoice_tts(text, speechRequest.prompt,
                              speechRequest.input, speechRequest.voice)
    y_stretch = np_audio
    if speechRequest.speed != 1.0:
        y_stretch = pyrb.time_stretch(np_audio, WAV_SR, speechRequest.speed)
    wav_buffer = io.BytesIO()
    sf.write(file=wav_buffer, data=y_stretch,
             samplerate=WAV_SR, format='WAV')
    buffer = wav_buffer
    response_format = speechRequest.response_format
    if response_format != 'wav':
        wav_audio = AudioSegment.from_wav(wav_buffer)
        wav_audio.frame_rate=WAV_SR
        buffer = io.BytesIO()
        wav_audio.export(buffer, format=response_format)

    return Response(content=buffer.getvalue(),
                    media_type=f"audio/{response_format}")


if __name__ == "__main__":
    uvicorn.run(app=app, host="0.0.0.0", port=8051)
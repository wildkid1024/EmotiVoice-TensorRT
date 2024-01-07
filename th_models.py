import os
import sys
import glob
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from yacs import config as CONFIG
from functools import lru_cache, cache

Emotivoice_PATH = os.path.join(os.path.dirname(__file__), "EmotiVoice/")
sys.path.append(Emotivoice_PATH)

from EmotiVoice.frontend import g2p_cn_en, ROOT_DIR, read_lexicon, G2p
from EmotiVoice.models.prompt_tts_modified.jets import JETSGenerator
from EmotiVoice.models.prompt_tts_modified.simbert import StyleEncoder
from EmotiVoice.config.joint.config import Config

from utils.configs import MODEL_PATH

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = Config()

config.bert_path = os.path.join(MODEL_PATH, "WangZeJun/simbert-base-chinese")
config.output_directory = os.path.join(MODEL_PATH, "outputs")


def scan_checkpoint(cp_dir, prefix, c=8):
    pattern = os.path.join(cp_dir, prefix + '?'*c)
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

@lru_cache(maxsize=10)
def get_models():
    am_checkpoint_path = scan_checkpoint(
        f'{config.output_directory}/prompt_tts_open_source_joint/ckpt', 'g_')

    # f'{config.output_directory}/style_encoder/ckpt/checkpoint_163431'
    style_encoder_checkpoint_path = scan_checkpoint(
        f'{config.output_directory}/style_encoder/ckpt', 'checkpoint_', 6)

    with open(config.model_config_path, 'r') as fin:
        conf = CONFIG.load_cfg(fin)

    conf.n_vocab = config.n_symbols
    conf.n_speaker = config.speaker_n_labels

    style_encoder = StyleEncoder(config)
    model_CKPT = torch.load(style_encoder_checkpoint_path, map_location="cpu")
    model_ckpt = {}
    for key, value in model_CKPT['model'].items():
        new_key = key[7:]
        model_ckpt[new_key] = value
    style_encoder.load_state_dict(model_ckpt, strict=False)
    generator = JETSGenerator(conf).to(DEVICE)

    model_CKPT = torch.load(am_checkpoint_path, map_location=DEVICE)
    generator.load_state_dict(model_CKPT['generator'])
    generator.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.bert_path)

    with open(config.token_list_path, 'r') as f:
        token2id = {t.strip(): idx for idx, t, in enumerate(f.readlines())}

    with open(config.speaker2id_path, encoding='utf-8') as f:
        speaker2id = {t.strip(): idx for idx, t in enumerate(f.readlines())}

    return (style_encoder, generator, tokenizer, token2id, speaker2id)


def run_frontend(inputs):
    lexicon = read_lexicon(os.path.join(Emotivoice_PATH, "lexicon/librispeech-lexicon.txt"))
    g2p = G2p()
    text = g2p_cn_en(inputs, g2p, lexicon)
    return text

class TorchPromptTTS(nn.Module):
    def __init__(self, model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = model

    def encoder_forward(self, inputs_ling, inputs_speaker, inputs_style_embedding , inputs_content_embedding, input_lengths=None):
        B = inputs_ling.size(0)
        T = inputs_ling.size(1)

        token_embed = self.model.src_word_emb(inputs_ling)
        x, _ = self.model.encoder(token_embed, None)
        speaker_embedding = self.model.spk_tokenizer(inputs_speaker)
        x = torch.concat([x, speaker_embedding.unsqueeze(1).expand(B, T, -1), 
                          inputs_style_embedding.unsqueeze(1).expand(B, T, -1), 
                          inputs_content_embedding.unsqueeze(1).expand(B, T, -1)], dim=-1)
        
        x = self.model.embed_projection1(x)
        p_outs = self.model.pitch_predictor(x)
        e_outs = self.model.energy_predictor(x)

        log_p_attn, ds, bin_loss, ps, es =None, None, None, None, None
        d_outs = self.model.duration_predictor.inference(x)
        p_embs = self.model.pitch_embed(p_outs.unsqueeze(1)).transpose(1, 2)
        e_embs = self.model.energy_embed(e_outs.unsqueeze(1)).transpose(1, 2)

        x = x + p_embs + e_embs
        return x, d_outs
    

    def decoder_forward(self, x, d_outs):
        x = self.model.length_regulator(x, d_outs, None, None)
        x, _ = self.model.decoder(x, None)
        x = self.model.to_mel(x)
        return x
    

    def forward(self, inputs_ling, inputs_speaker, inputs_style_embedding , inputs_content_embedding):
        x, d_outs = self.encoder_forward(inputs_ling, inputs_speaker, inputs_style_embedding , inputs_content_embedding)
        logmel = self.decoder_forward(x, d_outs)
        return logmel


class TorchHiFiGAN(nn.Module):
    def __init__(self, model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = model

    def forward(self, logmel):
        logmel = logmel.transpose(1, 2)
        return self.model(x=logmel)


class TorchSimBert(nn.Module):
    def __init__(self, model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = model

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.model.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ) # return a dict having ['last_hidden_state', 'pooler_output']

        pooled_output = outputs["pooler_output"]
        return pooled_output

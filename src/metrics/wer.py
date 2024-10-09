from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_wer

from torchaudio.models.decoder import download_pretrained_files
import kenlm

# TODO beam search / LM versions
# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved


class ArgmaxWERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        wers = []
        # predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(log_probs, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length, :])
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)


class BeamSearchWERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        wers = []
        # predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        #print('HAHAHA')
        #print(predictions.shape)
        lengths = log_probs_length.detach().numpy()
        #print(lengths)
        for log_prob_vec, length, target_text in zip(log_probs, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length, :], beam_search=True)
            #print("WTF!!!")
            #print(target_text)
            #print(pred_text)
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)


class LMBeamSearchWERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        lm_files = download_pretrained_files("librispeech-4-gram")
        lm_model = kenlm.Model(lm_files.lm)
        wers = []
        # predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        #print('HAHAHA')
        #print(predictions.shape)
        lengths = log_probs_length.detach().numpy()
        #print(lengths)
        for log_prob_vec, length, target_text in zip(log_probs, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length, :], beam_search=True, lm=True, lm_model=lm_model)
            #print("WTF!!!")
            #print(target_text)
            #print(pred_text)
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)


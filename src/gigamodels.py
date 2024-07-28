#https://github.com/salute-developers/GigaAM


import locale
locale.getpreferredencoding = lambda do_setlocale=True: "UTF-8"

import torch
from typing import List, Tuple, Union
import hydra
import soundfile as sf
from omegaconf import DictConfig, ListConfig, OmegaConf
import torchaudio

from nemo.collections.asr.modules.audio_preprocessing import (
    AudioToMelSpectrogramPreprocessor as NeMoAudioToMelSpectrogramPreprocessor,
)
from nemo.collections.asr.parts.preprocessing.features import (
    FilterbankFeaturesTA as NeMoFilterbankFeaturesTA,
)

class SpecScaler(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x.clamp_(1e-9, 1e9))

class GigaAMEmo(torch.nn.Module):
    def __init__(self, conf: Union[DictConfig, ListConfig]):
        super().__init__()
        self.id2name = conf.id2name
        self.feature_extractor = hydra.utils.instantiate(conf.feature_extractor)
        self.conformer = hydra.utils.instantiate(conf.encoder)
        self.linear_head = hydra.utils.instantiate(conf.classification_head)

    def forward(self, features, features_length=None):
        if features.dim() == 2:
            features = features.unsqueeze(0)
        if not features_length:
            features_length = torch.ones(features.shape[0]) * features.shape[-1]
            features_length = features_length.to(features.device)
        encoded, _ = self.conformer(audio_signal=features, length=features_length)
        encoded_pooled = torch.nn.functional.avg_pool1d(
            encoded, kernel_size=encoded.shape[-1]
        ).squeeze(-1)

        logits = self.linear_head(encoded_pooled)
        return logits

    def get_probs(self, audio_path: str) -> List[List[float]]:
        audio_signal, _ = sf.read(audio_path, dtype="float32")
        features = self.feature_extractor(
            torch.tensor(audio_signal).float().to(next(self.parameters()).device)
        )
        logits = self.forward(features)
        probs = torch.nn.functional.softmax(logits).detach().tolist()
        return probs
    
class FilterbankFeaturesTA(NeMoFilterbankFeaturesTA):
    def __init__(self, mel_scale: str = "htk", wkwargs=None, **kwargs):
        if "window_size" in kwargs:
            del kwargs["window_size"]
        if "window_stride" in kwargs:
            del kwargs["window_stride"]

        super().__init__(**kwargs)

        self._mel_spec_extractor: torchaudio.transforms.MelSpectrogram = (
            torchaudio.transforms.MelSpectrogram(
                sample_rate=self._sample_rate,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_mels=kwargs["nfilt"],
                window_fn=self.torch_windows[kwargs["window"]],
                mel_scale=mel_scale,
                norm=kwargs["mel_norm"],
                n_fft=kwargs["n_fft"],
                f_max=kwargs.get("highfreq", None),
                f_min=kwargs.get("lowfreq", 0),
                wkwargs=wkwargs,
            )
        )


class AudioToMelSpectrogramPreprocessor(NeMoAudioToMelSpectrogramPreprocessor):
    def __init__(self, mel_scale: str = "htk", **kwargs):
        super().__init__(**kwargs)
        kwargs["nfilt"] = kwargs["features"]
        del kwargs["features"]
        self.featurizer = (
            FilterbankFeaturesTA(  # Deprecated arguments; kept for config compatibility
                mel_scale=mel_scale,
                **kwargs,
            )
        )
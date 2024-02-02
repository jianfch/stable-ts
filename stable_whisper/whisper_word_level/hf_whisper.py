from typing import Optional, Union

import numpy as np
import torch

from ..audio import convert_demucs_kwargs, prep_audio
from ..non_whisper import transcribe_any


HF_MODELS = {
    "tiny.en": "openai/whisper-tiny.en",
    "tiny": "openai/whisper-tiny",
    "base.en": "openai/whisper-base.en",
    "base": "openai/whisper-base",
    "small.en": "openai/whisper-small.en",
    "small": "openai/whisper-small",
    "medium.en": "openai/whisper-medium.en",
    "medium": "openai/whisper-medium",
    "large-v1": "openai/whisper-large-v1",
    "large-v2": "openai/whisper-large-v2",
    "large-v3": "openai/whisper-large-v3",
    "large": "openai/whisper-large-v3",
}


def get_device(device: str = None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return 'cuda:0'
    if (mps := getattr(torch.backends, 'mps', None)) is not None:
        return mps.is_available()
    return 'cpu'


def load_hf_pipe(model_name: str, device: str = None, flash: bool = False):
    from transformers import pipeline
    model_kwargs = {'attn_implementation': 'flash_attention_2'} if flash else {'attn_implementation': 'sdpa'}
    pipe = pipeline(
        'automatic-speech-recognition',
        model=HF_MODELS.get(model_name, model_name),
        torch_dtype=torch.float16,
        device=get_device(device),
        model_kwargs=model_kwargs,
    )

    return pipe


class WhisperHF:

    def __init__(self, model_name: str, device: str = None, flash: bool = False):
        self._model_name = model_name
        self._pipe = load_hf_pipe(self._model_name, device, flash=flash)

    @property
    def sampling_rate(self):
        return self._pipe.feature_extractor.sampling_rate

    def _inner_transcribe(
            self,
            audio: Union[str, bytes, np.ndarray],
            language: str = None,
            task: str = None,
            batch_size: int = 24,
            word_timestamps=True,
            verbose: Optional[bool] = False
    ):
        generate_kwargs = {'task': task or 'transcribe', 'language': language}
        if verbose is not None:
            print(f'Transcribing with Hugging Face Whisper ({self._model_name})...')
        result = self._pipe(
            audio,
            chunk_length_s=30,
            batch_size=batch_size,
            generate_kwargs=generate_kwargs,
            return_timestamps='word' if word_timestamps else True,
        )['chunks']
        if verbose is not None:
            print(f'Transcription completed.')
        if word_timestamps:
            words = [
                dict(start=word['timestamp'][0], end=word['timestamp'][1], word=word['text'])
                for word in result
            ]
            return [words]
        segs = [
            dict(start=seg['timestamp'][0], end=seg['timestamp'][1], text=seg['text'])
            for seg in result
        ]
        return segs

    def transcribe(
            self,
            audio: Union[str, bytes, np.ndarray],
            *,
            language: str = None,
            task: str = None,
            batch_size: int = 24,
            word_timestamps: bool = True,
            verbose: Optional[bool] = False,
            regroup: Union[bool, str] = True,
            suppress_silence: bool = True,
            suppress_word_ts: bool = True,
            use_word_position: bool = True,
            q_levels: int = 20,
            k_size: int = 5,
            denoiser: Optional[str] = None,
            denoiser_options: Optional[dict] = None,
            vad: bool = False,
            vad_threshold: float = 0.35,
            vad_onnx: bool = False,
            min_word_dur: Optional[float] = None,
            nonspeech_error: float = 0.1,
            only_voice_freq: bool = False,
            only_ffmpeg: bool = False,
            check_sorted: bool = True,
            **options
    ):
        denoiser, denoiser_options = convert_demucs_kwargs(
            denoiser, denoiser_options,
            demucs=options.pop('demucs', None), demucs_options=options.pop('demucs_options', None)
        )

        if isinstance(audio, (str, bytes)):
            audio = prep_audio(audio, sr=self.sampling_rate).numpy()
            options['input_sr'] = self.sampling_rate

        if 'input_sr' not in options:
            options['input_sr'] = self.sampling_rate

        if denoiser or only_voice_freq:
            if 'audio_type' not in options:
                options['audio_type'] = 'numpy'
            if 'model_sr' not in options:
                options['model_sr'] = self.sampling_rate

        inference_kwargs = dict(
            audio=audio,
            language=language,
            task=task,
            batch_size=batch_size,
            word_timestamps=word_timestamps,
            verbose=verbose
        )
        return transcribe_any(
            inference_func=self._inner_transcribe,
            audio=audio,
            inference_kwargs=inference_kwargs,
            verbose=verbose,
            regroup=regroup,
            suppress_silence=suppress_silence,
            suppress_word_ts=suppress_word_ts,
            q_levels=q_levels,
            k_size=k_size,
            denoiser=denoiser,
            denoiser_options=denoiser_options,
            vad=vad,
            vad_threshold=vad_threshold,
            vad_onnx=vad_onnx,
            min_word_dur=min_word_dur,
            nonspeech_error=nonspeech_error,
            use_word_position=use_word_position,
            only_voice_freq=only_voice_freq,
            only_ffmpeg=only_ffmpeg,
            force_order=True,
            check_sorted=check_sorted,
            **options
        )


def load_hf_whisper(model_name: str, device: str = None, flash: bool = False):
    return WhisperHF(model_name, device, flash=flash)

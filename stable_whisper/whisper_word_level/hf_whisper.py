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
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    device = get_device(device)
    is_cpu = (device if isinstance(device, str) else getattr(device, 'type', None)) == 'cpu'
    dtype = torch.float32 if is_cpu or not torch.cuda.is_available() else torch.float16
    model_id = HF_MODELS.get(model_name, model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        use_flash_attention_2=flash
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    if not flash:
        try:
            model = model.to_bettertransformer()
        except ValueError:
            pass

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        torch_dtype=dtype,
        device=device,
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
            batch_size=batch_size,
            generate_kwargs=generate_kwargs,
            return_timestamps='word' if word_timestamps else True,
        )['chunks']
        if verbose is not None:
            print(f'Transcription completed.')

        def replace_none_ts(parts):
            total_dur = round(audio.shape[-1] / self.sampling_rate, 3) if isinstance(audio, np.ndarray) else None
            _medium_dur = _ts_nonzero_mask = None

            def ts_nonzero_mask() -> np.ndarray:
                nonlocal _ts_nonzero_mask
                if _ts_nonzero_mask is None:
                    _ts_nonzero_mask = np.array([(p['end'] or p['start']) is not None for p in parts])
                return _ts_nonzero_mask

            def medium_dur() -> float:
                nonlocal _medium_dur
                if _medium_dur is None:
                    nonzero_dus = [p['end'] - p['start'] for p in parts if None not in (p['end'], p['start'])]
                    nonzero_durs = np.array(nonzero_dus)
                    _medium_dur = np.median(nonzero_durs) * 2 if len(nonzero_durs) else 2.0
                return _medium_dur

            def _curr_max_end(start: float, next_idx: float) -> float:
                max_end = total_dur
                if next_idx != len(parts):
                    mask = np.flatnonzero(ts_nonzero_mask()[next_idx:])
                    if len(mask):
                        _part = parts[mask[0]+next_idx]
                        max_end = _part['start'] or _part['end']

                new_end = round(start + medium_dur(), 3)
                if max_end is None:
                    return new_end
                if new_end > max_end:
                    return max_end
                return new_end

            for i, part in enumerate(parts, 1):
                if part['start'] is None:
                    is_first = i == 1
                    if is_first:
                        new_start = round((part['end'] or 0) - medium_dur(), 3)
                        part['start'] = max(new_start, 0.0)
                    else:
                        part['start'] = parts[i - 2]['end']
                if part['end'] is None:
                    no_next_start = i == len(parts) or parts[i]['start'] is None
                    part['end'] = _curr_max_end(part['start'], i) if no_next_start else parts[i]['start']

        if word_timestamps:
            words = [
                dict(start=word['timestamp'][0], end=word['timestamp'][1], word=word['text'])
                for word in result
            ]
            replace_none_ts(words)
            return [words]
        segs = [
            dict(start=seg['timestamp'][0], end=seg['timestamp'][1], text=seg['text'])
            for seg in result
        ]
        replace_none_ts(segs)
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

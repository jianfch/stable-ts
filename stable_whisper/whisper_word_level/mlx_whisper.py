from types import MethodType
from typing import Optional, Union

from functools import lru_cache

import numpy as np
import mlx.core as mx

from ..audio import convert_demucs_kwargs, prep_audio
from ..non_whisper import transcribe_any
from ..utils import isolate_useful_options

from ..alignment import align, align_words, refine

MLX_MODELS = {
    "tiny.en": "mlx-community/whisper-tiny.en-mlx",
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base.en": "mlx-community/whisper-base.en-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small.en": "mlx-community/whisper-small.en-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium.en": "mlx-community/whisper-medium.en-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v1": "mlx-community/whisper-large-v1-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo": "mlx-community/whisper-large-v3-turbo"
}

MLX_TO_WHISPER_MAPPING = {
    "encoder.blocks.0.mlp1": "encoder.blocks.0.mlp.0",
    "encoder.blocks.0.mlp2": "encoder.blocks.0.mlp.2",
    "encoder.blocks.1.mlp1": "encoder.blocks.1.mlp.0",
    "encoder.blocks.1.mlp2": "encoder.blocks.1.mlp.2",
    "encoder.blocks.2.mlp1": "encoder.blocks.2.mlp.0",
    "encoder.blocks.2.mlp2": "encoder.blocks.2.mlp.2",
    "encoder.blocks.3.mlp1": "encoder.blocks.3.mlp.0",
    "encoder.blocks.3.mlp2": "encoder.blocks.3.mlp.2",
    "encoder.blocks.4.mlp1": "encoder.blocks.4.mlp.0",
    "encoder.blocks.4.mlp2": "encoder.blocks.4.mlp.2",
    "encoder.blocks.5.mlp1": "encoder.blocks.5.mlp.0",
    "encoder.blocks.5.mlp2": "encoder.blocks.5.mlp.2",
    "decoder.blocks.0.mlp1": "decoder.blocks.0.mlp.0",
    "decoder.blocks.0.mlp2": "decoder.blocks.0.mlp.2",
    "decoder.blocks.1.mlp1": "decoder.blocks.1.mlp.0",
    "decoder.blocks.1.mlp2": "decoder.blocks.1.mlp.2",
    "decoder.blocks.2.mlp1": "decoder.blocks.2.mlp.0",
    "decoder.blocks.2.mlp2": "decoder.blocks.2.mlp.2",
    "decoder.blocks.3.mlp1": "decoder.blocks.3.mlp.0",
    "decoder.blocks.3.mlp2": "decoder.blocks.3.mlp.2",
    "decoder.blocks.4.mlp1": "decoder.blocks.4.mlp.0",
    "decoder.blocks.4.mlp2": "decoder.blocks.4.mlp.2",
    "decoder.blocks.5.mlp1": "decoder.blocks.5.mlp.0",
    "decoder.blocks.5.mlp2": "decoder.blocks.5.mlp.2",
}


@lru_cache(maxsize=4)
def load_mlx_model(model_name: str, dtype=None):
    from mlx_whisper import load_models

    model_id = MLX_MODELS.get(model_name, model_name)
    if dtype is None:
        dtype = mx.float32

    return load_models.load_model(model_id, dtype=dtype)


class WhisperMLX:
    def __init__(self, model=None, model_name=None):
        self._model = model
        self._model_name = model_name
        self._sampling_rate = None

    @property
    def sampling_rate(self):
        if self._sampling_rate is None:
            from mlx_whisper.audio import SAMPLE_RATE
            self._sampling_rate = SAMPLE_RATE
        return self._sampling_rate

    @property
    def model_name(self):
        if hasattr(self._model, 'name_or_path'):
            return self._model.name_or_path
        return self._model_name

    def _process_segments(self, segments, audio=None, word_timestamps=True):
        processed_segments = []

        for seg in segments:
            segment_dict = {
                'start': seg.get('start'),
                'end': seg.get('end'),
                'text': seg.get('text', ''),
            }

            if word_timestamps and 'words' in seg:
                segment_dict['words'] = [
                    {
                        'word': word.get('word', ''),
                        'start': word.get('start'),
                        'end': word.get('end'),
                        'probability': word.get('probability', 1.0)
                    }
                    for word in seg.get('words', [])
                ]

            processed_segments.append(segment_dict)

        self._fix_timestamps(processed_segments, audio)
        return processed_segments

    def _fix_timestamps(self, parts, audio=None):
        total_dur = round(audio.shape[-1] / self.sampling_rate, 3) if isinstance(audio, np.ndarray) else None
        _medium_dur = None
        _ts_nonzero_mask = None

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
                    _part = parts[mask[0] + next_idx]
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

    def _inner_transcribe(
            self,
            audio: Union[str, bytes, np.ndarray],
            language: Optional[str] = None,
            task: Optional[str] = 'transcribe',
            word_timestamps: bool = True,
            verbose: Optional[bool] = False,
            **kwargs
    ):
        generate_kwargs = {}

        model_path = self._model_name

        if model_path and (model_path.endswith('.en') or model_path.endswith('.en-mlx')):
            language = task = None

        if task is not None:
            generate_kwargs['task'] = task
        if language is not None:
            generate_kwargs['language'] = language

        generate_kwargs.update(kwargs)

        if verbose:
            print(f'Transcribing with MLX Whisper ({model_path})...')

        if isinstance(audio, np.ndarray):
            audio_mx = mx.array(audio)
        else:
            audio_mx = audio

        from mlx_whisper import transcribe

        # Use the proper model path or ID for transcription
        model_path_or_repo = model_path
        if model_path in MLX_MODELS:
            model_path_or_repo = MLX_MODELS[model_path]

        transcribe_kwargs = {
            'path_or_hf_repo': model_path_or_repo,
            'word_timestamps': word_timestamps,
            'verbose': verbose
        }

        if language is not None:
            transcribe_kwargs['language'] = language
        if task is not None:
            transcribe_kwargs['task'] = task

        for key, value in generate_kwargs.items():
            if key not in transcribe_kwargs:
                transcribe_kwargs[key] = value

        try:
            output = transcribe(audio_mx, **transcribe_kwargs)
        except Exception as e:
            if verbose:
                print(f"Error during transcription: {str(e)}")
            raise

        is_english = model_path and (model_path.endswith('.en') or model_path.endswith('.en-mlx'))
        detected_language = output.get('language', 'en' if is_english else None)

        if verbose:
            print(f'Transcription completed.')

        segments = self._process_segments(
            output.get('segments', []),
            audio=audio,
            word_timestamps=word_timestamps
        )

        return {'segments': segments, 'language': detected_language}

    def transcribe(
            self,
            audio: Union[str, bytes, np.ndarray],
            *,
            language: Optional[str] = None,
            task: Optional[str] = 'transcribe',
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
            vad: Union[bool, dict] = False,
            vad_threshold: float = 0.35,
            vad_onnx: bool = False,
            min_word_dur: Optional[float] = None,
            nonspeech_error: float = 0.1,
            only_voice_freq: bool = False,
            only_ffmpeg: bool = False,
            check_sorted: bool = True,
            **options
    ):
        transcribe_any_options = isolate_useful_options(options, transcribe_any, pop=True)

        denoiser, denoiser_options = convert_demucs_kwargs(
            denoiser, denoiser_options,
            demucs=transcribe_any_options.pop('demucs', None),
            demucs_options=transcribe_any_options.pop('demucs_options', None)
        )

        if isinstance(audio, (str, bytes)):
            audio = prep_audio(audio, sr=self.sampling_rate).numpy()
            transcribe_any_options['input_sr'] = self.sampling_rate

        if 'input_sr' not in transcribe_any_options:
            transcribe_any_options['input_sr'] = self.sampling_rate

        if denoiser or only_voice_freq:
            if 'audio_type' not in transcribe_any_options:
                transcribe_any_options['audio_type'] = 'numpy'
            if 'model_sr' not in transcribe_any_options:
                transcribe_any_options['model_sr'] = self.sampling_rate

        inference_kwargs = {
            'audio': audio,
            'language': language,
            'task': task,
            'word_timestamps': word_timestamps,
            'verbose': verbose,
            **options
        }

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
            **transcribe_any_options
        )


def load_mlx_whisper(model_name: str, dtype=None, **model_kwargs):
    if dtype is None:
        dtype = mx.float32

    # Load the base model
    model = load_mlx_model(model_name, dtype)

    # Create the wrapper
    wrapper = WhisperMLX(model=model, model_name=model_name)

    # Attach methods to the model
    model.transcribe = wrapper.transcribe
    model.sampling_rate = wrapper.sampling_rate
    model.align = MethodType(align, model)
    model.align_words = MethodType(align_words, model)
    model.refine = MethodType(refine, model)

    return model
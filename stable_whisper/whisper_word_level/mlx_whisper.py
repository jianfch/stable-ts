from typing import Optional, Union

import numpy as np
import mlx.core as mx
import torch

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

# Define MLX to Whisper mapping for parameter conversion
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
    "encoder.layers": "encoder.blocks",
    "decoder.layers": "decoder.blocks",
    "final_layer_norm": "mlp_ln",
    "self_attn.q_proj": ".attn.query",
    "self_attn.k_proj": ".attn.key",
    "self_attn.v_proj": ".attn.value",
    "self_attn_layer_norm": ".attn_ln",
    "self_attn.out_proj": ".attn.out",
    "encoder_attn.q_proj": ".cross_attn.query",
    "encoder_attn.k_proj": ".cross_attn.key",
    "encoder_attn.v_proj": ".cross_attn.value",
    "encoder_attn_layer_norm": ".cross_attn_ln",
    "encoder_attn.out_proj": ".cross_attn.out",
    "decoder.layer_norm.": "decoder.ln.",
    "encoder.layer_norm.": "encoder.ln.",
    "embed_tokens": "token_embedding",
    "layer_norm": "ln_post",
}


def load_mlx_model(model_name: str, dtype=None, **model_kwargs):
    from mlx_whisper import load_models
    import mlx.core as mx

    model_id = MLX_MODELS.get(model_name, model_name)
    if dtype is None:
        dtype = mx.float32

    model = load_models.load_model(model_id, dtype=dtype)

    return model


class WhisperMLX:

    def __init__(self, model_name: str, dtype=None, **model_kwargs):
        self._model_name = model_name
        self._model = load_mlx_model(self._model_name, dtype=dtype, **model_kwargs)
        self._model_name = getattr(self._model, 'name_or_path', self._model_name)
        self._vanilla_model = None

    @property
    def sampling_rate(self):
        from mlx_whisper.audio import SAMPLE_RATE
        return SAMPLE_RATE

    @property
    def model_name(self):
        if self._model_name is None:
            return getattr(self._model, 'name_or_path', 'n/a')
        return self._model_name

    def _inner_transcribe(
            self,
            audio: Union[str, bytes, np.ndarray],
            language: Optional[str] = None,
            task: Optional[str] = 'transcribe',
            word_timestamps=True,
            verbose: Optional[bool] = False,
            **kwargs
    ):
        generate_kwargs = {}
        if self.model_name.endswith('en'):
            language = task = None
        if task is not None:
            generate_kwargs['task'] = task
        if language is not None:
            generate_kwargs['language'] = language
        generate_kwargs.update(kwargs)

        if verbose is not None:
            print(f'Transcribing with MLX Whisper ({self.model_name})...')

        # Convert audio to MLX array if needed
        if isinstance(audio, np.ndarray):
            audio_mx = mx.array(audio)
        else:
            audio_mx = audio

        # Process with MLX Whisper
        from mlx_whisper import transcribe

        # Prepare kwargs for transcribe
        transcribe_kwargs = {
            'path_or_hf_repo': MLX_MODELS.get(self._model_name, self._model_name),
            'word_timestamps': word_timestamps,
            'verbose': verbose
        }

        # Add language and task only if they're not None
        if language is not None:
            transcribe_kwargs['language'] = language
        if task is not None:
            transcribe_kwargs['task'] = task

        # Add any other kwargs, but don't override existing ones
        for key, value in generate_kwargs.items():
            if key not in transcribe_kwargs:
                transcribe_kwargs[key] = value

        output = transcribe(audio_mx, **transcribe_kwargs)

        # Process the output to match the expected structure
        detected_language = output.get('language', 'en' if self.model_name.endswith('en') else None)

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

        # Process segments with word timestamps if needed
        if 'segments' in output:
            segs = []
            for seg in output.get('segments', []):
                # Process segment
                segment_dict = {
                    'start': seg.get('start'),
                    'end': seg.get('end'),
                    'text': seg.get('text', ''),
                }

                # Process words if available
                if word_timestamps and 'words' in seg:
                    words = []
                    for word in seg.get('words', []):
                        words.append({
                            'word': word.get('word', ''),
                            'start': word.get('start'),
                            'end': word.get('end'),
                            'probability': word.get('probability', 1.0)
                        })
                    segment_dict['words'] = words

                segs.append(segment_dict)

            replace_none_ts(segs)
        else:
            # Fallback in case there are no segments
            segs = []

        return dict(segments=segs, language=detected_language)

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

        inference_kwargs = dict(
            audio=audio,
            language=language,
            task=task,
            word_timestamps=word_timestamps,
            verbose=verbose,
            **options
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
            **transcribe_any_options
        )

    def as_vanilla_model(self):
        """
        Return a vanilla Whisper model instance with current weights.

        The new instance is only loaded once. Most weights share the same memory as this MLX model instance.
        """
        if self._vanilla_model is not None:
            return self._vanilla_model

        from ..whisper_compatibility import ModelDimensions, Whisper, ln_to_fp32

        # Get the MLX model state dict and convert to numpy arrays
        state_dict = {}
        # In MLX, parameters() is a method that returns a dictionary
        # We need to use tree_flatten to get flattened (key, value) pairs
        from mlx.utils import tree_flatten
        for param_name, param in tree_flatten(self._model.parameters()):
            # Convert MLX array to numpy array
            param_np = np.array(param.astype(mx.float32))
            state_dict[param_name] = param_np

        # Apply MLX to Whisper mapping to convert parameter names
        whisper_state_dict = {}
        for key, value in state_dict.items():
            new_key = key

            # Direct mapping for known keys
            if key in MLX_TO_WHISPER_MAPPING:
                new_key = MLX_TO_WHISPER_MAPPING[key]
                whisper_state_dict[new_key] = value
                continue

            # Handle special case for conv weights that need transpose
            if 'conv1.weight' in key or 'conv2.weight' in key:
                # Transpose the convolutional weights to match the expected shape
                # MLX shape is [out_channels, kernel_size, in_channels]
                # PyTorch shape is [out_channels, in_channels, kernel_size]
                value = np.transpose(value, (0, 2, 1))

            # Handle pattern replacements
            for mlx_pattern, whisper_pattern in MLX_TO_WHISPER_MAPPING.items():
                if mlx_pattern in key:
                    new_key = new_key.replace(mlx_pattern, whisper_pattern)

            whisper_state_dict[new_key] = value

        # Get model configuration from MLX model
        config = self._model.dims

        # Create ModelDimensions object for vanilla Whisper
        dims = ModelDimensions(
            n_mels=config.n_mels,
            n_audio_ctx=config.n_audio_ctx,
            n_audio_state=config.n_audio_state,
            n_audio_head=config.n_audio_head,
            n_audio_layer=config.n_audio_layer,
            n_vocab=config.n_vocab,
            n_text_ctx=config.n_text_ctx,
            n_text_state=config.n_text_state,
            n_text_head=config.n_text_head,
            n_text_layer=config.n_text_layer
        )

        # Create vanilla Whisper model with the extracted dimensions
        new_model = Whisper(dims)

        # Convert numpy arrays to PyTorch tensors
        torch_state_dict = {}
        for name, array in whisper_state_dict.items():
            torch_state_dict[name] = torch.from_numpy(array)

        # Handle alignment heads if available
        alignment_heads = getattr(self._model, 'alignment_heads', None)
        if alignment_heads is not None:
            # Convert to tensor and handle appropriately
            # For MLX models, alignment_heads might be in a different format
            try:
                alignment_heads_tensor = torch.as_tensor(np.array(alignment_heads)).T
                final_heads = torch.zeros(new_model.dims.n_text_layer, new_model.dims.n_text_head, dtype=torch.bool)
                final_heads[alignment_heads_tensor[0], alignment_heads_tensor[1]] = True
                new_model.register_buffer("alignment_heads", final_heads.to_sparse(), persistent=False)
            except Exception as e:
                print(f"Could not process alignment heads: {e}")
                setattr(new_model, 'missing_alignment_heads', True)
        else:
            setattr(new_model, 'missing_alignment_heads', True)

        # Load state dictionary into the vanilla model
        try:
            # First try loading without strict mode to avoid missing key errors
            new_model.load_state_dict(torch_state_dict, strict=False)
        except Exception as e:
            # Fallback: Try loading weights one by one
            for name, param in new_model.named_parameters():
                if name in torch_state_dict:
                    try:
                        # Check if shapes match
                        if param.shape == torch_state_dict[name].shape:
                            param.data.copy_(torch_state_dict[name])
                    except Exception:
                        pass

        # Apply float32 to layer norm parameters
        ln_to_fp32(new_model)

        # Try to modify model if needed
        try:
            from .original_whisper import modify_model
            modify_model(new_model)
        except (ImportError, AttributeError):
            # The modification might not be available or necessary for MLX models
            pass

        # Save the vanilla model to avoid reloading
        self._vanilla_model = new_model
        return new_model

    align = align
    align_words = align_words
    refine = refine


def load_mlx_whisper(model_name: str, dtype=None, **model_kwargs):
    return WhisperMLX(model_name, dtype=dtype, **model_kwargs)
from typing import Optional, Union

import numpy as np
import torch

from ..audio import convert_demucs_kwargs, prep_audio
from ..non_whisper import transcribe_any
from ..utils import isolate_useful_options

from ..alignment import align, align_words, refine


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
    "large-v3-turbo": "openai/whisper-large-v3-turbo",
    "turbo": "openai/whisper-large-v3-turbo"
}

WHISPER_TO_HF_MAPPING = {
    "blocks": "layers",
    "mlp.0": "fc1",
    "mlp.2": "fc2",
    "mlp_ln": "final_layer_norm",
    ".attn.query": ".self_attn.q_proj",
    ".attn.key": ".self_attn.k_proj",
    ".attn.value": ".self_attn.v_proj",
    ".attn_ln": ".self_attn_layer_norm",
    ".attn.out": ".self_attn.out_proj",
    ".cross_attn.query": ".encoder_attn.q_proj",
    ".cross_attn.key": ".encoder_attn.k_proj",
    ".cross_attn.value": ".encoder_attn.v_proj",
    ".cross_attn_ln": ".encoder_attn_layer_norm",
    ".cross_attn.out": ".encoder_attn.out_proj",
    "decoder.ln.": "decoder.layer_norm.",
    "encoder.ln.": "encoder.layer_norm.",
    "token_embedding": "embed_tokens",
    "encoder.positional_embedding": "encoder.embed_positions.weight",
    "decoder.positional_embedding": "decoder.embed_positions.weight",
    "ln_post": "layer_norm",
}


def get_device(device: str = None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return 'cuda:0'
    if (mps := getattr(torch.backends, 'mps', None)) is not None and mps.is_available():
        return 'mps'
    return 'cpu'


def load_hf_pipe(model_name: str, device: str = None, flash: bool = False, **pipeline_kwargs):
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    from transformers.configuration_utils import PretrainedConfig
    device = get_device(device)
    is_cpu = (device if isinstance(device, str) else getattr(device, 'type', None)) == 'cpu'
    dtype = torch.float32 if is_cpu or not torch.cuda.is_available() else torch.float16
    model_id = HF_MODELS.get(model_name, model_name)
    
    if flash:
        config = PretrainedConfig(
            attn_implementation="flash_attention_2",
        )
    else:
        config = None
        
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        config=config
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    final_pipe_kwargs = dict(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        # chunk_length_s=30,
        torch_dtype=dtype,
        device=device,
        return_language=True
    )
    final_pipe_kwargs.update(**pipeline_kwargs)
    pipe = pipeline(**final_pipe_kwargs)

    return pipe


class WhisperHF:

    def __init__(self, model_name: str, device: str = None, flash: bool = False, pipeline=None, **pipeline_kwargs):
        self._model_name = model_name
        pipeline_kwargs['return_language'] = True
        self._pipe = load_hf_pipe(self._model_name, device, flash=flash, **pipeline_kwargs) if pipeline is None \
            else pipeline
        self._model_name = getattr(self._pipe.model, 'name_or_path', self._model_name)
        self._vanilla_model = None

    @property
    def sampling_rate(self):
        return self._pipe.feature_extractor.sampling_rate

    @property
    def model_name(self):
        if self._model_name is None:
            return getattr(self._pipe.model, 'name_or_path', 'n/a')
        return self._model_name

    def _inner_transcribe(
            self,
            audio: Union[str, bytes, np.ndarray],
            language: Optional[str] = None,
            task: Optional[str] = 'transcribe',
            batch_size: Optional[int] = 24,
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
            print(f'Transcribing with Hugging Face Whisper ({self.model_name})...')
        pipe_kwargs = dict(
            generate_kwargs=generate_kwargs,
            return_timestamps='word' if word_timestamps else True,
            return_language=True
        )
        if batch_size is not None:
            pipe_kwargs['batch_size'] = batch_size
        output = self._pipe(audio, **pipe_kwargs)
        result = output['chunks']
        if not language and not self._pipe.model.generation_config.is_multilingual:
            language = 'en'
        if not language and result and 'language' in result[0]:
            language = result[0]['language']
        if not language and hasattr(output, 'get') and 'detected_language' in output:
            language = output['detected_language']
        if not language:
            # HF Pipelines have broken language detection.
            # Manually detect language by generating tokens from the first 10 seconds of the audio.
            try:
                import torch
                sample_audio = audio[:int(self.sampling_rate * 10)]  # Use first 10 seconds
                inputs = self._pipe.feature_extractor(sample_audio, sampling_rate=self.sampling_rate, return_tensors="pt")
                
                # Ensure input features match model dtype and device
                model_dtype = next(self._pipe.model.parameters()).dtype
                model_device = next(self._pipe.model.parameters()).device
                inputs.input_features = inputs.input_features.to(dtype=model_dtype, device=model_device)
                
                # Generate with minimal tokens to detect language
                with torch.no_grad():
                    generated_ids = self._pipe.model.generate(
                        inputs.input_features,
                        max_new_tokens=10,
                        do_sample=False,
                        output_scores=True,
                        return_dict_in_generate=True
                    )
                
                # Decode the tokens to extract language information
                tokens = self._pipe.tokenizer.batch_decode(generated_ids.sequences, skip_special_tokens=False)[0]
                
                # Extract language token (format: <|en|>, <|fr|>, etc.)
                import re
                lang_match = re.search(r'<\|(\w{2})\|>', tokens)
                if lang_match:
                    language = lang_match.group(1)
                else:
                    language = None
                    
            except Exception as e:
                print(f'Error detecting language: {e}')
                language = None
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
            if words:
                segs = [
                        dict(
                            start=words[0]['start'],
                            end=words[-1]['end'],
                            text=''.join(w['word'] for w in words),
                            words=words
                        )
                ]
            else:
                segs = []
        else:
            segs = [
                dict(start=seg['timestamp'][0], end=seg['timestamp'][1], text=seg['text'])
                for seg in result
            ]
            replace_none_ts(segs)
        return dict(segments=segs, language=language)

    def transcribe(
            self,
            audio: Union[str, bytes, np.ndarray],
            *,
            language: Optional[str] = None,
            task: Optional[str] = 'transcribe',
            batch_size: Optional[int] = 24,
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
            batch_size=batch_size,
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

        The new instance is only loaded once. Most weights share the same memory as this Hugging Face model instance.
        """
        if self._vanilla_model is not None:
            return self._vanilla_model

        from ..whisper_compatibility import ModelDimensions, Whisper, ln_to_fp32
        from .original_whisper import modify_model
        try:
            from transformers.models.whisper.convert_openai_to_hf import WHISPER_MAPPING
            whisper2hf_mapping = WHISPER_MAPPING
        except (ImportError, ModuleNotFoundError):
            whisper2hf_mapping = WHISPER_TO_HF_MAPPING

        hf_mapping = {v: k for k, v in whisper2hf_mapping.items()}
        assert len(whisper2hf_mapping) == len(hf_mapping)

        state_dict = self._pipe.model.model.state_dict()
        config = self._pipe.model.config

        if 'encoder.layer_norm.' in hf_mapping:
            hf_mapping['encoder.layer_norm.'] = 'encoder.ln_post.'
        for key in list(state_dict.keys()):
            new_key = key
            for k, v in hf_mapping.items():
                if k in key:
                    new_key = new_key.replace(k, v)
            if new_key != key:
                state_dict[new_key] = state_dict.pop(key)

        dims = ModelDimensions(
            n_mels=config.num_mel_bins,
            n_audio_ctx=config.max_source_positions,
            n_audio_state=config.d_model,
            n_audio_head=config.encoder_attention_heads,
            n_audio_layer=config.encoder_layers,
            n_vocab=config.vocab_size,
            n_text_ctx=config.max_target_positions,
            n_text_state=self._pipe.model.model.decoder.embed_positions.embedding_dim,
            n_text_head=config.decoder_attention_heads,
            n_text_layer=config.decoder_layers
        )
        new_model = Whisper(dims)
        if alignment_heads := getattr(self._pipe.model.generation_config, 'alignment_heads', None):
            alignment_heads = torch.as_tensor(alignment_heads).T
            final_heads = torch.zeros(new_model.dims.n_text_layer, new_model.dims.n_text_head, dtype=torch.bool)
            final_heads[alignment_heads[0], alignment_heads[1]] = True
            new_model.register_buffer("alignment_heads", final_heads.to_sparse(), persistent=False)
        else:
            setattr(new_model, 'missing_alignment_heads', True)
        try:
            new_model.load_state_dict(state_dict, strict=True, assign=True)
        except TypeError:
            new_model.load_state_dict(state_dict, strict=True)
        new_model.to(device=self._pipe.model.device)
        ln_to_fp32(new_model)
        modify_model(new_model)
        self._vanilla_model = new_model
        return self._vanilla_model

    align = align
    align_words = align_words
    refine = refine


def load_hf_whisper(model_name: str, device: str = None, flash: bool = False, pipeline=None, **pipeline_kwargs):
    return WhisperHF(model_name, device, flash=flash, pipeline=pipeline, **pipeline_kwargs)

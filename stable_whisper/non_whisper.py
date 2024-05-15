import os
import warnings
import io
import torch
import torchaudio
import numpy as np
from typing import Union, Callable, Optional

from .audio import AudioLoader, get_denoiser_func, convert_demucs_kwargs
from .audio.utils import load_source, load_audio, voice_freq_filter, get_samplerate, resample, audio_to_tensor_resample
from .result import WhisperResult
from .utils import update_options

AUDIO_TYPES = ('str', 'byte', 'torch', 'numpy')

AUDIO_TYPE_BY_CLASS = {
    str: 'str',
    bytes: 'bytes',
    np.ndarray: 'numpy',
    torch.Tensor: 'pytorch',
    AudioLoader: None
}


def transcribe_any(
        inference_func: Callable,
        audio: Union[str, np.ndarray, torch.Tensor, bytes, AudioLoader],
        audio_type: str = None,
        input_sr: int = None,
        model_sr: int = None,
        inference_kwargs: dict = None,
        temp_file: str = None,
        verbose: Optional[bool] = False,
        regroup: Union[bool, str] = True,
        suppress_silence: bool = True,
        suppress_word_ts: bool = True,
        q_levels: int = 20,
        k_size: int = 5,
        denoiser: Optional[str] = None,
        denoiser_options: Optional[dict] = None,
        demucs: bool = False,
        demucs_options: dict = None,
        vad: bool = False,
        vad_threshold: float = 0.35,
        vad_onnx: bool = False,
        min_word_dur: Optional[float] = None,
        min_silence_dur: Optional[float] = None,
        nonspeech_error: float = 0.1,
        use_word_position: bool = True,
        only_voice_freq: bool = False,
        only_ffmpeg: bool = False,
        force_order: bool = False,
        check_sorted: bool = True
) -> WhisperResult:
    """
    Transcribe ``audio`` using any ASR system.

    Parameters
    ----------
    inference_func : Callable
        Function that runs ASR when provided the ``audio`` and return data in the appropriate format.
        For format examples see, https://github.com/jianfch/stable-ts/blob/main/examples/non-whisper.ipynb.
    audio : str or numpy.ndarray or torch.Tensor or bytes or AudioLoader
        Path/URL to the audio file, the audio waveform, bytes of audio file or
        instance of :class:`stable_whisper.audio.AudioLoader`.
    audio_type : {'str', 'byte', 'torch', 'numpy', None}, default None, meaning same type as ``audio``
        The type that ``audio`` needs to be for ``inference_func``.
        'str' is a path to the file.
        'byte' is bytes (used for APIs or to avoid writing any data to hard drive).
        'torch' is an instance of :class:`torch.Tensor` containing the audio waveform, in float32 dtype, on CPU.
        'numpy' is an instance of :class:`numpy.ndarray` containing the audio waveform, in float32 dtype.
    input_sr : int, default None, meaning auto-detected if ``audio`` is ``str`` or ``bytes``
        The sample rate of ``audio``.
    model_sr : int, default None, meaning same sample rate as ``input_sr``
        The sample rate to resample the audio into for ``inference_func``.
    inference_kwargs : dict, optional
        Dictionary of arguments to pass into ``inference_func``.
    temp_file : str, default './_temp_stable-ts_audio_.wav'
        Temporary path for the preprocessed audio when ``audio_type = 'str'``.
    verbose: bool, False
        Whether to displays all the details during transcription, If ``False``, displays progressbar. If ``None``, does
        not display anything.
    regroup: str or bool, default True
         String representation of a custom regrouping algorithm or ``True`` use to the default algorithm 'da'. Only
         applies if ``word_timestamps = False``.
    suppress_silence : bool, default True
        Whether to enable timestamps adjustments based on the detected silence.
    suppress_word_ts : bool, default True
        Whether to adjust word timestamps based on the detected silence. Only enabled if ``suppress_silence = True``.
    q_levels : int, default 20
        Quantization levels for generating timestamp suppression mask; ignored if ``vad = true``.
        Acts as a threshold to marking sound as silent.
        Fewer levels will increase the threshold of volume at which to mark a sound as silent.
    k_size : int, default 5
        Kernel size for avg-pooling waveform to generate timestamp suppression mask; ignored if ``vad = true``.
        Recommend 5 or 3; higher sizes will reduce detection of silence.
    denoiser : str, optional
        String of the denoiser to use for preprocessing ``audio``.
        See ``stable_whisper.audio.SUPPORTED_DENOISERS`` for supported denoisers.
    denoiser_options : dict, optional
        Options to use for ``denoiser``.
    vad : bool, default False
        Whether to use Silero VAD to generate timestamp suppression mask.
        Silero VAD requires PyTorch 1.12.0+. Official repo, https://github.com/snakers4/silero-vad.
    vad_threshold : float, default 0.35
        Threshold for detecting speech with Silero VAD. Low threshold reduces false positives for silence detection.
    vad_onnx : bool, default False
        Whether to use ONNX for Silero VAD.
    min_word_dur : float or None, default None meaning use ``stable_whisper.default.DEFAULT_VALUES``
        Shortest duration each word is allowed to reach for silence suppression.
    min_silence_dur : float, optional
        Shortest duration of silence allowed for silence suppression.
    nonspeech_error : float, default 0.1
        Relative error of non-speech sections that appear in between a word for silence suppression.
    use_word_position : bool, default True
        Whether to use position of the word in its segment to determine whether to keep end or start timestamps if
        adjustments are required. If it is the first word, keep end. Else if it is the last word, keep the start.
    only_voice_freq : bool, default False
        Whether to only use sound between 200 - 5000 Hz, where majority of human speech are.
    only_ffmpeg : bool, default False
        Whether to use only FFmpeg (instead of not yt-dlp) for URls
    force_order : bool, default False
        Whether to use adjacent timestamps to replace timestamps that are out of order. Use this parameter only if
        the words/segments returned by ``inference_func`` are expected to be in chronological order.
    check_sorted : bool, default True
        Whether to raise an error when timestamps returned by ``inference_func`` are not in ascending order.

    Returns
    -------
    stable_whisper.result.WhisperResult
        All timestamps, words, probabilities, and other data from the transcription of ``audio``.

    Notes
    -----
    For ``audio_type = 'str'``:
        If ``audio`` is a file and no audio preprocessing is set, ``audio`` will be directly passed into
            ``inference_func``.
        If audio preprocessing is ``denoiser`` or ``only_voice_freq``, the processed audio will be encoded into
            ``temp_file`` and then passed into ``inference_func``.

    For ``audio_type = 'byte'``:
        If ``audio`` is file, the bytes of file will be passed into ``inference_func``.
        If ``audio`` is :class:`torch.Tensor` or :class:`numpy.ndarray`, the bytes of the ``audio`` will be encoded
            into WAV format then passed into ``inference_func``.

    Resampling is only performed on ``audio`` when ``model_sr`` does not match the sample rate of the ``audio`` before
        passing into ``inference_func`` due to ``input_sr`` not matching ``model_sr``, or sample rate changes due to
        audio preprocessing from ``denoiser``.
    """
    denoiser, denoiser_options = convert_demucs_kwargs(
        denoiser, denoiser_options, demucs=demucs, demucs_options=demucs_options
    )
    if audio_type is not None and (audio_type := audio_type.lower()) not in AUDIO_TYPES:
        raise NotImplementedError(f'``audio_type="{audio_type}"`` is not supported. Types: {AUDIO_TYPES}')

    if isinstance(audio, AudioLoader) and audio_type is not None:
        raise ValueError(f'``audio_type`` can only be ``None`` when ``audio`` is an AudioLoader instance,'
                         f'but got {audio_type}')

    if audio_type is None:
        if type(audio) in AUDIO_TYPE_BY_CLASS:
            audio_type = AUDIO_TYPE_BY_CLASS[type(audio)]
        else:
            raise TypeError(f'{type(audio)} is not supported for ``audio``.')

    if (
            input_sr is None and
            isinstance(audio, (np.ndarray, torch.Tensor)) and
            (denoiser or only_voice_freq or suppress_silence or model_sr)
    ):
        raise ValueError('``input_sr`` is required when ``audio`` is a PyTorch tensor or NumPy array.')

    if (
            model_sr is None and
            isinstance(audio, (str, bytes)) and
            audio_type in ('torch', 'numpy')
    ):
        raise ValueError('``model_sr`` is required when ``audio_type`` is a "pytorch" or "numpy".')

    if isinstance(audio, str):
        audio = load_source(audio, verbose=verbose, only_ffmpeg=only_ffmpeg)

    if inference_kwargs is None:
        inference_kwargs = {}

    temp_file = os.path.abspath(temp_file or './_temp_stable-ts_audio_.wav')
    temp_audio_file = None

    if isinstance(audio, AudioLoader):
        if denoiser and not audio._denoiser:
            warnings.warn(
                '``denoiser`` will have no affect unless specified at AudioLoader initialization.',
                stacklevel=2
            )
        denoiser = None

        if only_voice_freq and not audio._only_voice_freq:
            warnings.warn(
                '``only_voice_freq=True`` will have no affect unless specified at AudioLoader initialization.',
                stacklevel=2
            )
        only_voice_freq = False

        if suppress_silence:
            warnings.warn(
                '``suppress_silence=True`` is not yet supported when ``audio`` is an AudioLoader.',
                stacklevel=2
            )
        suppress_silence = False

        if input_sr is not None and input_sr != audio.sr:
            warnings.warn(
                f'``input_sr`` ({input_sr}) does not match ``sr`` of AudioLoader ({audio.sr})',
                stacklevel=2
            )
        input_sr = audio.sr

    is_audio_encoded = isinstance(audio, (str, bytes))

    audio_sr = input_sr

    def curr_audio_sr(is_optional: bool = False):
        nonlocal audio_sr
        if is_optional and is_audio_encoded:
            return None
        if audio_sr is not None:
            return audio_sr
        assert isinstance(audio, (str, bytes)), f'No ``input_sr`` specified.'
        audio_sr = get_samplerate(audio)
        assert audio_sr is not None, 'Failed to get samplerate from ``audio``'
        return audio_sr

    if denoiser:
        denoise_model = denoiser_options.pop('model', None)
        if denoise_model is None:
            denoise_model = get_denoiser_func(denoiser, 'load')(True)
    else:
        denoise_model = None

    if denoiser:
        denoiser_options = update_options(
            denoiser_options,
            True,
            audio=torch.from_numpy(audio) if isinstance(audio, np.ndarray) else audio,
            input_sr=curr_audio_sr(True),
            model=denoise_model,
            verbose=verbose
        )
        denoise_run = get_denoiser_func(denoiser, 'run')
        audio = denoise_run(**denoiser_options)
        audio_sr = denoise_model.samplerate
        is_audio_encoded = False
        if (denoise_output := denoiser_options.get('save_path')) and audio_type == 'str':
            audio = denoise_output

    if only_voice_freq:
        if is_audio_encoded and audio_sr and model_sr:
            audio_sr = max(audio_sr, model_sr)
        audio = audio_to_tensor_resample(
            audio,
            original_sample_rate=curr_audio_sr(),
            verbose=verbose,
            only_ffmpeg=only_ffmpeg
        )
        audio = voice_freq_filter(audio, audio_sr)
        is_audio_encoded = False

    final_audio = audio

    if model_sr is None:
        final_audio_sr = audio_sr
    else:
        final_audio_sr = curr_audio_sr()

        if final_audio_sr != model_sr:
            if isinstance(final_audio, (str, bytes)):
                final_audio = load_audio(
                    final_audio,
                    sr=model_sr,
                    verbose=verbose,
                    only_ffmpeg=only_ffmpeg
                )
                final_audio_sr = model_sr
            else:
                if isinstance(final_audio, np.ndarray):
                    final_audio = torch.from_numpy(final_audio)
                if isinstance(final_audio, torch.Tensor):
                    final_audio = resample(final_audio, audio_sr, model_sr)
                    final_audio_sr = model_sr

    if audio_type in ('torch', 'numpy'):

        if isinstance(final_audio, (str, bytes)):
            final_audio = load_audio(
                final_audio,
                sr=model_sr,
                verbose=verbose,
                only_ffmpeg=only_ffmpeg
            )

        if not isinstance(final_audio, AudioLoader):
            if audio_type == 'torch':
                if isinstance(final_audio, np.ndarray):
                    final_audio = torch.from_numpy(final_audio)
            elif isinstance(final_audio, torch.Tensor):
                final_audio = final_audio.cpu().numpy()

    elif audio_type == 'str':

        if isinstance(final_audio, (torch.Tensor, np.ndarray)):
            if isinstance(final_audio, np.ndarray):
                final_audio = torch.from_numpy(final_audio)
            if final_audio.ndim < 2:
                final_audio = final_audio[None]
            torchaudio.save(temp_file, final_audio.cpu(), final_audio_sr)
            final_audio = temp_audio_file = temp_file

        elif isinstance(final_audio, bytes):
            with open(temp_file, 'wb') as f:
                f.write(final_audio)
            final_audio = temp_audio_file = temp_file

    elif audio_type == 'byte':

        if isinstance(final_audio, (torch.Tensor, np.ndarray)):
            if isinstance(final_audio, np.ndarray):
                final_audio = torch.from_numpy(final_audio)
            if final_audio.ndim < 2:
                final_audio = final_audio[None]
            with io.BytesIO() as f:
                torchaudio.save(f, final_audio.cpu(), final_audio_sr, format="wav")
                f.seek(0)
                final_audio = f.read()

        elif isinstance(final_audio, str):
            with open(final_audio, 'rb') as f:
                final_audio = f.read()

    inference_kwargs['audio'] = final_audio

    result = None
    try:
        result = inference_func(**inference_kwargs)
        if not isinstance(result, WhisperResult):
            result = WhisperResult(result, force_order=force_order, check_sorted=check_sorted)
        if suppress_silence:
            result.adjust_by_silence(
                audio, vad,
                vad_onnx=vad_onnx, vad_threshold=vad_threshold,
                q_levels=q_levels, k_size=k_size,
                sample_rate=curr_audio_sr(True), min_word_dur=min_word_dur,
                word_level=suppress_word_ts, verbose=verbose is not None,
                nonspeech_error=nonspeech_error,
                use_word_position=use_word_position,
                min_silence_dur=min_silence_dur
            )
            result.set_current_as_orig()

        if result.has_words and regroup:
            result.regroup(regroup)

    finally:
        if temp_audio_file is not None:
            try:
                os.unlink(temp_audio_file)
            except Exception as e:
                warnings.warn(f'Failed to remove temporary audio file {temp_audio_file}. {e}')

    return result

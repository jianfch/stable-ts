import os
import warnings
import io
import torch
import torchaudio
import numpy as np
from typing import Union, Callable, Optional

from .audio import load_audio
from .result import WhisperResult

AUDIO_TYPES = ('str', 'byte', 'torch', 'numpy')


def transcribe_any(
        inference_func: Callable,
        audio: Union[str, np.ndarray, torch.Tensor, bytes],
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
        demucs: bool = False,
        demucs_device: str = None,
        demucs_output: str = None,
        demucs_options: dict = None,
        vad: bool = False,
        vad_threshold: float = 0.35,
        vad_onnx: bool = False,
        min_word_dur: float = 0.1,
        nonspeech_error: float = 0.3,
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
        Function that runs ASR when provided the [audio] and return data in the appropriate format.
        For format examples see, https://github.com/jianfch/stable-ts/blob/main/examples/non-whisper.ipynb.
    audio : str or numpy.ndarray or torch.Tensor or bytes
        Path/URL to the audio file, the audio waveform, or bytes of audio file.
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
    demucs : bool or torch.nn.Module, default False
        Whether to preprocess ``audio`` with Demucs to isolate vocals / remove noise. Set ``demucs`` to an instance of
        a Demucs model to avoid reloading the model for each run.
        Demucs must be installed to use. Official repo, https://github.com/facebookresearch/demucs.
    demucs_output : str, optional
        Path to save the vocals isolated by Demucs as WAV file. Ignored if ``demucs = False``.
        Demucs must be installed to use. Official repo, https://github.com/facebookresearch/demucs.
    demucs_options : dict, optional
        Options to use for :func:`stable_whisper.audio.demucs_audio`.
    demucs_device : str, default None, meaning 'cuda' if cuda is available with ``torch`` else 'cpu'
        Device to use for demucs.
    vad : bool, default False
        Whether to use Silero VAD to generate timestamp suppression mask.
        Silero VAD requires PyTorch 1.12.0+. Official repo, https://github.com/snakers4/silero-vad.
    vad_threshold : float, default 0.35
        Threshold for detecting speech with Silero VAD. Low threshold reduces false positives for silence detection.
    vad_onnx : bool, default False
        Whether to use ONNX for Silero VAD.
    min_word_dur : float, default 0.1
        Shortest duration each word is allowed to reach for silence suppression.
    nonspeech_error : float, default 0.3
        Relative error of non-speech sections that appear in between a word for silence suppression.
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
        If audio preprocessing is ``demucs`` or ``only_voice_freq``, the processed audio will be encoded into
            ``temp_file`` and then passed into ``inference_func``.

    For ``audio_type = 'byte'``:
        If ``audio`` is file, the bytes of file will be passed into ``inference_func``.
        If ``audio`` is :class:`torch.Tensor` or :class:`numpy.ndarray`, the bytes of the ``audio`` will be encoded
            into WAV format then passed into ``inference_func``.

    Resampling is only performed on ``audio`` when ``model_sr`` does not match the sample rate of the ``audio`` before
        passing into ``inference_func`` due to ``input_sr`` not matching ``model_sr``, or sample rate changes due to
        audio preprocessing from ``demucs = True``.
    """
    if demucs_options is None:
        demucs_options = {}
    if demucs_output:
        if 'save_path' not in demucs_options:
            demucs_options['save_path'] = demucs_output
        warnings.warn('``demucs_output`` is deprecated. Use ``demucs_options`` with ``save_path`` instead. '
                      'E.g. demucs_options=dict(save_path="demucs_output.mp3")',
                      DeprecationWarning, stacklevel=2)
    if demucs_device:
        if 'device' not in demucs_options:
            demucs_options['device'] = demucs_device
        warnings.warn('``demucs_device`` is deprecated. Use ``demucs_options`` with ``device`` instead. '
                      'E.g. demucs_options=dict(device="cpu")',
                      DeprecationWarning, stacklevel=2)

    if audio_type is not None and (audio_type := audio_type.lower()) not in AUDIO_TYPES:
        raise NotImplementedError(f'[audio_type]={audio_type} is not supported. Types: {AUDIO_TYPES}')

    if audio_type is None:
        if isinstance(audio, str):
            audio_type = 'str'
        elif isinstance(audio, bytes):
            audio_type = 'byte'
        elif isinstance(audio, torch.Tensor):
            audio_type = 'pytorch'
        elif isinstance(audio, np.ndarray):
            audio_type = 'numpy'
        else:
            raise TypeError(f'{type(audio)} is not supported for [audio].')

    if (
            input_sr is None and
            isinstance(audio, (np.ndarray, torch.Tensor)) and
            (demucs or only_voice_freq or suppress_silence or model_sr)
    ):
        raise ValueError('[input_sr] is required when [audio] is a PyTorch tensor or NumPy array.')

    if (
            model_sr is None and
            isinstance(audio, (str, bytes)) and
            audio_type in ('torch', 'numpy')
    ):
        raise ValueError('[model_sr] is required when [audio_type] is a "pytorch" or "numpy".')

    if isinstance(audio, str):
        from .audio import _load_file
        audio = _load_file(audio, verbose=verbose, only_ffmpeg=only_ffmpeg)

    if inference_kwargs is None:
        inference_kwargs = {}

    temp_file = os.path.abspath(temp_file or './_temp_stable-ts_audio_.wav')
    temp_audio_file = None

    curr_sr = input_sr

    if demucs:
        if demucs is True:
            from .audio import load_demucs_model
            demucs_model = load_demucs_model()
        else:
            demucs_model = demucs
            demucs = True
    else:
        demucs_model = None

    def get_input_sr():
        nonlocal input_sr
        if not input_sr and isinstance(audio, (str, bytes)):
            from .audio import get_samplerate
            input_sr = get_samplerate(audio)
        return input_sr

    if only_voice_freq:
        from .audio import voice_freq_filter
        if demucs_model is None:
            curr_sr = model_sr or get_input_sr()
        else:
            curr_sr = demucs_model.samplerate
            if model_sr is None:
                model_sr = get_input_sr()
        audio = load_audio(audio, sr=curr_sr, verbose=verbose, only_ffmpeg=only_ffmpeg)
        audio = voice_freq_filter(audio, curr_sr)

    if demucs:
        from .audio import demucs_audio
        if demucs_device is None:
            demucs_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        demucs_kwargs = dict(
            audio=audio,
            input_sr=curr_sr,
            model=demucs_model,
            save_path=demucs_output,
            device=demucs_device,
            verbose=verbose
        )
        demucs_kwargs.update(demucs_options or {})
        audio = demucs_audio(
            **demucs_kwargs
        )
        curr_sr = demucs_model.samplerate
        if demucs_output and audio_type == 'str':
            audio = demucs_output

    final_audio = audio

    if model_sr is not None:

        if curr_sr is None:
            curr_sr = get_input_sr()

        if curr_sr != model_sr:
            if isinstance(final_audio, (str, bytes)):
                final_audio = load_audio(
                    final_audio,
                    sr=model_sr,
                    verbose=verbose,
                    only_ffmpeg=only_ffmpeg
                )
            else:
                if isinstance(final_audio, np.ndarray):
                    final_audio = torch.from_numpy(final_audio)
                if isinstance(final_audio, torch.Tensor):
                    final_audio = torchaudio.functional.resample(
                        final_audio,
                        orig_freq=curr_sr,
                        new_freq=model_sr,
                        resampling_method="kaiser_window"
                    )

    if audio_type in ('torch', 'numpy'):

        if isinstance(final_audio, (str, bytes)):
            final_audio = load_audio(
                final_audio,
                sr=model_sr,
                verbose=verbose,
                only_ffmpeg=only_ffmpeg
            )

        else:
            if audio_type == 'torch':
                if isinstance(final_audio, np.ndarray):
                    final_audio = torch.from_numpy(final_audio)
            elif audio_type == 'numpy' and isinstance(final_audio, torch.Tensor):
                final_audio = final_audio.cpu().numpy()

    elif audio_type == 'str':

        if isinstance(final_audio, (torch.Tensor, np.ndarray)):
            if isinstance(final_audio, np.ndarray):
                final_audio = torch.from_numpy(final_audio)
            if final_audio.ndim < 2:
                final_audio = final_audio[None]
            torchaudio.save(temp_file, final_audio, model_sr)
            final_audio = temp_audio_file = temp_file

        elif isinstance(final_audio, bytes):
            with open(temp_file, 'wb') as f:
                f.write(final_audio)
            final_audio = temp_audio_file = temp_file

    else:  # audio_type == 'byte'

        if isinstance(final_audio, (torch.Tensor, np.ndarray)):
            if isinstance(final_audio, np.ndarray):
                final_audio = torch.from_numpy(final_audio)
            if final_audio.ndim < 2:
                final_audio = final_audio[None]
            with io.BytesIO() as f:
                torchaudio.save(f, final_audio, model_sr, format="wav")
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
                sample_rate=curr_sr, min_word_dur=min_word_dur,
                word_level=suppress_word_ts, verbose=True,
                nonspeech_error=nonspeech_error
            )

        if result.has_words and regroup:
            result.regroup(regroup)

    finally:
        if temp_audio_file is not None:
            try:
                os.unlink(temp_audio_file)
            except Exception as e:
                warnings.warn(f'Failed to remove temporary audio file {temp_audio_file}. {e}')

    return result

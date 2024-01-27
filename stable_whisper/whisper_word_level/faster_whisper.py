from types import MethodType
from typing import Union, Optional, Callable

import numpy as np
from tqdm import tqdm

from ..result import Segment, WhisperResult
from ..non_whisper import transcribe_any
from ..utils import safe_print, isolate_useful_options
from ..audio import audioloader_not_supported, convert_demucs_kwargs

from whisper.tokenizer import LANGUAGES
from whisper.audio import SAMPLE_RATE


def faster_transcribe(
        model: "WhisperModel",
        audio: Union[str, bytes, np.ndarray],
        *,
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
        demucs: bool = False,
        demucs_options: dict = None,
        vad: bool = False,
        vad_threshold: float = 0.35,
        vad_onnx: bool = False,
        min_word_dur: Optional[float] = None,
        nonspeech_error: float = 0.1,
        only_voice_freq: bool = False,
        only_ffmpeg: bool = False,
        check_sorted: bool = True,
        progress_callback: Callable = None,
        **options
) -> WhisperResult:
    """
    Transcribe audio using faster-whisper (https://github.com/guillaumekln/faster-whisper).

    This is uses the transcribe method from faster-whisper, :meth:`faster_whisper.WhisperModel.transcribe`, while
    still allowing additional preprocessing and postprocessing. The preprocessing performed on the audio includes:
    voice isolation / noise removal and low/high-pass filter. The postprocessing performed on the
    transcription result includes: adjusting timestamps with VAD and custom regrouping segments based punctuation
    and speech gaps.

    Parameters
    ----------
    model : faster_whisper.WhisperModel
        The faster-whisper ASR model instance.
    audio : str or numpy.ndarray or torch.Tensor or bytes
        Path/URL to the audio file, the audio waveform, or bytes of audio file.
        If audio is :class:`numpy.ndarray` or :class:`torch.Tensor`, the audio must be already at sampled to 16kHz.
    verbose : bool or None, default False
        Whether to display the text being decoded to the console.
        Displays all the details if ``True``. Displays progressbar if ``False``. Display nothing if ``None``.
    word_timestamps : bool, default True
        Extract word-level timestamps using the cross-attention pattern and dynamic time warping,
        and include the timestamps for each word in each segment.
        Disabling this will prevent segments from splitting/merging properly.
    regroup : bool or str, default True, meaning the default regroup algorithm
        String for customizing the regrouping algorithm. False disables regrouping.
        Ignored if ``word_timestamps = False``.
    suppress_silence : bool, default True
        Whether to enable timestamps adjustments based on the detected silence.
    suppress_word_ts : bool, default True
        Whether to adjust word timestamps based on the detected silence. Only enabled if ``suppress_silence = True``.
    use_word_position : bool, default True
        Whether to use position of the word in its segment to determine whether to keep end or start timestamps if
        adjustments are required. If it is the first word, keep end. Else if it is the last word, keep the start.
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
    nonspeech_error : float, default 0.3
        Relative error of non-speech sections that appear in between a word for silence suppression.
    only_voice_freq : bool, default False
        Whether to only use sound between 200 - 5000 Hz, where majority of human speech are.
    only_ffmpeg : bool, default False
        Whether to use only FFmpeg (instead of not yt-dlp) for URls
    check_sorted : bool, default True
        Whether to raise an error when timestamps returned by faster-whipser are not in ascending order.
    progress_callback : Callable, optional
        A function that will be called when transcription progress is updated.
        The callback need two parameters.
        The first parameter is a float for seconds of the audio that has been transcribed.
        The second parameter is a float for total duration of audio in seconds.
    options
        Additional options used for :meth:`faster_whisper.WhisperModel.transcribe` and
        :func:`stable_whisper.non_whisper.transcribe_any`.

    Returns
    -------
    stable_whisper.result.WhisperResult
        All timestamps, words, probabilities, and other data from the transcription of ``audio``.

    Examples
    --------
    >>> import stable_whisper
    >>> model = stable_whisper.load_faster_whisper('base')
    >>> result = model.transcribe_stable('audio.mp3', vad=True)
    >>> result.to_srt_vtt('audio.srt')
    Saved: audio.srt
    """
    audioloader_not_supported(audio)
    extra_options = isolate_useful_options(options, transcribe_any, pop=True)
    denoiser, denoiser_options = convert_demucs_kwargs(
        denoiser, denoiser_options, demucs=demucs, demucs_options=demucs_options
    )
    if denoiser or only_voice_freq:
        if 'audio_type' not in extra_options:
            extra_options['audio_type'] = 'numpy'
        if 'model_sr' not in extra_options:
            extra_options['model_sr'] = SAMPLE_RATE
    faster_whisper_options = options
    faster_whisper_options['model'] = model
    faster_whisper_options['audio'] = audio
    faster_whisper_options['word_timestamps'] = word_timestamps
    faster_whisper_options['verbose'] = verbose
    faster_whisper_options['progress_callback'] = progress_callback

    return transcribe_any(
        inference_func=_inner_transcribe,
        audio=audio,
        inference_kwargs=faster_whisper_options,
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
        **extra_options
    )


def _inner_transcribe(model, audio, verbose, **faster_transcribe_options):
    if isinstance(audio, bytes):
        import io
        audio = io.BytesIO(audio)
    progress_callback = faster_transcribe_options.pop('progress_callback', None)
    segments, info = model.transcribe(audio, **faster_transcribe_options)
    language = LANGUAGES.get(info.language, info.language)
    if verbose is not None:
        print(f'Detected Language: {language}')
        print(f'Transcribing with faster-whisper ({model.model_size_or_path})...\r', end='')

    final_segments = []
    task = faster_transcribe_options.get('task', 'transcribe').title()
    total_duration = round(info.duration, 2)

    with tqdm(total=total_duration, unit='sec', disable=verbose is not False, desc=task) as tqdm_pbar:

        def update_pbar(seek):
            tqdm_pbar.update(seek - tqdm_pbar.n)
            if progress_callback is not None:
                progress_callback(seek, total_duration)

        for segment in segments:
            segment = segment._asdict()
            if (words := segment.get('words')) is not None:
                segment['words'] = [w._asdict() for w in words]
            else:
                del segment['words']
            if verbose:
                safe_print(Segment(**segment).to_display_str())
            final_segments.append(segment)
            update_pbar(segment["end"])
        update_pbar(tqdm_pbar.total)

    if verbose:
        print(f'Completed transcription with faster-whisper ({model.model_size_or_path}).')

    return dict(language=language, segments=final_segments)


def load_faster_whisper(model_size_or_path: str, **model_init_options):
    """
    Load an instance of :class:`faster_whisper.WhisperModel`.

    Parameters
    ----------
    model_size_or_path : {'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v1',
        'large-v2', 'large-v3', or 'large'}
        Size of the model.

    model_init_options
        Additional options to use for initialization of :class:`faster_whisper.WhisperModel`.

    Returns
    -------
    faster_whisper.WhisperModel
        A modified instance with :func:`stable_whisper.whisper_word_level.faster_whisper.faster_transcribe`
        assigned to :meth:`faster_whisper.WhisperModel.transcribe_stable`.
    """
    from faster_whisper import WhisperModel
    faster_model = WhisperModel(model_size_or_path, **model_init_options)
    faster_model.model_size_or_path = model_size_or_path

    faster_model.transcribe_stable = MethodType(faster_transcribe, faster_model)
    from ..alignment import align
    faster_model.align = MethodType(align, faster_model)

    return faster_model

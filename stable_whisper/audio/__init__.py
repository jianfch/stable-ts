import subprocess
import warnings
import torch
import numpy as np
from typing import Union, Optional, Tuple

from .utils import (
    is_ytdlp_available, load_source, load_audio, voice_freq_filter, get_samplerate, get_metadata
)
from .demucs import is_demucs_available, load_demucs_model, demucs_audio
from .dfnet import is_dfnet_available, load_dfnet_model, dfnet_audio
from .output import save_audio_tensor
from ..utils import update_options

from whisper.audio import SAMPLE_RATE


SUPPORTED_DENOISERS = {
    'demucs': {'run': demucs_audio, 'load': load_demucs_model, 'access': is_demucs_available},
    'dfnet': {'run': dfnet_audio, 'load': load_dfnet_model, 'access': is_dfnet_available}
}


def get_denoiser_func(denoiser: Optional[str], key: str):
    if denoiser is None:
        return
    if denoiser not in SUPPORTED_DENOISERS:
        raise NotImplementedError(f'"{denoiser}" is one of the supported denoisers: '
                                  f'{tuple(SUPPORTED_DENOISERS.keys())}.')
    assert key in ('run', 'load', 'access')
    return SUPPORTED_DENOISERS[denoiser][key]


def _load_file(**kwargs):
    warnings.warn('This function is deprecated. Use `stable_whisper.audio.load_source()`.',
                  stacklevel=2)
    return load_source(**kwargs)


def convert_demucs_kwargs(
        denoiser: Optional[str] = None,
        denoiser_options: Optional[dict] = None,
        demucs: bool = None,
        demucs_options: Optional[dict] = None,
) -> Tuple[Union[str, None], dict]:
    if demucs:
        warnings.warn('``demucs`` is deprecated and will be removed in future versions. '
                      'Use ``denoiser="demucs"`` instead.',
                      stacklevel=2)
        if denoiser:
            if denoiser != 'demucs':
                raise ValueError(f'Demucs is enabled but got "{denoiser}" for denoiser.')
        else:
            denoiser = 'demucs'

    if denoiser_options is None:
        denoiser_options = {}

    if demucs_options:
        warnings.warn('``demucs_options`` is deprecated and will be removed in future versions. '
                      'Use ``denoiser_options`` instead.',
                      stacklevel=2)
        if denoiser == 'demucs':
            if demucs_options:
                denoiser_options = demucs_options
            if isinstance(demucs, torch.nn.Module):
                denoiser_options['model'] = demucs

    return denoiser, denoiser_options


def prep_audio(
        audio: Union[str, np.ndarray, torch.Tensor, bytes],
        denoiser: Optional[str] = None,
        denoiser_options: Optional[dict] = None,
        only_voice_freq: bool = False,
        only_ffmpeg: bool = False,
        verbose: Optional[bool] = False,
        sr: int = None,
        demucs: Optional[str] = None,
        demucs_options: Optional[dict] = None,
) -> torch.Tensor:
    """
    Converts input audio of many types into a mono waveform as a torch.Tensor.

    Parameters
    ----------
    audio : str or numpy.ndarray or torch.Tensor or bytes
        Path/URL to the audio file, the audio waveform, or bytes of audio file.
        If audio is :class:`numpy.ndarray` or :class:`torch.Tensor`, the audio must be already at sampled to 16kHz.
    denoiser : str, optional
        String of the denoiser to use for preprocessing ``audio``.
        See ``stable_whisper.audio.SUPPORTED_DENOISERS`` for supported denoisers.
    denoiser_options : dict, optional
        Options to use for ``denoiser``.
    only_voice_freq : bool, default False
        Whether to only use sound between 200 - 5000 Hz, where majority of human speech are.
    sr : int, default None, meaning ``whisper.audio.SAMPLE_RATE``, 16kHZ
        The sample rate of ``audio``.
    verbose : bool or None, default True
        Verbosity for yd-dlp and displaying content metadata when ``file`` is a URL. If not ``None``, display metadata.
        For yd-dlp: ``None`` is "--quiet"; ``True`` is "--progress"; ``False`` is "--progress" + "--quiet".
    only_ffmpeg: bool, default False
        Whether to use only FFmpeg (and not yt-dlp) for URls.

    Returns
    -------
    torch.Tensor
        A mono waveform.
    """
    denoiser, denoiser_options = convert_demucs_kwargs(
        denoiser, denoiser_options, demucs=demucs, demucs_options=demucs_options
    )
    if not sr:
        sr = SAMPLE_RATE

    denoise_func = get_denoiser_func(denoiser, 'run')

    if isinstance(audio, (str, bytes)):
        if denoise_func is None:
            audio = torch.from_numpy(load_audio(audio, sr=sr, verbose=verbose, only_ffmpeg=only_ffmpeg))
        else:
            denoiser_options = update_options(
                denoiser_options,
                True,
                audio=audio,
                output_sr=sr,
                verbose=verbose
            )
            audio = denoise_func(**denoiser_options)
    else:
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        if denoise_func is not None:
            denoiser_options = update_options(
                denoiser_options,
                True,
                audio=audio,
                input_sr=sr,
                output_sr=sr,
                verbose=verbose,
            )
            audio = denoise_func(**denoiser_options)
    if only_voice_freq:
        audio = voice_freq_filter(audio.cpu(), sr)

    return audio


class AudioLoader:
    def __init__(
            self,
            source: Union[str, np.ndarray, torch.Tensor, bytes],
            buffer_size: Union[int, str] = None,
            stream: Optional[bool] = None,
            sr: int = None,
            test_first_chunk: bool = True,
            verbose: Optional[bool] = False,
            only_ffmpeg: bool = False,
            new_chunk_divisor: Optional[int] = 512,
            save_path: Optional[str] = None,
            post_prep_callback=None,
            denoiser: Optional[str] = None,
            denoiser_options: Optional[dict] = None,
            only_voice_freq: bool = False,
            demucs: Optional[str] = None,
            demucs_options: Optional[dict] = None,
    ):
        if stream and not isinstance(source, str):
            raise NotImplementedError(f'``stream=True`` only supported for string ``source`` but got {type(source)}.')
        self.source = source
        if sr is None:
            from whisper.audio import SAMPLE_RATE
            sr = SAMPLE_RATE
        self._sr = sr
        if buffer_size is None:
            buffer_size = (sr * 30)
        self._buffer_size = self._valid_buffer_size(self.parse_chunk_size(buffer_size))
        self._stream = isinstance(source, str) if stream is None else stream
        self._accum_samples = 0
        self.verbose = verbose
        self.only_ffmpeg = only_ffmpeg
        self.new_chunk_divisor = new_chunk_divisor
        self._post_prep_callback = post_prep_callback
        self._denoiser = denoiser
        self._denoiser_options = denoiser_options or {}
        self._denoiser, self._denoiser_options = convert_demucs_kwargs(
            self._denoiser, self._denoiser_options, demucs=demucs, demucs_options=demucs_options
        )
        self._final_save_path = save_path
        self._denoised_save_path = self._denoiser_options.pop('save_path', None)
        self._only_voice_freq = only_voice_freq
        self._denoised_samples_to_save = []
        self._final_samples_to_save = []
        metadata = get_metadata(source)
        self._source_sr, self._duration_estimation = metadata['sr'] or 0, metadata['duration'] or 0
        self._total_sample_estimation = round(self._duration_estimation * self._sr)
        self._denoise_model, self._min_chunk = self._load_denoise_model()
        self.check_min_chunk_requirement()
        self._prep = self._get_prep_func()
        self._extra_process = None
        self._prev_seek = None
        self._buffered_samples = torch.tensor([])
        self._prev_unprep_samples = np.array([])
        self._process = self._audio_loading_process()
        if test_first_chunk and self.next_chunk(0) is None:
            raise RuntimeError(f'FFmpeg failed to read "{source}".')

    @property
    def buffer_size(self):
        return self._buffer_size

    @property
    def sr(self):
        return self._sr

    @property
    def source_sr(self):
        return self._source_sr

    @property
    def stream(self):
        return self._stream

    @property
    def prev_seek(self):
        return self._prev_seek

    @buffer_size.setter
    def buffer_size(self, size: int):
        self._buffer_size = self._valid_buffer_size(size)
        self.check_min_chunk_requirement()

    @staticmethod
    def _valid_buffer_size(size: int):
        if size < 0:
            raise ValueError('buffer size must be at least 0')
        return size

    def parse_chunk_size(self, chunk_size: Union[int, str]) -> int:
        if isinstance(chunk_size, int):
            return chunk_size
        if not chunk_size.endswith('s'):
            raise ValueError('string ``chunk_size`` must end with "s"')
        chunk_size = chunk_size[:-1]
        seconds = float(chunk_size)
        return round(seconds * self._sr)

    def _load_denoise_model(self):
        if not self._denoiser:
            return None, None
        model = get_denoiser_func(self._denoiser, 'load')(True)
        length = int(getattr(model, 'segment', 5) * self._sr)
        return model, length

    def check_min_chunk_requirement(self):
        if self._min_chunk is not None and self._min_chunk > self.buffer_size:
            warnings.warn(
                f'It is advised to keep ``chunk_size`` ({self.buffer_size}) at least {self._min_chunk}'
                f' or "{round(self._min_chunk / self._sr)}s" for ``demucs=True``.',
                stacklevel=2
            )

    def get_duration(self, ndigits: int = None):
        if self._stream:
            dur = (
                self._duration_estimation
                if (dur := (self._accum_samples or 0) / self._sr) < self._duration_estimation else
                dur
            )
        else:
            dur = self._duration_estimation
        return dur if ndigits is None else round(dur, ndigits=ndigits)

    def get_total_samples(self):
        if not self._stream:
            return self._total_sample_estimation
        return (
            self._total_sample_estimation
            if (self._accum_samples / self._sr) < self._duration_estimation else
            self._accum_samples
        )

    def update_post_prep_callback(self, callback):
        self._post_prep_callback = callback
        if self._post_prep_callback is not None and len(self._buffered_samples):
            self._post_prep_callback(self._buffered_samples)

    def __del__(self):
        self.terminate()

    def divisible_min_chunk(self, min_chunk: int) -> int:
        if not self.new_chunk_divisor:
            return min_chunk
        if r := min_chunk % self.new_chunk_divisor:
            return min_chunk + self.new_chunk_divisor - r
        return min_chunk

    def _seek_buffered_samples(self, seek) -> int:
        is_first_time_loading = self._prev_seek is None
        if is_first_time_loading:
            if self._process is None:
                samples_to_load_discard = 0
                self._buffered_samples = self._prep(self.source)
                if self._final_save_path:
                    self._final_samples_to_save.append(self._buffered_samples.cpu())
                self._total_sample_estimation = self._buffered_samples.shape[-1]
                self._duration_estimation = (self._total_sample_estimation / self._sr)
                self._buffered_samples = self._buffered_samples[seek:]
            else:
                samples_to_load_discard = seek
                self._buffered_samples = torch.tensor([])
        else:
            assert seek >= self._prev_seek, '``seek`` must be >= the previous ``seek`` value'
            seek_delta = seek - self._prev_seek
            samples_to_load_discard = max(0, seek_delta - len(self._buffered_samples))
            self._buffered_samples = self._buffered_samples[seek_delta:]

        self._prev_seek = seek

        return samples_to_load_discard

    def _read_samples(self, samples: int) -> bytes:
        if self._process is None or self._process.poll() is not None:
            return b''
        n = samples * 2
        b = b''
        while len(b) < n and self._process.poll() is None:
            nb = self._process.stdout.read(n)
            b += nb
        return b

    def _read_append_to_buffer(self, samples_to_read: int, samples_to_discard: Optional[int] = None):
        sample_bytes = self._read_samples(samples_to_read)
        if not sample_bytes:
            return
        new_samples = self._prep_samples(sample_bytes, samples_to_discard)
        if len(self._buffered_samples):
            self._buffered_samples = torch.concat([self._buffered_samples, new_samples], dim=-1)
        else:
            self._buffered_samples = new_samples

    def save_denoised_audio(self, path: Optional[str] = None):
        if not self._denoised_samples_to_save:
            warnings.warn('Failed to save denoised audio. No stored denoised audio samples found.',
                          stacklevel=2)
            return
        if not (path or self._denoised_save_path):
            warnings.warn('Failed to save denoised audio. No specified path to save.',
                          stacklevel=2)
            return
        save_audio_tensor(torch.cat(self._denoised_samples_to_save), path or self._denoised_save_path, self._sr)

    def save_final_audio(self, path: Optional[str] = None):
        if not self._final_samples_to_save:
            warnings.warn('Failed to save final audio. No stored final audio samples found.',
                          stacklevel=2)
            return
        if not (path or self._final_save_path):
            warnings.warn('Failed to save denoised audio. No specified path to save.',
                          stacklevel=2)
            return
        save_audio_tensor(torch.cat(self._final_samples_to_save), path or self._final_save_path, self._sr)

    def next_chunk(self, seek: int, size: Optional[int] = None) -> Union[torch.Tensor, None]:
        samples_to_load_discard = self._seek_buffered_samples(seek)
        samples_to_load_keep = max(self._buffer_size, size or 0) - len(self._buffered_samples)
        if samples_to_load_keep > 0:
            samples_to_load_keep = self.divisible_min_chunk(samples_to_load_keep)
        samples_to_load = max(samples_to_load_discard + samples_to_load_keep, 0)
        self._read_append_to_buffer(samples_to_load, samples_to_load_discard)

        samples = self._buffered_samples[:self._buffer_size if size is None else size]

        return samples if len(samples) else None

    def _get_prep_func(self):

        if self._denoiser:
            self._denoiser_options['model'] = self._denoise_model
            if 'progress' not in self._denoiser_options:
                self._denoiser_options['progress'] = False if self._stream else (self.verbose is not None)

        if self._stream:
            if self._denoised_save_path:
                if self._final_save_path:
                    warnings.warn('Both ``save_path`` in AudioLoad and ``denoiser_options`` were specified, '
                                  'but only the final audio will be saved for ``stream=True`` in either case. '
                                  '``denoiser_options`` will be prioritized for ``save_path``.',
                                  stacklevel=2)
                else:
                    self._final_save_path = self._denoised_save_path
                self._denoised_save_path = None

            denoise_func = get_denoiser_func(self._denoiser, 'run')

            def prep(audio):
                audio = torch.from_numpy(audio)
                if denoise_func is not None:
                    denoiser_options = update_options(
                        self._denoiser_options,
                        True,
                        audio=audio,
                        input_sr=self._sr,
                        output_sr=self._sr,
                        verbose=self.verbose,
                    )
                    audio = denoise_func(**denoiser_options)
                if self._only_voice_freq:
                    audio = voice_freq_filter(audio.cpu(), self._sr)

                return audio

            return prep

        if self._denoised_save_path:
            def append_denoised(samples: torch.Tensor):
                self._denoised_samples_to_save.append(samples.cpu())

            self._denoiser_options['save_path'] = append_denoised

        def prep(audio):
            return prep_audio(
                audio,
                denoiser=self._denoiser,
                denoiser_options=self._denoiser_options,
                only_voice_freq=self._only_voice_freq,
                only_ffmpeg=self.only_ffmpeg,
                verbose=self.verbose,
                sr=self._sr,
            )

        return prep

    def _prep_samples(self, new_samples: bytes, samples_to_discard: Optional[int] = None) -> torch.Tensor:
        if samples_to_discard:
            assert not len(self._buffered_samples)
            i = samples_to_discard * 2
            discarded_samples_bytes, new_samples = new_samples[:i], new_samples[i:]
        else:
            discarded_samples_bytes = b''
        new_samples = np.frombuffer(new_samples, np.int16).flatten().astype(np.float32) / 32768.0
        new_sample_length = new_samples.shape[-1]
        self._accum_samples += new_sample_length
        if self._min_chunk:
            if (
                    len(self._prev_unprep_samples) + len(discarded_samples_bytes)
                    and (missing_length := self._min_chunk - new_sample_length) > 0
            ):
                prev_unprep_samples = self._prev_unprep_samples
                if discarded_samples_bytes:
                    discarded_samples_bytes = discarded_samples_bytes[-missing_length*2:]
                    discarded_samples_bytes = np.frombuffer(
                        discarded_samples_bytes, np.int16
                    ).flatten().astype(np.float32) / 32768.0
                    if discarded_samples_bytes.shape[-1] < missing_length:
                        prev_unprep_samples = np.concatenate((prev_unprep_samples, discarded_samples_bytes), axis=-1)
                    else:
                        prev_unprep_samples = discarded_samples_bytes
                prev_unprep_samples = prev_unprep_samples[-missing_length:]
                new_samples = np.concatenate((prev_unprep_samples, new_samples), axis=-1)
                prepped_samples = self._prep(new_samples)[-new_sample_length:]
            else:
                prepped_samples = self._prep(new_samples)
            self._prev_unprep_samples = new_samples
        else:
            prepped_samples = self._prep(new_samples)

        if self._final_save_path:
            self._final_samples_to_save.append(prepped_samples.cpu())
        if self._post_prep_callback is not None:
            self._post_prep_callback(prepped_samples)

        return prepped_samples

    def terminate(self):
        if getattr(self, '_extra_process', None) is not None and self._extra_process.poll() is None:
            self._extra_process.terminate()
        if getattr(self, '_process', None) is not None and self._process.poll() is None:
            self._process.terminate()
        if getattr(self, '_denoised_save_path'):
            self.save_denoised_audio()
        if getattr(self, '_final_save_path', None):
            self.save_final_audio()

    def _audio_loading_process(self):
        if not isinstance(self.source, str) or not self._stream:
            return
        only_ffmpeg = False
        source = load_source(self.source, verbose=self.verbose, only_ffmpeg=only_ffmpeg, return_dict=True)
        if isinstance(source, dict):
            info = source
            source = info.pop('popen')
        else:
            info = None
        if info and info['duration']:
            self._duration_estimation = info['duration']
            if not self._stream and info['is_live']:
                warnings.warn('The audio appears to be a continuous stream but setting was set to `stream=False`.')

        if isinstance(source, subprocess.Popen):
            self._extra_process, stdin = source, source.stdout
        else:
            stdin = None
        try:
            # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
            # Requires the ffmpeg CLI in PATH.
            cmd = [
                "ffmpeg",
                "-loglevel", "error",
                "-nostdin",
                "-threads", "0",
                "-i", self.source if stdin is None else "pipe:",
                "-f", "s16le",
                "-ac", "1",
                "-acodec", "pcm_s16le",
                "-ar", str(self._sr),
                "-"
            ]
            out = subprocess.Popen(cmd, stdin=stdin, stdout=subprocess.PIPE)

        except subprocess.SubprocessError as e:
            raise RuntimeError(f"Failed to load audio: {e}") from e

        return out

    def validate_external_args(
            self,
            sr=None,
            vad=None,
            stream=None,
            denoiser=None,
            denoiser_options=None,
            only_voice_freq=False
    ):
        if sr and sr != self._sr:
            raise ValueError(
                f'AudioLoader must be initialized with ``sr={sr}`` but ``sr`` of this instance is {self._sr}.'
            )
        if vad:
            from ..stabilization.silero_vad import VAD_WINDOWS
            if self.new_chunk_divisor not in VAD_WINDOWS:
                raise ValueError(
                    f'When ``vad=True``, AudioLoader must be initialized with ``chunk_divisor`` '
                    f'set to one of the following values: {VAD_WINDOWS}. '
                    f'But got {self.new_chunk_divisor} instead.'
                )
        if stream and not self._stream:
            warnings.warn(
                f'``stream=True`` will have no effect unless specified at AudioLoader initialization.',
                stacklevel=2
            )

        if denoiser and not self._denoiser:
            warnings.warn(
                '``denoiser`` will have no effect unless specified at AudioLoader initialization.',
                stacklevel=2
            )

        if denoiser_options and denoiser_options != self._denoiser_options:
            warnings.warn(
                f'``denoiser_options`` does not match the ``denoiser_options`` '
                f'AudioLoader was initialized with.',
                stacklevel=2
            )

        if only_voice_freq and not self._only_voice_freq:
            warnings.warn(
                '``only_voice_freq=True`` will have no effect unless specified at AudioLoader initialization.',
                stacklevel=2
            )
        return self._stream, self._denoiser, self._denoiser_options, self._only_voice_freq


def audioloader_not_supported(audio):
    if isinstance(audio, AudioLoader):
        raise NotImplementedError('This function does not support AudioLoader instances.')

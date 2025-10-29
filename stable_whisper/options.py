import warnings
from typing import Union, List, Optional, Callable


class BasicOptions:

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def pop(self):
        kwargs = self._kwargs
        if self._kwargs:
            self._kwargs = {}
        return self, kwargs

    def raise_extras(self):
        if self._kwargs:
            raise TypeError(f'got unexpected keyword argument(s): {", ".join(self._kwargs.keys())}')

    def dict(self) -> dict:
        d = self.__dict__.copy()
        del d['_kwargs']
        return d

    def _has_obj(self, key: str):
        if key not in self._kwargs:
            return False
        kwargs = self._kwargs.pop(key)
        if kwargs is None:
            return False
        if not isinstance(kwargs, self.__class__):
            raise TypeError(f'expected "{key}" to be {self.__class__} but got {type(kwargs)}')
        for k, v in kwargs.__dict__.items():
            self.__setattr__(k, v)
        return True

    def _pop(self, key: str, default):
        return self._kwargs.pop(key, default)

    def update(self, options: dict):
        for k in list(options.keys()):
            if hasattr(self, k):
                setattr(self, k, options.pop(k))


class AllOptions(BasicOptions):

    def __init__(
            self,
            options: dict,
            progress: bool = True,
            pre: bool = True,
            post: bool = True,
            silence: bool = True,
            align: bool = True,
            vanilla_align: bool = False
    ):
        super().__init__(**options)
        if self._has_obj('all_options'):
            return
        self.progress: Union[None, ProgressOptions] = \
            self._process(progress and ProgressOptions, 'progress_options')
        self.pre: Union[None, PreprocessingOptions] = \
            self._process(pre and PreprocessingOptions, 'preprocessing_options')
        self.post: Union[None, PostprocessingOptions] = \
            self._process(post and PostprocessingOptions, 'postprocessing_options')
        self.silence: Union[None, SilenceOptions] = \
            self._process(silence and SilenceOptions, 'silence_options')
        self.align: Union[None, AlignmentOptions] = \
            self._process(align and AlignmentOptions, 'alignment_options')
        if self.align is not None and not vanilla_align:
            self.align.to_non_vanilla()
        self.raise_extras()

    def dict(self) -> dict:
        return dict(
            progress_options=self.progress,
            preprocessing_options=self.pre,
            postprocessing_options=self.post,
            silence_options=self.silence,
            alignment_options=self.align
        )

    def update(self, options: dict):
        if not options:
            return
        for option_obj in self.dict().values():
            if option_obj is not None:
                option_obj.update(options)
        self._kwargs = options
        self.raise_extras()

    def _process(self, option_class, key: str):
        if not option_class:
            return None
        if key in self._kwargs:
            option = option_class(**self._kwargs.pop(key))
            option.raise_extra()
        else:
            option, self._kwargs = option_class(**self._kwargs).pop()
        return option


class ProgressOptions(BasicOptions):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self._has_obj('progress_options'):
            return
        self.verbose: Optional[bool] = self._pop('verbose', False)
        self.progress_callback: Callable = self._pop('progress_callback', None)


class PreprocessingOptions(BasicOptions):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self._has_obj('preprocessing_options'):
            return
        self.denoiser: Optional[str] = self._pop('denoiser', None)
        self.denoiser_options: Optional[dict] = self._pop('denoiser_options', None)
        self.only_voice_freq: bool = self._pop('only_voice_freq', False)
        self.stream: Optional[bool] = self._pop('stream', None)
        self.only_ffmpeg: bool = self._pop('stream', False)


class SilenceOptions(BasicOptions):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self._has_obj('silence_options'):
            return
        self.q_levels: int = self._pop('q_levels', 20)
        self.k_size: int = self._pop('k_size', 5)
        self.vad: Union[bool, dict] = self._pop('vad', False)
        self.vad_threshold: float = self._pop('vad_threshold', 0.35)


class PostprocessingOptions(BasicOptions):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self._has_obj('postprocessing_options'):
            return
        self.regroup: bool = self._pop('regroup', True)
        self.suppress_silence: bool = self._pop('suppress_silence', True)
        self.suppress_word_ts: bool = self._pop('suppress_word_ts', True)
        self.use_word_position: bool = self._pop('use_word_position', True)
        self.min_word_dur: Optional[float] = self._pop('min_word_dur', None)
        self.min_silence_dur: Optional[float] = self._pop('min_silence_dur', None)
        self.nonspeech_error: float = self._pop('nonspeech_error', 0.1)
        self.prepend_punctuations: Optional[str] = self._pop('prepend_punctuations', None)
        self.append_punctuations: Optional[str] = self._pop('append_punctuations', None)


class AlignmentOptions(BasicOptions):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self._has_obj('alignment_options'):
            return
        self.split_callback: Callable = self._pop('split_callback', None)
        self.gap_padding: Optional[str] = self._pop('gap_padding', ' ...')
        self.presplit: Union[bool, List[str]] = self._pop('presplit', True)
        self.extra_models: Optional[list] = self._pop('extra_models', None)
        self.dynamic_heads: Optional[Union[bool, int, str]] = self._pop('dynamic_heads', None)
        self.aligner: Union[str, dict] = self._pop('aligner', 'legacy')

    def to_non_vanilla(self):
        if self.extra_models:
            warnings.warn('``extra_models`` is only supported for vanilla Whisper models')
        if self.dynamic_heads:
            warnings.warn('``dynamic_heads`` is only supported for vanilla Whisper models')

import warnings
import importlib.metadata
from importlib.util import find_spec
from contextlib import contextmanager
from .utils import get_func_parameters, exact_div

IS_WHISPERLESS_VERSION = True
IS_WHISPER_AVAILABLE = find_spec('whisper') is not None


_COMPATIBLE_WHISPER_VERSIONS = (
    '20230314',
    '20230918',
    '20231105',
    '20231106',
    '20231117',
    '20240927',
    '20240930',
)
_required_whisper_ver = _COMPATIBLE_WHISPER_VERSIONS[-1]

_faster_compatibility = {}

if IS_WHISPER_AVAILABLE:
    import whisper.tokenizer
    _TOKENIZER_PARAMS = get_func_parameters(whisper.tokenizer.get_tokenizer)
else:
    _TOKENIZER_PARAMS = ()


def is_faster_whisper_on_pt() -> bool:
    if 'is_on_pt' not in _faster_compatibility:
        try:
            requirements = importlib.metadata.distribution('faster-whisper').requires
        except importlib.metadata.PackageNotFoundError:
            _faster_compatibility["is_on_pt"] = False
        else:
            _faster_compatibility["is_on_pt"] = any(r.startswith('torch') for r in requirements)
    return _faster_compatibility["is_on_pt"]


def whisper_not_available(*args, **kwargs):
    raise ModuleNotFoundError("Please install Whisper: "
                              "'pip install -U openai-whisper'. "
                              "Official Whisper repo: https://github.com/openai/whisper")


class Unavailable:
    __init__ = whisper_not_available


@contextmanager
def _dummy_contextmanager():
    yield


if IS_WHISPER_AVAILABLE:
    import whisper
    from whisper.audio import (
        SAMPLE_RATE, N_FRAMES, HOP_LENGTH, N_SAMPLES_PER_TOKEN, N_SAMPLES,
        TOKENS_PER_SECOND, N_FFT, FRAMES_PER_SECOND, CHUNK_LENGTH
    )
    from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE

    from whisper.audio import log_mel_spectrogram, pad_or_trim
    from whisper.timing import median_filter, dtw, merge_punctuations
    from whisper.tokenizer import get_tokenizer as get_whisper_tokenizer

    from whisper.tokenizer import Tokenizer
    from whisper.model import Whisper
    from whisper.decoding import DecodingTask, DecodingOptions, DecodingResult, SuppressTokens
    try:
        from whisper.model import disable_sdpa
    except ImportError:
        disable_sdpa = _dummy_contextmanager
else:
    import torch
    import numpy as np
    whisper = None
    # hard-coded audio hyperparameters from Whisper
    SAMPLE_RATE = 16000
    N_FFT = 400
    HOP_LENGTH = 160
    CHUNK_LENGTH = 30
    N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
    N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input
    N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
    FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
    TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token

    log_mel_spectrogram = median_filter = dtw = merge_punctuations = get_whisper_tokenizer \
        = whisper_not_available
    Tokenizer = Whisper = DecodingTask = DecodingOptions = DecodingResult = SuppressTokens = Unavailable
    LANGUAGES = {
        "en": "english",
        "zh": "chinese",
        "de": "german",
        "es": "spanish",
        "ru": "russian",
        "ko": "korean",
        "fr": "french",
        "ja": "japanese",
        "pt": "portuguese",
        "tr": "turkish",
        "pl": "polish",
        "ca": "catalan",
        "nl": "dutch",
        "ar": "arabic",
        "sv": "swedish",
        "it": "italian",
        "id": "indonesian",
        "hi": "hindi",
        "fi": "finnish",
        "vi": "vietnamese",
        "he": "hebrew",
        "uk": "ukrainian",
        "el": "greek",
        "ms": "malay",
        "cs": "czech",
        "ro": "romanian",
        "da": "danish",
        "hu": "hungarian",
        "ta": "tamil",
        "no": "norwegian",
        "th": "thai",
        "ur": "urdu",
        "hr": "croatian",
        "bg": "bulgarian",
        "lt": "lithuanian",
        "la": "latin",
        "mi": "maori",
        "ml": "malayalam",
        "cy": "welsh",
        "sk": "slovak",
        "te": "telugu",
        "fa": "persian",
        "lv": "latvian",
        "bn": "bengali",
        "sr": "serbian",
        "az": "azerbaijani",
        "sl": "slovenian",
        "kn": "kannada",
        "et": "estonian",
        "mk": "macedonian",
        "br": "breton",
        "eu": "basque",
        "is": "icelandic",
        "hy": "armenian",
        "ne": "nepali",
        "mn": "mongolian",
        "bs": "bosnian",
        "kk": "kazakh",
        "sq": "albanian",
        "sw": "swahili",
        "gl": "galician",
        "mr": "marathi",
        "pa": "punjabi",
        "si": "sinhala",
        "km": "khmer",
        "sn": "shona",
        "yo": "yoruba",
        "so": "somali",
        "af": "afrikaans",
        "oc": "occitan",
        "ka": "georgian",
        "be": "belarusian",
        "tg": "tajik",
        "sd": "sindhi",
        "gu": "gujarati",
        "am": "amharic",
        "yi": "yiddish",
        "lo": "lao",
        "uz": "uzbek",
        "fo": "faroese",
        "ht": "haitian creole",
        "ps": "pashto",
        "tk": "turkmen",
        "nn": "nynorsk",
        "mt": "maltese",
        "sa": "sanskrit",
        "lb": "luxembourgish",
        "my": "myanmar",
        "bo": "tibetan",
        "tl": "tagalog",
        "mg": "malagasy",
        "as": "assamese",
        "tt": "tatar",
        "haw": "hawaiian",
        "ln": "lingala",
        "ha": "hausa",
        "ba": "bashkir",
        "jw": "javanese",
        "su": "sundanese",
        "yue": "cantonese",
    }

    # language code lookup by name, with a few language aliases
    TO_LANGUAGE_CODE = {
        **{language: code for code, language in LANGUAGES.items()},
        "burmese": "my",
        "valencian": "ca",
        "flemish": "nl",
        "haitian": "ht",
        "letzeburgesch": "lb",
        "pushto": "ps",
        "panjabi": "pa",
        "moldavian": "ro",
        "moldovan": "ro",
        "sinhalese": "si",
        "castilian": "es",
        "mandarin": "zh",
    }


    def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
        """
        Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
        """
        if torch.is_tensor(array):
            if array.shape[axis] > length:
                array = array.index_select(
                    dim=axis, index=torch.arange(length, device=array.device)
                )

            if array.shape[axis] < length:
                pad_widths = [(0, 0)] * array.ndim
                pad_widths[axis] = (0, length - array.shape[axis])
                array = torch.nn.functional.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
        else:
            if array.shape[axis] > length:
                array = array.take(indices=range(length), axis=axis)

            if array.shape[axis] < length:
                pad_widths = [(0, 0)] * array.ndim
                pad_widths[axis] = (0, length - array.shape[axis])
                array = np.pad(array, pad_widths)

        return array

    disable_sdpa = _dummy_contextmanager


def warn_compatibility_issues(
        whisper_module,
        ignore: bool = False,
        additional_msg: str = ''
):
    compatibility_warning = ''
    if not ignore:
        if whisper_module.__version__ not in _COMPATIBLE_WHISPER_VERSIONS:
            compatibility_warning += (f'Whisper {whisper_module.__version__} is installed. '
                                      f'Compatible versions: {", ".join(_COMPATIBLE_WHISPER_VERSIONS)}\n')
        _is_whisper_repo_version = bool(importlib.metadata.distribution('openai-whisper').read_text('direct_url.json'))
        if _is_whisper_repo_version:
            compatibility_warning += ('The detected version appears to be installed from the repository '
                                      'which can have compatibility issues '
                                      'due to multiple commits sharing the same version number. '
                                      f'It is recommended to install version {_required_whisper_ver} from PyPI.\n')

        if compatibility_warning:
            compatibility_warning = (
                    'The installed version of Whisper might be incompatible.\n'
                    + compatibility_warning +
                    'To prevent errors and performance issues, install the latest compatible version: '
                    f'`pip install openai-whisper=={_COMPATIBLE_WHISPER_VERSIONS[-1]}` \n'
                    f'Use `ignore_compatibility=True` to ignore this warning.'
            )
            if additional_msg:
                compatibility_warning += f' {additional_msg}'
            warnings.warn(compatibility_warning)


def get_valid_language(language: str, is_faster_model: bool, model=None):
    if language is None:
        if model is None:
            return language
        if is_faster_model:
            return model.supported_languages[0] if len(model.supported_languages) == 1 else language
        return language if model.is_multilingual else 'en'

    if is_faster_model:
        from faster_whisper.tokenizer import _LANGUAGE_CODES
        if language in _LANGUAGE_CODES:
            return language
        faster_language_code_lower = {code.lower(): code for code in _LANGUAGE_CODES}
        if language.lower() in faster_language_code_lower:
            return faster_language_code_lower[language.lower()]
        for k, v in LANGUAGES.items():
            if v.lower() == language.lower() and k.lower() in faster_language_code_lower:
                return faster_language_code_lower[k.lower()]

        raise ValueError(f'{language} is not a valid language or language code. '
                         f'Available languages: {tuple(_LANGUAGE_CODES.keys())}')
    else:
        if language in LANGUAGES:
            return language
        language_codes_lower = {code.lower(): code for code in LANGUAGES}
        if language.lower() in language_codes_lower:
            return language_codes_lower[language.lower()]
        for k, v in LANGUAGES.items():
            if v.lower() == language.lower():
                return k
        raise ValueError(f'{language} is not a valid language or language code. '
                         f'Available languages: {tuple(LANGUAGES.keys())}')


def get_tokenizer(model=None, is_faster_model: bool = False, **kwargs):
    """
    Backward compatible wrapper of :func:`whisper.tokenizer.get_tokenizer` and
    :class:`faster_whisper.tokenizer.Tokenizer`.
    """
    if is_faster_model:
        import faster_whisper.tokenizer
        from faster_whisper.tokenizer import _LANGUAGE_CODES
        tokenizer = faster_whisper.tokenizer.Tokenizer
        params = get_func_parameters(tokenizer)
        if model is not None and 'tokenizer' not in kwargs:
            kwargs['tokenizer'] = model.hf_tokenizer
    else:
        tokenizer = whisper.tokenizer.get_tokenizer
        params = _TOKENIZER_PARAMS
    if model is not None and 'multilingual' not in kwargs:
        kwargs['multilingual'] = \
            (model.is_multilingual if hasattr(model, 'is_multilingual') else model.model.is_multilingual)
    if 'num_languages' in params:
        if hasattr(model, 'num_languages'):
            kwargs['num_languages'] = \
                (model.num_languages if hasattr(model, 'num_languages') else model.model.num_languages)
    elif 'num_languages' in kwargs:
        del kwargs['num_languages']
    kwargs['language'] = get_valid_language(kwargs.get('language'), is_faster_model, model)
    return tokenizer(**kwargs)

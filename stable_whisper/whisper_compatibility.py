import warnings
import importlib.metadata

import whisper.tokenizer

from .utils import get_func_parameters

_COMPATIBLE_WHISPER_VERSIONS = (
    '20230314',
    '20230918',
    '20231105',
    '20231106'
)
_required_whisper_ver = _COMPATIBLE_WHISPER_VERSIONS[-1]

_TOKENIZER_PARAMS = get_func_parameters(whisper.tokenizer.get_tokenizer)


def warn_compatibility_issues(
        whisper_module,
        ignore: bool = False,
        additional_msg: str = ''
):
    compatibility_warning = ''
    if not ignore:
        if whisper_module.__version__ not in _COMPATIBLE_WHISPER_VERSIONS:
            compatibility_warning += (f'Whisper {whisper_module.__version__} is installed.'
                                      f'Versions confirm to be compatible: {", ".join(_COMPATIBLE_WHISPER_VERSIONS)}\n')
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
                    'To prevent errors and performance issues, reinstall correct version with: '
                    f'"pip install --upgrade --no-deps --force-reinstall openai-whisper=={_required_whisper_ver}".'
            )
            if additional_msg:
                compatibility_warning += f' {additional_msg}'
            warnings.warn(compatibility_warning)


def get_tokenizer(model=None, is_faster_model: bool = False, **kwargs):
    """
    Backward compatible wrapper of :func:`whisper.tokenizer.get_tokenizer` and
    :class:`faster_whisper.tokenizer.Tokenizer`.
    """
    if is_faster_model:
        import faster_whisper.tokenizer
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
    return tokenizer(**kwargs)


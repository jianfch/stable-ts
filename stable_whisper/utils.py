import warnings
import importlib.metadata
import inspect

import whisper

_COMPATIBLE_WHISPER_VERSIONS = (
    '20230314',
    '20230918'
)
_required_whisper_ver = _COMPATIBLE_WHISPER_VERSIONS[-1]
_is_whisper_repo_version = bool(importlib.metadata.distribution('openai-whisper').read_text('direct_url.json'))


def warn_compatibility_issues(
        ignore: bool = False,
        additional_msg: str = ''
):
    compatibility_warning = ''
    if not ignore:
        if whisper.__version__ not in _COMPATIBLE_WHISPER_VERSIONS:
            compatibility_warning += (f'Whisper {whisper.__version__} is installed.'
                                      f'Versions confirm to be compatible: {", ".join(_COMPATIBLE_WHISPER_VERSIONS)}\n')
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


def str_to_valid_type(val: str):
    if len(val) == 0:
        return None
    if '/' in val:
        return [a.split('*') if '*' in a else a for a in val.split('/')]
    try:
        val = float(val) if '.' in val else int(val)
    except ValueError:
        pass
    finally:
        return val


def get_func_parameters(func):
    return inspect.signature(func).parameters.keys()


def isolate_useful_options(options: dict, method, pop: bool = False) -> dict:
    _get = dict.pop if pop else dict.get
    return {k: _get(options, k) for k in get_func_parameters(method) if k in options}

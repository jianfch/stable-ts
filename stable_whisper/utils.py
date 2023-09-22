import warnings
import importlib.metadata

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

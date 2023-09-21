import warnings
import re
import importlib.metadata

import whisper

_required_whisper_ver = re.findall(
    r'=(\d+)',
    next(filter(lambda x: x.startswith('openai-whisper'), importlib.metadata.distribution('stable-ts').requires))
)[0]
_is_whisper_repo_version = bool(importlib.metadata.distribution('openai-whisper').read_text('direct_url.json'))


def warn_compatibility_issues(
        ignore: bool = False,
        additional_msg: str = ''
):
    compatibility_warning = ''
    if not ignore:
        if _required_whisper_ver != whisper.__version__:
            compatibility_warning += (f'Version {_required_whisper_ver} is required '
                                      f'but {whisper.__version__} is installed.\n')
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

import os
from typing import Optional

DEFAULT_KWARGS = dict(
    prepend_punctuations="\"'“¿([{-「",
    append_punctuations="\"'.。,，!！?？:：”)]}、」",
    min_word_dur=0.1
)

permissions = {}

cached_model_instances = dict(
    demucs={
        'htdemucs': None
    },
    silero_vad={
        True: None,
        False: None
    },
    dfnet={
        'dfnet': None
    }
)

__all__ = [
    'get_prepend_punctuations',
    'get_append_punctuations',
    'get_min_word_dur',
    'is_allow_overwrite',
    'set_global_overwrite_permission',
    'cached_model_instances'
]


def has_key(key: str):
    if key not in DEFAULT_KWARGS:
        raise KeyError(f'the key, {key}, not found in DEFAULT_VALUES')
    return True


def set_val(key: str, val):
    has_key(key)
    DEFAULT_KWARGS[key] = val


def get_val(key: str, default=None):
    if default is None:
        has_key(key)
        return DEFAULT_KWARGS[key]
    return default


def set_get_val(key: str, new_val=None):
    if new_val is not None:
        set_val(key, new_val)
    return get_val(key)


def get_prepend_punctuations(default: Optional[str] = None) -> str:
    return get_val('prepend_punctuations', default)


def get_append_punctuations(default: Optional[str] = None) -> str:
    return get_val('append_punctuations', default)


def get_min_word_dur(default: Optional[float] = None) -> float:
    return get_val('min_word_dur', default)


def is_allow_overwrite(filepath: str, default: (bool, None) = None) -> bool:
    if default is not None:
        return default
    if not os.path.isfile(filepath) or permissions.get('overwrite'):
        return True
    resp = input(f'"{filepath}" already exist, overwrite (y/n)? ').lower()
    if resp in ('y', 'n'):
        return resp == 'y'
    print(f'Expected "y" or "n", but got {resp}.')
    return is_allow_overwrite(filepath)


def set_global_overwrite_permission(overwrite: bool):
    permissions['overwrite'] = overwrite

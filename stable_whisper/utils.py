import inspect
import sys

system_encoding = sys.getdefaultencoding()

if system_encoding != "utf-8":

    def make_safe(string):
        # replaces any character not representable using the system default encoding with an '?',
        # avoiding UnicodeEncodeError (https://github.com/openai/whisper/discussions/729).
        return string.encode(system_encoding, errors="replace").decode(system_encoding)

else:

    def make_safe(string):
        # utf-8 can encode any Unicode code point, so no need to do the round-trip encoding
        return string


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


def safe_print(msg: str, _print=None):
    if msg:
        (_print or print)(make_safe(msg))


def format_timestamp(
    seconds: float, always_include_hours: bool = False, decimal_marker: str = "."
):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


def exact_div(x, y):
    assert x % y == 0
    return x // y


class UnsortedException(Exception):

    def __init__(self, message: str = None, data: dict = None):
        if not message:
            message = 'Timestamps are not in ascending order. If data is produced by Stable-ts, please submit an issue.'
        super().__init__(message)
        self.data = data

    def get_data(self):
        return self.data


def update_options(_options: dict = None, _reversed: bool = False, **kwargs) -> dict:
    if _options is None:
        _options = {}
    if _reversed:
        _options, kwargs = kwargs, _options
    _options.update(kwargs)
    return _options

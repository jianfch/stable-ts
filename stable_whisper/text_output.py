import json
import os
import warnings
from typing import List, Tuple, Union, Callable
from itertools import chain
from .stabilization.utils import valid_ts

__all__ = ['result_to_srt_vtt', 'result_to_ass', 'result_to_tsv', 'result_to_txt', 'save_as_json', 'load_result']
SUPPORTED_FORMATS = ('srt', 'vtt', 'ass', 'tsv', 'txt')


def _save_as_file(content: str, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'Saved: {os.path.abspath(path)}')


def _get_segments(result: (dict, list), min_dur: float, reverse_text: Union[bool, tuple] = False):
    if isinstance(result, dict):
        if reverse_text:
            warnings.warn(f'``reverse_text=True`` only applies to WhisperResult but result is {type(result)}')
        return result.get('segments')
    elif not isinstance(result, list) and callable(getattr(result, 'segments_to_dicts', None)):
        return result.apply_min_dur(min_dur, inplace=False).segments_to_dicts(reverse_text=reverse_text)
    return result


def finalize_text(text: str, strip: bool = True):
    if not strip:
        return text
    return text.strip().replace('\n ', '\n')


def sec2hhmmss(seconds: (float, int)):
    mm, ss = divmod(seconds, 60)
    hh, mm = divmod(mm, 60)
    return hh, mm, ss


def sec2milliseconds(seconds: (float, int)) -> int:
    return round(seconds * 1000)


def sec2centiseconds(seconds: (float, int)) -> int:
    return round(seconds * 100)


def sec2vtt(seconds: (float, int)) -> str:
    hh, mm, ss = sec2hhmmss(seconds)
    return f'{hh:0>2.0f}:{mm:0>2.0f}:{ss:0>6.3f}'


def sec2srt(seconds: (float, int)) -> str:
    return sec2vtt(seconds).replace(".", ",")


def sec2ass(seconds: (float, int)) -> str:
    hh, mm, ss = sec2hhmmss(seconds)
    return f'{hh:0>1.0f}:{mm:0>2.0f}:{ss:0>2.2f}'


def segment2vttblock(segment: dict, strip=True) -> str:
    return f'{sec2vtt(segment["start"])} --> {sec2vtt(segment["end"])}\n' \
           f'{finalize_text(segment["text"], strip)}'


def segment2srtblock(segment: dict, idx: int, strip=True) -> str:
    return f'{idx}\n{sec2srt(segment["start"])} --> {sec2srt(segment["end"])}\n' \
           f'{finalize_text(segment["text"], strip)}'


def segment2assblock(segment: dict, idx: int, strip=True) -> str:
    return f'Dialogue: {idx},{sec2ass(segment["start"])},{sec2ass(segment["end"])},Default,,0,0,0,,' \
           f'{finalize_text(segment["text"], strip)}'


def segment2tsvblock(segment: dict, strip=True) -> str:
    return f'{sec2milliseconds(segment["start"])}' \
           f'\t{sec2milliseconds(segment["end"])}' \
           f'\t{segment["text"].strip() if strip else segment["text"]}'


def words2segments(words: List[dict], tag: Tuple[str, str], reverse_text: bool = False) -> List[dict]:
    def add_tag(idx: int):
        return ''.join(
            (
                f" {tag[0]}{w['word'][1:]}{tag[1]}"
                if w['word'].startswith(' ') else
                f"{tag[0]}{w['word']}{tag[1]}"
            )
            if w['word'] not in ('', ' ') and idx_ == idx else
            w['word']
            for idx_, w in idx_filled_words
        )

    filled_words = []
    for i, word in enumerate(words):
        curr_end = round(word['end'], 3)
        filled_words.append(dict(word=word['word'], start=round(word['start'], 3), end=curr_end))
        if word != words[-1]:
            next_start = round(words[i + 1]['start'], 3)
            if next_start - curr_end != 0:
                filled_words.append(dict(word='', start=curr_end, end=next_start))
    idx_filled_words = list(enumerate(filled_words))
    if reverse_text:
        idx_filled_words = list(reversed(idx_filled_words))

    segments = [dict(text=add_tag(i), start=filled_words[i]['start'], end=filled_words[i]['end'])
                for i in range(len(filled_words))]
    return segments


def to_word_level_segments(segments: List[dict], tag: Tuple[str, str]) -> List[dict]:
    return list(
        chain.from_iterable(
            words2segments(s['words'], tag, reverse_text=s.get('reversed_text'))
            for s in segments
        )
    )


def to_vtt_word_level_segments(segments: List[dict], tag: Tuple[str, str] = None) -> List[dict]:
    def to_segment_string(segment: dict):
        segment_string = ''
        prev_end = 0
        for i, word in enumerate(segment['words']):
            if i != 0:
                curr_start = word['start']
                if prev_end == curr_start:
                    segment_string += f"<{sec2vtt(curr_start)}>"
                else:
                    if segment_string.endswith(' '):
                        segment_string = segment_string[:-1]
                    elif segment['words'][i]['word'].startswith(' '):
                        segment['words'][i]['word'] = segment['words'][i]['word'][1:]
                    segment_string += f"<{sec2vtt(prev_end)}> <{sec2vtt(curr_start)}>"
            segment_string += word['word']
            prev_end = word['end']
        return segment_string

    return [
        dict(
            text=to_segment_string(s),
            start=s['start'],
            end=s['end']
        )
        for s in segments
    ]


def to_ass_word_level_segments(segments: List[dict], tag: Tuple[str, str], karaoke: bool = False) -> List[dict]:

    def to_segment_string(segment: dict):
        segment_string = ''
        for i, word in enumerate(segment['words']):
            curr_word, space = (word['word'][1:], " ") if word['word'].startswith(" ") else (word['word'], "")
            segment_string += (
                    space +
                    r"{\k" +
                    ("f" if karaoke else "") +
                    f"{sec2centiseconds(word['end']-word['start'])}" +
                    r"}" +
                    curr_word
            )
        return segment_string

    return [
        dict(
            text=to_segment_string(s),
            start=s['start'],
            end=s['end']
        )
        for s in segments
    ]


def to_word_level(segments: List[dict]) -> List[dict]:
    return [dict(text=w['word'], start=w['start'], end=w['end']) for s in segments for w in s['words']]


def _confirm_word_level(segments: List[dict]) -> bool:
    if not all(bool(s.get('words')) for s in segments):
        warnings.warn('Result is missing word timestamps. Word-level timing cannot be exported. '
                      'Use ``word_level=False`` to avoid this warning')
        return False
    return True


def _preprocess_args(result: (dict, list),
                     segment_level: bool,
                     word_level: bool,
                     min_dur: float,
                     reverse_text: Union[bool, tuple] = False):
    assert segment_level or word_level, '`segment_level` or `word_level` must be True'
    segments = _get_segments(result, min_dur, reverse_text=reverse_text)
    if word_level:
        word_level = _confirm_word_level(segments)
    return segments, segment_level, word_level


def result_to_any(result: (dict, list),
                  filepath: str = None,
                  filetype: str = None,
                  segments2blocks: Callable = None,
                  segment_level=True,
                  word_level=True,
                  min_dur: float = 0.02,
                  tag: Tuple[str, str] = None,
                  default_tag: Tuple[str, str] = None,
                  strip=True,
                  reverse_text: Union[bool, tuple] = False,
                  to_word_level_string_callback: Callable = None):
    """
    Generate file from ``result`` to display segment-level and/or word-level timestamp.

    Returns
    -------
    str
        String of the content if ``filepath`` is ``None``.
    """
    segments, segment_level, word_level = _preprocess_args(
        result, segment_level, word_level, min_dur, reverse_text=reverse_text
    )

    if filetype is None:
        filetype = os.path.splitext(filepath)[-1][1:] or 'srt'
    if filetype.lower() not in SUPPORTED_FORMATS:
        raise NotImplementedError(f'{filetype} not supported')
    if filepath and not filepath.lower().endswith(f'.{filetype}'):
        filepath += f'.{filetype}'

    if word_level and segment_level:
        if tag is None:
            if default_tag is None:
                tag = ('<font color="#00ff00">', '</font>') if filetype == 'srt' else ('<u>', '</u>')
            else:
                tag = default_tag
        if to_word_level_string_callback is None:
            to_word_level_string_callback = to_word_level_segments
        segments = to_word_level_string_callback(segments, tag)
    elif word_level:
        segments = to_word_level(segments)

    valid_ts(segments)

    if segments2blocks is None:
        sub_str = '\n\n'.join(segment2srtblock(s, i, strip=strip) for i, s in enumerate(segments))
    else:
        sub_str = segments2blocks(segments)

    if filepath:
        _save_as_file(sub_str, filepath)
    else:
        return sub_str


def result_to_srt_vtt(result: (dict, list),
                      filepath: str = None,
                      segment_level=True,
                      word_level=True,
                      min_dur: float = 0.02,
                      tag: Tuple[str, str] = None,
                      vtt: bool = None,
                      strip=True,
                      reverse_text: Union[bool, tuple] = False):
    """
    Generate SRT/VTT from ``result`` to display segment-level and/or word-level timestamp.

    Parameters
    ----------
    result : dict or list or stable_whisper.result.WhisperResult
        Result of transcription.
    filepath : str, default None, meaning content will be returned as a ``str``
        Path to save file.
    segment_level : bool, default True
        Whether to use segment-level timestamps in output.
    word_level : bool, default True
        Whether to use word-level timestamps in output.
    min_dur : float, default 0.2
        Minimum duration allowed for any word/segment before the word/segments are merged with adjacent word/segments.
    tag: tuple of (str, str), default None, meaning ('<font color="#00ff00">', '</font>') if SRT else ('<u>', '</u>')
        Tag used to change the properties a word at its timestamp.
    vtt : bool, default None, meaning determined by extension of ``filepath`` or ``False`` if no valid extension.
        Whether to output VTT.
    strip : bool, default True
        Whether to remove spaces before and after text on each segment for output.
    reverse_text: bool or tuple, default False
        Whether to reverse the order of words for each segment or provide the ``prepend_punctuations`` and
        ``append_punctuations`` as tuple pair instead of ``True`` which is for the default punctuations.

    Returns
    -------
    str
        String of the content if ``filepath`` is ``None``.

    Notes
    -----
    ``reverse_text`` will not fix RTL text not displaying tags properly which is an issue with some video player. VLC
    seems to not suffer from this issue.

    Examples
    --------
    >>> import stable_whisper
    >>> model = stable_whisper.load_model('base')
    >>> result = model.transcribe('audio.mp3')
    >>> result.to_srt_vtt('audio.srt')
    Saved: audio.srt
    """
    is_srt = (filepath is None or not filepath.lower().endswith('.vtt')) if vtt is None else not vtt
    if is_srt:
        segments2blocks = None
        to_word_level_string_callback = None
    else:
        def segments2blocks(segments):
            return 'WEBVTT\n\n' + '\n\n'.join(segment2vttblock(s, strip=strip) for i, s in enumerate(segments))
        to_word_level_string_callback = to_vtt_word_level_segments if tag is None else tag

    return result_to_any(
        result=result,
        filepath=filepath,
        filetype=('vtt', 'srt')[is_srt],
        segments2blocks=segments2blocks,
        segment_level=segment_level,
        word_level=word_level,
        min_dur=min_dur,
        tag=tag,
        strip=strip,
        reverse_text=reverse_text,
        to_word_level_string_callback=to_word_level_string_callback
    )


def result_to_tsv(result: (dict, list),
                  filepath: str = None,
                  segment_level: bool = None,
                  word_level: bool = None,
                  min_dur: float = 0.02,
                  strip=True,
                  reverse_text: Union[bool, tuple] = False):
    """
    Generate TSV from ``result`` to display segment-level and/or word-level timestamp.

    Parameters
    ----------
    result : dict or list or stable_whisper.result.WhisperResult
        Result of transcription.
    filepath : str, default None, meaning content will be returned as a ``str``
        Path to save file.
    segment_level : bool, default True
        Whether to use segment-level timestamps in output.
    word_level : bool, default True
        Whether to use word-level timestamps in output.
    min_dur : float, default 0.2
        Minimum duration allowed for any word/segment before the word/segments are merged with adjacent word/segments.
    strip : bool, default True
        Whether to remove spaces before and after text on each segment for output.
    reverse_text: bool or tuple, default False
        Whether to reverse the order of words for each segment or provide the ``prepend_punctuations`` and
        ``append_punctuations`` as tuple pair instead of ``True`` which is for the default punctuations.

    Returns
    -------
    str
        String of the content if ``filepath`` is ``None``.

    Notes
    -----
    ``reverse_text`` will not fix RTL text not displaying tags properly which is an issue with some video player. VLC
    seems to not suffer from this issue.

    Examples
    --------
    >>> import stable_whisper
    >>> model = stable_whisper.load_model('base')
    >>> result = model.transcribe('audio.mp3')
    >>> result.to_tsv('audio.tsv')
    Saved: audio.tsv
    """
    if segment_level is None and word_level is None:
        segment_level = True
    assert word_level is not segment_level, '[word_level] and [segment_level] cannot be the same ' \
                                            'since [tag] is not support for this format'

    def segments2blocks(segments):
        return '\n\n'.join(segment2tsvblock(s, strip=strip) for i, s in enumerate(segments))
    return result_to_any(
        result=result,
        filepath=filepath,
        filetype='tsv',
        segments2blocks=segments2blocks,
        segment_level=segment_level,
        word_level=word_level,
        min_dur=min_dur,
        strip=strip,
        reverse_text=reverse_text
    )


def result_to_ass(result: (dict, list),
                  filepath: str = None,
                  segment_level=True,
                  word_level=True,
                  min_dur: float = 0.02,
                  tag: Union[Tuple[str, str], int] = None,
                  font: str = None,
                  font_size: int = 24,
                  strip=True,
                  highlight_color: str = None,
                  karaoke=False,
                  reverse_text: Union[bool, tuple] = False,
                  **kwargs):
    """
    Generate Advanced SubStation Alpha (ASS) file from ``result`` to display segment-level and/or word-level timestamp.

    Parameters
    ----------
    result : dict or list or stable_whisper.result.WhisperResult
        Result of transcription.
    filepath : str, default None, meaning content will be returned as a ``str``
        Path to save file.
    segment_level : bool, default True
        Whether to use segment-level timestamps in output.
    word_level : bool, default True
        Whether to use word-level timestamps in output.
    min_dur : float, default 0.2
        Minimum duration allowed for any word/segment before the word/segments are merged with adjacent word/segments.
    tag: tuple of (str, str) or int, default None, meaning use default highlighting
        Tag used to change the properties a word at its timestamp. -1 for individual word highlight tag.
    font : str, default `Arial`
        Word font.
    font_size : int, default 48
        Word font size.
    strip : bool, default True
        Whether to remove spaces before and after text on each segment for output.
    highlight_color : str, default '00ff00'
        Hexadecimal of the color use for default highlights as '<bb><gg><rr>'.
    karaoke : bool, default False
        Whether to use progressive filling highlights (for karaoke effect).
    reverse_text: bool or tuple, default False
        Whether to reverse the order of words for each segment or provide the ``prepend_punctuations`` and
        ``append_punctuations`` as tuple pair instead of ``True`` which is for the default punctuations.
    kwargs:
        Format styles:
        'Name', 'Fontname', 'Fontsize', 'PrimaryColour', 'SecondaryColour', 'OutlineColour', 'BackColour', 'Bold',
        'Italic', 'Underline', 'StrikeOut', 'ScaleX', 'ScaleY', 'Spacing', 'Angle', 'BorderStyle', 'Outline',
        'Shadow', 'Alignment', 'MarginL', 'MarginR', 'MarginV', 'Encoding'

    Returns
    -------
    str
        String of the content if ``filepath`` is ``None``.

    Notes
    -----
    ``reverse_text`` will not fix RTL text not displaying tags properly which is an issue with some video player. VLC
    seems to not suffer from this issue.

    Examples
    --------
    >>> import stable_whisper
    >>> model = stable_whisper.load_model('base')
    >>> result = model.transcribe('audio.mp3')
    >>> result.to_ass('audio.ass')
    Saved: audio.ass
    """
    if tag == ['-1']:  # CLI
        tag = -1
    if highlight_color is None:
        highlight_color = '00ff00'

    def segments2blocks(segments):
        fmt_style_dict = {'Name': 'Default', 'Fontname': 'Arial', 'Fontsize': '48', 'PrimaryColour': '&Hffffff',
                          'SecondaryColour': '&Hffffff', 'OutlineColour': '&H0', 'BackColour': '&H0', 'Bold': '0',
                          'Italic': '0', 'Underline': '0', 'StrikeOut': '0', 'ScaleX': '100', 'ScaleY': '100',
                          'Spacing': '0', 'Angle': '0', 'BorderStyle': '1', 'Outline': '1', 'Shadow': '0',
                          'Alignment': '2', 'MarginL': '10', 'MarginR': '10', 'MarginV': '10', 'Encoding': '0'}

        for k, v in filter(lambda x: 'colour' in x[0].lower() and not str(x[1]).startswith('&H'), kwargs.items()):
            kwargs[k] = f'&H{kwargs[k]}'

        fmt_style_dict.update((k, v) for k, v in kwargs.items() if k in fmt_style_dict)

        if tag is None and 'PrimaryColour' not in kwargs:
            fmt_style_dict['PrimaryColour'] = \
                highlight_color if highlight_color.startswith('&H') else f'&H{highlight_color}'

        if font:
            fmt_style_dict.update(Fontname=font)
        if font_size:
            fmt_style_dict.update(Fontsize=font_size)

        fmts = f'Format: {", ".join(map(str, fmt_style_dict.keys()))}'

        styles = f'Style: {",".join(map(str, fmt_style_dict.values()))}'

        sub_str = f'[Script Info]\nScriptType: v4.00+\nPlayResX: 384\nPlayResY: 288\nScaledBorderAndShadow: yes\n\n' \
                  f'[V4+ Styles]\n{fmts}\n{styles}\n\n' \
                  f'[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n\n'

        sub_str += '\n'.join(segment2assblock(s, i, strip=strip) for i, s in enumerate(segments))

        return sub_str

    if tag is not None and karaoke:
        warnings.warn(f'``tag`` is not support for ``karaoke=True``; ``tag`` will be ignored.')

    return result_to_any(
        result=result,
        filepath=filepath,
        filetype='ass',
        segments2blocks=segments2blocks,
        segment_level=segment_level,
        word_level=word_level,
        min_dur=min_dur,
        tag=None if tag == -1 else tag,
        default_tag=(r'{\1c' + f'{highlight_color}&' + '}', r'{\r}'),
        strip=strip,
        reverse_text=reverse_text,
        to_word_level_string_callback=(
            (lambda s, t: to_ass_word_level_segments(s, t, karaoke=karaoke))
            if karaoke or (word_level and segment_level and tag is None)
            else None
        )
    )


def result_to_txt(
        result: (dict, list),
        filepath: str = None,
        min_dur: float = 0.02,
        strip=True,
        reverse_text: Union[bool, tuple] = False
):
    """
    Generate plain-text without timestamps from ``result``.

    Parameters
    ----------
    result : dict or list or stable_whisper.result.WhisperResult
        Result of transcription.
    filepath : str, default None, meaning content will be returned as a ``str``
        Path to save file.
    min_dur : float, default 0.2
        Minimum duration allowed for any word/segment before the word/segments are merged with adjacent word/segments.
    strip : bool, default True
        Whether to remove spaces before and after text on each segment for output.
    reverse_text: bool or tuple, default False
        Whether to reverse the order of words for each segment or provide the ``prepend_punctuations`` and
        ``append_punctuations`` as tuple pair instead of ``True`` which is for the default punctuations.

    Returns
    -------
    str
        String of the content if ``filepath`` is ``None``.

    Notes
    -----
    ``reverse_text`` will not fix RTL text not displaying tags properly which is an issue with some video player. VLC
    seems to not suffer from this issue.

    Examples
    --------
    >>> import stable_whisper
    >>> model = stable_whisper.load_model('base')
    >>> result = model.transcribe('audio.mp3')
    >>> result.to_txt('audio.txt')
    Saved: audio.txt
    """

    def segments2blocks(segments: dict, _strip=True) -> str:
        return '\n'.join(f'{segment["text"].strip() if _strip else segment["text"]}' for segment in segments)

    return result_to_any(
        result=result,
        filepath=filepath,
        filetype='txt',
        segments2blocks=segments2blocks,
        segment_level=True,
        word_level=False,
        min_dur=min_dur,
        strip=strip,
        reverse_text=reverse_text
    )


def save_as_json(result: dict, path: str, ensure_ascii: bool = False, **kwargs):
    """
    Save ``result`` as JSON file to ``path``.

    Parameters
    ----------
    result : dict or list or stable_whisper.result.WhisperResult
        Result of transcription.
    path : str
        Path to save file.
    ensure_ascii : bool, default False
        Whether to escape non-ASCII characters.

    Examples
    --------
    >>> import stable_whisper
    >>> model = stable_whisper.load_model('base')
    >>> result = model.transcribe('audio.mp3')
    >>> result.save_as_json('audio.json')
    Saved: audio.json
    """
    if not isinstance(result, dict) and callable(getattr(result, 'to_dict')):
        result = result.to_dict()
    if not path.lower().endswith('.json'):
        path += '.json'
    result = json.dumps(result, allow_nan=True, ensure_ascii=ensure_ascii, **kwargs)
    _save_as_file(result, path)


def load_result(json_path: str) -> dict:
    """
    Return a ``dict`` of the contents in ``json_path``.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

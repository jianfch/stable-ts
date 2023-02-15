import json
import os
from typing import List
from itertools import groupby, chain
from stable_whisper.stabilization import group_word_timestamps, tighten_timestamps, MIN_DUR

__all__ = ['results_to_sentence_srt', 'results_to_word_srt', 'results_to_token_srt',
           'results_to_sentence_word_ass', 'to_srt', 'results_to_srt', 'save_as_json',
           'load_results', 'finalize_segment_word_ts']


def _save_as_file(content: str, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'Saved: {os.path.abspath(path)}')


def to_srt(lines: List[dict], save_path: str = None, strip=False) -> str:
    """
    lines: List[dict]
        [{start:<start-timestamp-of-text>, end:<end-timestamp-of-text>, text:<str-of-text>}, ...]
    """

    def secs_to_hhmmss(secs: (float, int)):
        mm, ss = divmod(secs, 60)
        hh, mm = divmod(mm, 60)
        return f'{hh:0>2.0f}:{mm:0>2.0f}:{ss:0>6.3f}'.replace(".", ",")

    srt_str = '\n'.join(
        f'{i}\n'
        f'{secs_to_hhmmss(sub["start"])} --> {secs_to_hhmmss(sub["end"])}\n'
        f'{sub["text"].strip() if strip else sub["text"]}\n'
        for i, sub in enumerate(lines, 1))

    if not save_path.endswith('.srt'):
        save_path += '.srt'

    if save_path:
        _save_as_file(srt_str, save_path)

    return srt_str


def results_to_srt(res: dict, srt_path, word_level=True, combine_compound=False,
                   end_at_last_word=False, end_before_period=False, start_at_first_word=False, strip=False):
    if word_level:
        results_to_word_srt(res, srt_path, combine_compound=combine_compound, strip=strip)
    else:
        results_to_sentence_srt(res, srt_path,
                                end_at_last_word=end_at_last_word,
                                end_before_period=end_before_period,
                                start_at_first_word=start_at_first_word,
                                strip=strip)


def finalize_segment_word_ts(res: (dict, list),
                             end_at_last_word=False,
                             end_before_period=False,
                             start_at_first_word=False,
                             combine_compound=False,
                             min_dur: float = None,
                             force_max_len: int = None,
                             strip=True,
                             ts_key: str = None,
                             ass_format: bool = False):
    """

    Parameters
    ----------
    res: dict
        results from modified model
    end_at_last_word: bool
        set end-of-segment to timestamp-of-last-token (Default: False)
    end_before_period: bool
        set end-of-segment to timestamp-of-last-non-period-token (Default: False)
    start_at_first_word: bool
        set start-of-segment to timestamp-of-first-token (Default: False)
    combine_compound: bool
        concatenate words without inbetween spacing (Default: False)
    min_dur: float
        minimum duration for each word (i.e. concat the word if it is less than specified value; Default 0.02)
        Note: it applies to token instead of word if [ts_key]='word_timestamps'
    force_max_len: int
        force a max number of characters per phrase. Ignored if None (Default: None)
        Note: character count is still allow to go under this number for stability reasons.
    strip: bool
        perform strip() on each segment (Default: True)
    ts_key : str
        key of the timestamps to finalize (Default: 'whole_word_timestamps')
    ass_format: bool
        keep output ready to be formatted into .ass (Default: False)
    """

    if min_dur is None:
        min_dur = MIN_DUR

    def split_extra_words(words_tss_: list):
        curr_words_len_ = 0
        word_tss_split = []
        for wts_ in words_tss_:
            curr_words_len_ += len(wts_['text'])
            if curr_words_len_ > force_max_len or not word_tss_split:
                if strip:
                    wts_['text'] = wts_['text'].strip()
                word_tss_split.append([wts_])
                curr_words_len_ = len(wts_['text'])
            else:
                word_tss_split[-1].append(wts_)
        return word_tss_split

    def merge_grouped_wtss(wtss0: List[dict], wtss1: List[dict]):
        last_idx = -2 if len(wtss0) >= 2 and wtss0[-1]['text'] == ' ' else -1
        if wtss0[last_idx]['end'] - wtss0[last_idx]['start'] < min_dur or wtss1[0]['end'] - wtss1[0]['start'] < min_dur:
            wtss0[last_idx]['end'] = wtss1[0]['end']
            wtss0[last_idx]['text'] += wtss1[0]['text']
            wtss = (wtss0 + wtss1[1:]) if last_idx == -1 else (wtss0[:-1] + wtss1[1:])
        else:
            wtss = wtss0 + wtss1
        return wtss
    segments = tighten_timestamps(res,
                                  end_at_last_word=end_at_last_word,
                                  end_before_period=end_before_period,
                                  start_at_first_word=start_at_first_word)['segments']

    prev_extra_word_tss = []
    word_tss_ls = []

    for seg_i, seg in enumerate(segments):

        seg_grouped = group_word_timestamps([seg], combine_compound=combine_compound, min_dur=min_dur,
                                            ts_key=ts_key)

        merge_prev = False
        if prev_extra_word_tss:
            if prev_extra_word_tss[-1]['end'] is None:
                prev_extra_word_tss[-1]['end'] = seg['start']
            word_timestamps = merge_grouped_wtss(prev_extra_word_tss, seg_grouped)
            if word_tss_ls and (word_tss_ls[-1][-1]['end'] - word_tss_ls[-1][0]['start'] < min_dur):
                word_timestamps = merge_grouped_wtss(word_tss_ls[-1], word_timestamps)
                merge_prev = True
            prev_extra_word_tss = []
        else:
            word_timestamps = seg_grouped

        cut = False
        if force_max_len:
            curr_len = 0
            for word_i, curr_wts in enumerate(word_timestamps):
                curr_len += len(curr_wts['text'].strip() if word_i == 0 and strip else curr_wts['text'])
                if word_i != 0 and curr_len > force_max_len:

                    remaining_word_tss = word_timestamps[word_i:]
                    if merge_prev:
                        word_tss_ls[-1] = word_timestamps[:word_i]
                    else:
                        word_tss_ls.append(word_timestamps[:word_i])

                    next_seg_text = segments[seg_i + 1]['text'] if seg_i < len(segments) - 1 else ''

                    remaining_words_len = sum(map(lambda x: len(x[1].strip()) if x == 0 and strip else len(x[1]),
                                                  enumerate(remaining_word_tss)))
                    next_seg_text_len = len(next_seg_text.strip() if strip else next_seg_text)
                    if next_seg_text_len + remaining_words_len + 1 > force_max_len \
                            or word_i == len(word_timestamps) - 1:
                        word_tss_ls.extend(split_extra_words(remaining_word_tss))
                    else:
                        prev_extra_word_tss = remaining_word_tss + [dict(text=' ',
                                                                         start=remaining_word_tss[-1]['end'],
                                                                         end=None)]
                    cut = True
                    break
        if not cut:
            word_tss_ls.append(word_timestamps)

    final_seg_word_ts = []
    prev_len = 0

    for twtss_i, temp_word_timestamps in enumerate(word_tss_ls):
        is_last = twtss_i == len(word_tss_ls) - 1
        dur = temp_word_timestamps[-1]['end'] - temp_word_timestamps[0]['start']
        if dur < min_dur:
            temp_word_timestamps = [dict(text=''.join(wts['text'] for wts in temp_word_timestamps),
                                         start=temp_word_timestamps[0]['start'],
                                         end=temp_word_timestamps[-1]['end'])]
            word_tss_ls[twtss_i] = temp_word_timestamps
        prev_dur = final_seg_word_ts[-1]['end'] - final_seg_word_ts[-prev_len]['start'] if twtss_i != 0 else 0
        next_dur = word_tss_ls[twtss_i + 1][-1]['end'] - word_tss_ls[twtss_i + 1][0]['start'] \
            if not is_last else 0
        next_len = len(word_tss_ls[twtss_i + 1]) if not is_last else 0
        replace_last = (dur < min_dur <= prev_dur and
                        twtss_i != 0 and
                        (prev_dur < next_dur or
                         (prev_dur == next_dur and prev_len <= next_len) or
                         is_last))
        if replace_last:
            temp_word_timestamps = merge_grouped_wtss(word_tss_ls[twtss_i - 1], temp_word_timestamps)
            final_seg_word_ts = final_seg_word_ts[:-prev_len]
        prev_len = len(temp_word_timestamps)

        curr_words = [wts['text'] for wts in temp_word_timestamps]
        for wts_i, word_ts in enumerate(temp_word_timestamps):
            f_word_ts = dict(words=curr_words, idx=wts_i,
                             start=word_ts['start'], end=word_ts['end'])
            final_seg_word_ts.append(f_word_ts)

    if not ass_format:
        def sort_remove(x):
            x = sorted(x, key=lambda j: j['idx'])
            for i in x:
                del i['words'], i['idx']
            return x

        return [(i[0], sort_remove(i[1])) for i in groupby(final_seg_word_ts, lambda x: x['words'])]

    return final_seg_word_ts


def clamp_segment_ts(res: dict,
                     end_at_last_word=False,
                     end_before_period=False,
                     start_at_first_word=False):
    """

    Parameters
    ----------
    res: dict
        results from modified model
    end_at_last_word: bool
        set end-of-segment to timestamp-of-last-token
    end_before_period: bool
        set end-of-segment to timestamp-of-last-non-period-token
    start_at_first_word: bool
        set start-of-segment to timestamp-of-first-token

    """
    segs = tighten_timestamps(res,
                              end_at_last_word=end_at_last_word,
                              end_before_period=end_before_period,
                              start_at_first_word=start_at_first_word)['segments']

    max_idx = len(segs) - 1
    i = 1
    while i <= max_idx:
        if not (segs[i]['end'] - segs[i]['start']):
            if segs[i - 1]['end'] == segs[i]['end']:
                segs[i - 1]['text'] += (' ' + segs[i]['text'].strip())
                del segs[i]
                max_idx -= 1
                continue
            else:
                segs[i]['start'] = segs[i - 1]['end']
        i += 1

    return segs


def results_to_sentence_srt(res: dict, srt_path,
                            end_at_last_word=False,
                            end_before_period=False,
                            start_at_first_word=False,
                            force_max_len: int = None,
                            strip=True):
    """

    Parameters
    ----------
    res: dict
        results from modified model
    srt_path: str
        output path of srt
    end_at_last_word: bool
        set end of segment to timestamp of last token
    end_before_period: bool
        set end of segment to timestamp of last non-period token
    start_at_first_word: bool
        set start of segment to timestamp of first-token
    force_max_len: int
        limit a max number of characters per segment. Ignored if None (Default: None)
        Note: character count is still allow to go under this number for stability reasons.
    strip: bool
        perform strip() on each segment

    """
    if force_max_len:
        segs = finalize_segment_word_ts(res,
                                        end_at_last_word=end_at_last_word,
                                        end_before_period=end_before_period,
                                        start_at_first_word=start_at_first_word,
                                        force_max_len=force_max_len,
                                        strip=strip)
        segs = [dict(text=''.join(i), start=j[0]['start'], end=j[-1]['end']) for i, j in segs]
    else:
        segs = clamp_segment_ts(res,
                                end_at_last_word=end_at_last_word,
                                end_before_period=end_before_period,
                                start_at_first_word=start_at_first_word)

    to_srt(segs, srt_path, strip=strip)


def results_to_word_srt(res: dict, srt_path, combine_compound=False, strip=False, min_dur: float = None):
    """

    Parameters
    ----------
    res: dict
        results from modified model
    srt_path: str
        output path of srt
    combine_compound: bool
        concatenate words without inbetween spacing
    strip: bool
        perform strip() on each word
    min_dur: float
        minimum duration for each word (i.e. concat the words if it is less than specified value; Default 0.02)

    """
    word_ts = finalize_segment_word_ts(res,
                                       combine_compound=combine_compound,
                                       min_dur=min_dur,
                                       strip=strip)
    word_ts = [dict(text=j[0], **j[1]) for j in chain.from_iterable(zip(*i) for i in word_ts)]
    to_srt(word_ts, srt_path, strip=strip)


def results_to_token_srt(res: dict, srt_path, combine_compound=False, strip=False, min_dur: float = None):
    """

    Parameters
    ----------
    res: dict
        results from modified model
    srt_path: str
        output path of srt
    combine_compound: bool
        concatenate words without inbetween spacing
    strip: bool
        perform strip() on each token
    min_dur: float
        minimum duration for each token (i.e. concat the tokens if it is less than specified value; Default 0.02)

    """
    word_ts = finalize_segment_word_ts(res,
                                       combine_compound=combine_compound,
                                       min_dur=min_dur,
                                       strip=strip,
                                       ts_key='word_timestamps')
    word_ts = [dict(text=j[0], **j[1]) for j in chain.from_iterable(zip(*i) for i in word_ts)]
    to_srt(word_ts, srt_path, strip=strip)


def results_to_sentence_word_ass(res: (dict, list), ass_path: str,
                                 color: str = None, underline=True,
                                 prefmt: str = None, suffmt: str = None,
                                 font: str = None, font_size: int = 48,
                                 end_at_last_word=False,
                                 end_before_period=False,
                                 start_at_first_word=False,
                                 combine_compound=False,
                                 min_dur: float = None,
                                 force_max_len: int = None,
                                 strip=True, **kwargs):
    """

    Generate Advanced SubStation Alpha (ASS) file from results to
    display both phrase-level & word-level timestamp simultaneously by:
     -using segment-level timestamps display phrases as usual
     -using word-level timestamps change formats (e.g. color/underline) of the word in the displayed segment

    Note: ass file is used in the same way as srt, vtt, etc.

    Parameters
    ----------
    res: dict
        results from modified model
    ass_path: str
        output path (e.g. caption.ass)
    color: str
        color code for a word at its corresponding timestamp
        <bbggrr> reverse order hexadecimal RGB value (e.g. FF0000 is full intensity blue. Default: 00FF00)
    underline: bool
        whether to underline a word at its corresponding timestamp
    prefmt: str
        used to specify format for word-level timestamps (must be use with 'suffmt' and overrides 'color'&'underline')
        appears as such in the .ass file:
            Hi, {<prefmt>}how{<suffmt>} are you?
        reference [Appendix A: Style override codes] in http://www.tcax.org/docs/ass-specs.htm
    suffmt: str
        used to specify format for word-level timestamps (must be use with 'prefmt' and overrides 'color'&'underline')
        appears as such in the .ass file:
            Hi, {<prefmt>}how{<suffmt>} are you?
        reference [Appendix A: Style override codes] in http://www.tcax.org/docs/ass-specs.htm
    font: str
        word font (default: Arial)
    font_size: int
        word font size (default: 48)
    end_at_last_word: bool
        set end of segment to timestamp of last token
    end_before_period: bool
        set end of segment to timestamp of last non-period token
    start_at_first_word: bool
        set start of segment to timestamp of first-token
    combine_compound: bool
        concatenate words without inbetween spacing
    min_dur: float
        minimum duration for each word (i.e. concat the word if it is less than specified value; Default 0.02)
    force_max_len: int
        force a max number of characters per segment. Ignored if None (Default: None)
    strip: bool
        perform strip() on each segment
    kwargs:
        used for format styles:
        'Name', 'Fontname', 'Fontsize', 'PrimaryColour', 'SecondaryColour', 'OutlineColour', 'BackColour', 'Bold',
        'Italic', 'Underline', 'StrikeOut', 'ScaleX', 'ScaleY', 'Spacing', 'Angle', 'BorderStyle', 'Outline',
        'Shadow', 'Alignment', 'MarginL', 'MarginR', 'MarginV', 'Encoding'

    """

    if min_dur is None:
        min_dur = MIN_DUR

    fmt_style_dict = {'Name': 'Default', 'Fontname': 'Arial', 'Fontsize': '48', 'PrimaryColour': '&Hffffff',
                      'SecondaryColour': '&Hffffff', 'OutlineColour': '&H0', 'BackColour': '&H0', 'Bold': '0',
                      'Italic': '0', 'Underline': '0', 'StrikeOut': '0', 'ScaleX': '100', 'ScaleY': '100',
                      'Spacing': '0', 'Angle': '0', 'BorderStyle': '1', 'Outline': '1', 'Shadow': '0',
                      'Alignment': '2', 'MarginL': '10', 'MarginR': '10', 'MarginV': '10', 'Encoding': '0'}

    for k, v in filter(lambda x: 'colour' in x[0].lower() and not str(x[1]).startswith('&H'), kwargs.items()):
        kwargs[k] = f'&H{kwargs[k]}'

    fmt_style_dict.update((k, v) for k, v in kwargs.items() if k in fmt_style_dict)

    if font:
        fmt_style_dict.update(Fontname=font)
    if font_size:
        fmt_style_dict.update(Fontsize=font_size)

    fmts = f'Format: {", ".join(map(str, fmt_style_dict.keys()))}'

    styles = f'Style: {",".join(map(str, fmt_style_dict.values()))}'

    ass_str = f'[Script Info]\nScriptType: v4.00+\nPlayResX: 384\nPlayResY: 288\nScaledBorderAndShadow: yes\n\n' \
              f'[V4+ Styles]\n{fmts}\n{styles}\n\n' \
              f'[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n\n'

    if prefmt or suffmt:
        if suffmt:
            assert prefmt, 'prefmt must be used along with suffmt'
        else:
            suffmt = r'\r'
    else:
        if not color:
            color = 'HFF00'
        underline_code = r'\u1' if underline else ''

        prefmt = r'{\1c&' + f'{color.upper()}&{underline_code}' + '}'
        suffmt = r'{\r}'

    def secs_to_hhmmss(secs: (float, int)):
        mm, ss = divmod(secs, 60)
        hh, mm = divmod(mm, 60)
        return f'{hh:0>1.0f}:{mm:0>2.0f}:{ss:0>2.2f}'

    def dialogue(words: List[str], idx, start, end) -> str:
        text = ''.join((f'{prefmt}{word}{suffmt}'
                        if not word.startswith(' ') or word == ' ' else
                        f' {prefmt}{word.strip()}{suffmt}')
                       if curr_idx == idx else
                       word
                       for curr_idx, word in enumerate(words))
        return f"Dialogue: 0,{secs_to_hhmmss(start)},{secs_to_hhmmss(end)}," \
               f"Default,,0,0,0,,{text.strip() if strip else text}"

    final_phrase_word_ts = finalize_segment_word_ts(res,
                                                    end_at_last_word=end_at_last_word,
                                                    end_before_period=end_before_period,
                                                    start_at_first_word=start_at_first_word,
                                                    combine_compound=combine_compound,
                                                    min_dur=min_dur,
                                                    force_max_len=force_max_len,
                                                    strip=strip,
                                                    ass_format=True)

    ass_str += '\n'.join(map(lambda x: dialogue(**x), final_phrase_word_ts))

    if not ass_path.endswith('.ass'):
        ass_path += '.ass'

    _save_as_file(ass_str, ass_path)


def save_as_json(results: dict, path: str):
    if not path.endswith('.json'):
        path += '.json'
    results = json.dumps(results)
    _save_as_file(results, path)


def load_results(json_path: str):
    """
    Load results saved as json.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

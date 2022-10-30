import ffmpeg
import whisper
import warnings
import numpy as np
import torch
from torch import Tensor
from typing import List, Optional, Tuple, Union
from whisper.audio import SAMPLE_RATE, N_FRAMES, HOP_LENGTH, pad_or_trim, log_mel_spectrogram
from whisper.decoding import DecodingOptions, DecodingResult
from whisper.tokenizer import LANGUAGES
from whisper.utils import exact_div, format_timestamp, compression_ratio
from whisper.model import Whisper
from whisper.decoding import DecodingTask
from whisper.tokenizer import Tokenizer, get_tokenizer
from types import MethodType
from itertools import chain, repeat
from copy import deepcopy
import os
import json


# no_caption changed to no_speech newer commits
def get_new_attrs(obj_, attr: str):
    if attr == 'no_caption_probs':
        return getattr(obj_, attr) if hasattr(obj_, 'no_caption_probs') else getattr(obj_, 'no_speech_probs')
    elif attr == 'no_caption_prob':
        return getattr(obj_, attr) if hasattr(obj_, 'no_caption_prob') else getattr(obj_, 'no_speech_prob')
    elif attr == 'no_captions':
        return getattr(obj_, attr) if hasattr(obj_, 'no_captions') else getattr(obj_, 'no_speech')
    else:
        raise NotImplementedError(attr)


def check_ascending_sequence(seq: Union[List[Union[int, float]], np.ndarray], verbose=True) -> bool:
    """
    check if a sequence of numbers are in ascending order
    """
    is_ascending = True
    for idx, (i, j) in enumerate(zip(seq[:-1], seq[1:])):
        if i > j:
            is_ascending = False
            if verbose:
                print(f'[Index{idx}]:{i} > [Index{idx + 1}]:{j}')
            else:
                break

    return is_ascending


def check_ascending_sentence_ts(res: (dict, list)) -> bool:
    segs = res['segments'] if isinstance(res, dict) else res
    return check_ascending_sequence(list(chain.from_iterable((float(i['start']), float(i['end']))
                                                             for i in segs)))


def check_ascending_word_ts(res: (dict, list)) -> bool:
    cc = group_word_timestamps(res['segments'] if isinstance(res, dict) else res, ts_key='word_timestamps')
    return check_ascending_sequence((list(chain.from_iterable((float(i['start']), float(i['end']))
                                                              for i in cc))))


def is_equal_ts(a: (float, int, np.ndarray), b: (float, int, np.ndarray), rtol=1e-03):
    """
    check if timestamp a and timestamp b are equal within the relative tolerance (rtol)
    """
    return np.isclose(a, b, rtol=rtol)


def check_is_same_results(res0: (dict, list), res1: (dict, list), check_unstable=False) -> bool:
    """
    check if res0 and res1 have same timestamps
    """
    if isinstance(res0, dict):
        res0 = res0['segments']
    if isinstance(res1, dict):
        res1 = res1['segments']
    ts_key = 'unstable_word_timestamps' if check_unstable else 'word_timestamps'
    inner_ts_key = 'timestamps' if check_unstable else 'timestamp'

    def _reduce(x):
        if isinstance(x, np.ndarray):
            return set(tuple(x)) == {True}
        return x

    t = set(set(_reduce(is_equal_ts(a[inner_ts_key], b[inner_ts_key])) for a, b in zip(i[ts_key], j[ts_key])) == {True}
            for i, j in zip(res0['segments'], res1['segments']))
    return t == {True}


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

    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(srt_str)
        print(f'Saved: {os.path.abspath(save_path)}')

    return srt_str


def group_word_timestamps(res: (dict, list), one_group=True, combine_compound=False, ts_key='whole_word_timestamps'):
    def group_ts(ts_: List[dict], start) -> List[dict]:
        first_group: List[dict] = []
        for w_ts in ts_:
            if first_group:
                if (not combine_compound or w_ts['word'].startswith(' ')) and \
                        (w_ts['timestamp'] - first_group[-1]['start']) > 0.02 and \
                        first_group[-1]['end'] < w_ts['timestamp']:
                    first_group.append(dict(start=first_group[-1]['end'],
                                            end=w_ts['timestamp'],
                                            text=w_ts['word']))
                else:
                    first_group[-1]['end'] = max(first_group[-1]['end'], w_ts['timestamp'])
                    first_group[-1]['text'] += w_ts['word']
            else:
                first_group.append(dict(start=start,
                                        end=w_ts['timestamp'],
                                        text=w_ts['word']))

        return first_group

    def group_zero_duration(first_group: List[dict]) -> List[dict]:
        final_group: List[dict] = []
        for ts_dict in first_group:
            if not final_group or (ts_dict['end'] - ts_dict['start']) > 0:
                final_group.append(ts_dict)
            else:
                final_group[-1]['end'] = ts_dict['end']
                final_group[-1]['text'] += ts_dict['text']

        return final_group

    segs: List[dict] = res['segments'] if isinstance(res, dict) else res
    assert set(ts_key in seg for seg in segs) == {True}, f'input contains missing {ts_key}'

    grouped = (group_ts(seg[ts_key], seg['start']) for seg in segs)
    return group_zero_duration(list(chain.from_iterable(grouped))) if one_group else list(grouped)


def tighten_timestamps(res: dict, end_at_last_word=True, end_before_period=False, start_at_first_word=False) -> dict:
    res = deepcopy(res)
    for i in range(len(res['segments'])):
        if start_at_first_word:
            res['segments'][i]['start'] = res['segments'][i]['word_timestamps'][0]['timestamp']
        if end_before_period and \
                res['segments'][i]['word_timestamps'][-1] == '.' and \
                len(res['segments'][i]['word_timestamps']) > 1:
            res['segments'][i]['end'] = res['segments'][i]['word_timestamps'][-2]['timestamp']
        elif end_at_last_word:
            res['segments'][i]['end'] = res['segments'][i]['word_timestamps'][-1]['timestamp']

    return res


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


def results_to_sentence_srt(res: dict, srt_path,
                            end_at_last_word=False,
                            end_before_period=False,
                            start_at_first_word=False,
                            strip=True):
    """

    Parameters
    ----------
    res: dict
        results from modified model
    srt_path: str
        output path of srt
    end_at_last_word: bool
        set end-of-sentence to timestamp-of-last-token
    end_before_period: bool
        set end-of-sentence to timestamp-of-last-non-period-token
    start_at_first_word: bool
        set start-of-sentence to timestamp-of-first-token
    strip: bool
        perform strip() on each sentence

    """
    strict = any((end_at_last_word, end_before_period, start_at_first_word))
    segs = tighten_timestamps(res,
                              end_at_last_word=end_at_last_word,
                              end_before_period=end_before_period,
                              start_at_first_word=start_at_first_word)['segments'] \
        if strict else res['segments']

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

    to_srt(segs, srt_path, strip=strip)


def results_to_word_srt(res: dict, srt_path, combine_compound=False, strip=False):
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

    """
    to_srt(group_word_timestamps(res, combine_compound=combine_compound),
           srt_path, strip=strip)


def results_to_token_srt(res: dict, srt_path, combine_compound=False, strip=False):
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

    """
    to_srt(group_word_timestamps(res, combine_compound=combine_compound, ts_key='word_timestamps'),
           srt_path, strip=strip)


def _get_min_estimation(estimations: List[Union[list, np.ndarray]],
                        min_: (int, float) = None,
                        max_: (int, float) = None) -> np.ndarray:
    estimations = deepcopy(estimations)
    estimations = list(map(lambda est_: np.array(est_) if isinstance(est_, list) else est_, estimations))
    prev_min = min_ or 0
    curr_max = max_ or np.max(estimations[-1])

    min_est = []
    for curr_est in estimations:
        curr_min = curr_est[np.logical_and(curr_max > curr_est, curr_est > prev_min)]
        curr_min = np.min(curr_min) if curr_min.shape[0] else prev_min
        min_est.append(curr_min)
        prev_min = curr_min

    return np.array(min_est)


def _get_max_estimation(estimations: List[Union[list, np.ndarray]],
                        max_: (int, float) = None,
                        min_: (int, float) = None) -> np.ndarray:
    estimations = deepcopy(estimations)
    estimations = list(map(lambda est_: np.array(est_) if isinstance(est_, list) else est_, estimations))
    prev_max = max_ or np.max(estimations[-1])
    curr_min = np.min(estimations[0]) if min_ is None else min_

    max_est = []
    for curr_est in reversed(estimations):
        curr_max = curr_est[np.logical_and(prev_max > curr_est, curr_est > curr_min)]
        curr_max = np.max(curr_max) if curr_max.shape[0] else prev_max
        max_est.append(curr_max)
        prev_max = curr_max

    max_est.reverse()
    return np.array(max_est)


def _remove_overestimation(x: Union[np.ndarray, List[Union[int, float]]], alt_est: List[Union[list, np.ndarray]] = None,
                           max_: (int, float) = None, min_: (int, float) = None,
                           aggressive=False) -> np.ndarray:
    x = np.array(x) if isinstance(x, list) else deepcopy(x)
    if alt_est is not None:
        alt_est = list(map(lambda est_: np.array(est_) if isinstance(est_, list) else est_, alt_est))
    assert x.ndim == 1
    assert alt_est is None or len(alt_est) == x.shape[0]
    max_val = x[-1] if max_ is None else max_
    min_val = x[0] if min_ is None else min_

    def curr_max_min(val):
        if min_ is None:
            return val
        return max(min_, val)

    if min_ is not None:
        x[x < min_] = min_
    reduce_ = np.min if aggressive else np.mean
    for i in range(x.shape[-1] - 1, -1, -1):
        if x[i] > max_val or (i > 1 and x[i] < reduce_(x[:i])):  # spikes or dips
            if alt_est is None or alt_est[i] is None:
                x[i] = max_val
            else:
                tmp_min = min_val if i < 2 else curr_max_min(np.mean(x[:i]))
                alt_ = alt_est[i][np.logical_and(alt_est[i] < max_val, alt_est[i] > tmp_min)]
                x[i] = max_val if alt_.shape[0] == 0 else alt_[0]
        max_val = x[i]
    return x


def _remove_underestimation(x: Union[np.ndarray, List[Union[int, float]]],
                            alt_est: List[Union[list, np.ndarray]] = None,
                            min_: (int, float) = None, max_: (int, float) = None,
                            aggressive=False) -> np.ndarray:
    x = np.array(x) if isinstance(x, list) else deepcopy(x)
    if alt_est is not None:
        alt_est = list(map(lambda est_: np.array(est_) if isinstance(est_, list) else est_, alt_est))
    assert x.ndim == 1
    assert alt_est is None or len(alt_est) == x.shape[0]
    min_val = x[0] if min_ is None else min_
    max_val = x[-1] if max_ is None else max_

    def curr_min_max(val):
        if max_ is None:
            return val
        return min(max_, val)

    if max_ is not None:
        x[x > max_] = max_
    reduce_ = np.max if aggressive else np.mean
    max_i_reduce = x.shape[-1] - 2
    for i in range(0, x.shape[-1]):
        if x[i] < min_val or (i < max_i_reduce and x[i] > reduce_(x[i + 1:])):  # dips or spikes
            if alt_est is None or alt_est[i] is None:
                x[i] = min_val
            else:
                tmp_max = max_val if i >= max_i_reduce else curr_min_max(np.mean(x[i + 1:]))
                alt_ = alt_est[i][np.logical_and(alt_est[i] > min_val, alt_est[i] < tmp_max)]
                x[i] = min_val if alt_.shape[0] == 0 else alt_[0]
        min_val = x[i]
    return x


def _merge_max_min_estimation(mx: Union[np.ndarray, List[Union[int, float]]],
                              mn: Union[np.ndarray, List[Union[int, float]]],
                              alt_est: List[Union[list, np.ndarray]] = None) -> np.ndarray:
    mx = np.array(mx) if isinstance(mx, list) else deepcopy(mx)
    mn = np.array(mn) if isinstance(mn, list) else deepcopy(mn)
    if alt_est is not None:
        alt_est = list(map(lambda est_: np.array(est_) if isinstance(est_, list) else est_, alt_est))
    assert mx.ndim == 1 and mn.ndim == 1
    assert mx.shape[0] == mn.shape[0]
    assert alt_est is None or len(alt_est) == mx.shape[0]

    pref_mx = np.var(mx) > np.var(mn)
    if pref_mx:
        mn[0] = mx[0]
    prev_min = mn[0]
    for i in range(1, mn.shape[0]):
        if prev_min > mn[i]:
            if mn[i] > mx[i]:  # prev_min > mn[i] > mx[i]
                mn[i] = prev_min
            elif mx[i] > mn[i]:
                if prev_min > mx[i]:  # prev_min > mx[i] > mn[i]
                    mn[i] = prev_min
                else:  # mx[i] > prev_min > mn[i]
                    alt_ = alt_est[i][np.logical_and(alt_est[i] > prev_min, alt_est[i] < mx[i])]
                    mn[i] = (mx[i] if pref_mx else prev_min) if alt_.shape[0] == 0 else alt_[0]
            else:  # prev_min > mn[i] == mx[i]
                mn[i] = prev_min
        elif mn[i] > prev_min:
            # if prev_min > mx[i]:  # mn[i] > prev_min > mx[i]
            #     pass
            if mx[i] > prev_min:
                if mn[i] > mx[i]:  # mn[i] > mx[i] > prev_min
                    pass
                elif mx[i] > mn[i]:  # mx[i] > mn[i] > prev_min
                    alt_ = alt_est[i][np.logical_and(alt_est[i] > mn[i], alt_est[i] < mx[i])]
                    if alt_.shape[0]:
                        mn[i] = alt_[0]
                    elif pref_mx:
                        mn[i] = mx[i]
            #     else:  # mx[i] == mn[i] > prev_min
            #         pass
            # else:  # mn[i] > mx[i] == prev_min
            #     pass
        else:  # mn[i] == prev_min
            if mx[i] > mn[i]:  # mx[i] > mn[i] == prev_min
                alt_ = alt_est[i][np.logical_and(alt_est[i] > mn[i], alt_est[i] < mx[i])]
                if alt_.shape[0]:
                    mn[i] = alt_[0]
                elif pref_mx:
                    mn[i] = mx[i]
            # elif mn[i] > mx[i]:  # mn[i] == prev_min > mx[i]
            #     pass
            # else:  # mn[i] == prev_min == mx[i]
            #     pass

        prev_min = mn[i]

    return mn


def _avg_merge_min_max(mx: Union[np.ndarray, List[Union[int, float]]],
                       mn: Union[np.ndarray, List[Union[int, float]]],
                       alt_timestamps: List[Union[List[Union[int, float]], np.ndarray]] = None,
                       max_: (int, float) = None, min_: (int, float) = None):
    mx = np.array(mx) if isinstance(mx, list) else deepcopy(mx)
    mn = np.array(mn) if isinstance(mn, list) else deepcopy(mn)
    assert mx.ndim == mn.ndim == 1
    assert mx.shape[0] == mn.shape[0]

    avg_ = (mx + mn) / 2

    if check_ascending_sequence(avg_, verbose=False):
        return avg_

    if not max_:
        max_ = max(mx[-1], mn[-1])
    if min_ is None:
        min_ = min(mn[0], mx[0])

    return _stabilize_timestamps(avg_, alt_timestamps, max_=max_, min_=min_)


def _stabilize_timestamps(timestamps: Union[np.ndarray, List[Union[int, float]]],
                          alt_timestamps: List[Union[List[Union[int, float]], np.ndarray]] = None,
                          max_: (int, float) = None, min_: (int, float) = None, aggressive=False) -> np.ndarray:
    mx = _remove_overestimation(timestamps, alt_est=alt_timestamps, max_=max_, min_=min_, aggressive=aggressive)
    mn = _remove_underestimation(timestamps, alt_est=alt_timestamps, max_=max_, min_=min_, aggressive=aggressive)
    return _merge_max_min_estimation(mx, mn, alt_timestamps)


def _stabilize_more_timestamps(timestamps: List[Union[list, np.ndarray]],
                               max_: (int, float) = None, min_: (int, float) = None, average=True) -> np.ndarray:
    mx = _get_max_estimation(timestamps, max_=max_, min_=min_)
    mn = _get_min_estimation(timestamps, max_=max_, min_=min_)
    if average:
        return _avg_merge_min_max(mx, mn, timestamps, max_=max_, min_=min_)
    return _merge_max_min_estimation(mx, mn, timestamps)


def stabilize_timestamps(segments: Union[List[dict], dict],
                         top_focus=False, aggressive=False, average=True) -> List[dict]:
    """

    Parameters
    ----------
    segments: Union[List[dict], dict]
        result['segments'] or result
    top_focus: bool
        adhere closely to the top predictions for word timestamps
    aggressive: bool
        only if top_focus=True,
        allow greater variation in word_timestamps/whole_word_timestamps
    average: bool
        only if top_focus=False,
        average min and max of unstable_word_timestamps to get word_timestamps/whole_word_timestamps

    """
    if isinstance(segments, dict):
        segments = segments['segments']
    if not segments:
        warnings.warn('No Segments Found')
        return []
    missing_ts_idx = set(map(lambda x: None if x[1].get('unstable_word_timestamps') else x[0], enumerate(segments))) - {
        None}
    no_word_timestamps = len(missing_ts_idx) == len(segments)
    if not no_word_timestamps and missing_ts_idx:
        warnings.warn(f'Segments {list(missing_ts_idx)} are missing unstable_word_timestamps. '
                      f'Word-level timestamp stabilization will skipped')

    segments = deepcopy(segments)
    sectioned_segments: List[List] = [[]]
    for i, seg in enumerate(segments, 1):
        sectioned_segments[-1].append(seg)
        if seg['anchor_point']:
            if i < len(segments):
                sectioned_segments.append([])

    assert all(set(len(set(s['offset'] for s in segs)) == 1 for segs in sectioned_segments))

    sectioned_segments_timestamps = [dict(min_=segs[-1]['offset'],
                                          max_=segs[-1]['next_offset'],
                                          timestamps=list(chain.from_iterable((s['start'], s['end']) for s in segs)),
                                          alt_timestamps=list(chain.from_iterable((s['alt_start_timestamps'],
                                                                                   s['alt_end_timestamps'])
                                                                                  for s in segs)))
                                     for segs in sectioned_segments]

    sectioned_stab_timestamps = [_stabilize_timestamps(**kwargs).reshape(-1, 2) for kwargs in
                                 sectioned_segments_timestamps]

    for i in range(len(sectioned_segments)):
        for j in range(len(sectioned_segments[i])):
            sectioned_segments[i][j]['start'], sectioned_segments[i][j]['end'] = sectioned_stab_timestamps[i][j]

            if not missing_ts_idx:
                if top_focus:
                    top_word_ts = [ts_['timestamps'][0] for ts_ in
                                   sectioned_segments[i][j]['unstable_word_timestamps']]
                    alt_word_ts = [ts_['timestamps'][1:] for ts_ in
                                   sectioned_segments[i][j]['unstable_word_timestamps']]
                    temp_stab_word_ts = _stabilize_timestamps(top_word_ts, alt_word_ts,
                                                              max_=sectioned_segments[i][j]['end'],
                                                              min_=sectioned_segments[i][j]['start'],
                                                              aggressive=aggressive)
                else:
                    word_ts = [ts_['timestamps'] for ts_ in sectioned_segments[i][j]['unstable_word_timestamps']]
                    temp_stab_word_ts = _stabilize_more_timestamps(word_ts,
                                                                   max_=sectioned_segments[i][j]['end'],
                                                                   min_=sectioned_segments[i][j]['start'],
                                                                   average=average)

                temp_stab_word_ts = [{'word': sectioned_segments[i][j]['unstable_word_timestamps'][k]['word'],
                                      'token': sectioned_segments[i][j]['unstable_word_timestamps'][k]['token'],
                                      'timestamp': temp_stab_word_ts[k]}
                                     for k in range(temp_stab_word_ts.shape[0])]

                sectioned_segments[i][j]['word_timestamps'] = temp_stab_word_ts

    return list(chain.from_iterable(sectioned_segments))


def save_as_json(results, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f)


def add_whole_word_ts(tokenizer: Tokenizer, segments: Union[List[dict], dict], merge_non_space: bool = None,
                      prepend_punctuations: Union[List[str], Tuple[str]] = None,
                      append_punctuations: Union[List[str], Tuple[str]] = None):
    merge_non_space = (tokenizer.language in ['en'] or tokenizer.language is None) \
        if merge_non_space is None else merge_non_space
    if prepend_punctuations is None:
        prepend_punctuations = r'“¿([{'
    if append_punctuations is None:
        append_punctuations = r'.。,，!！?？:：”)]}、'
    if isinstance(segments, dict):
        segments = segments['segments']
    if not segments:
        print('No segments found, whole-word timestamps cannot be added.')
        return

    missing_idx = set(-1 if seg.get('word_timestamps') else i for i, seg in enumerate(segments)) - {-1}

    if missing_idx:
        if len(missing_idx) == len(segments):
            print('No word_timestamps found, whole-word timestamps cannot be added.')
            return
        print(f'Some word_timestamps not found, '
              f'whole-word timestamps cannot be added to the following segments: {tuple(missing_idx)}')

    failed_idx = []

    for seg_idx, seg in enumerate(segments):
        if seg.get('word_timestamps'):
            prev_idx = 0
            remaining_text = seg['text']
            has_prepend = False
            whole_word_timestamps: List[dict] = []
            for wts_idx in range(1, len(seg['word_timestamps']) + 1):
                max_ts = seg['word_timestamps'][wts_idx - 1]['timestamp']
                tokens = [wts['token'] for wts in seg['word_timestamps'][prev_idx: wts_idx]]
                temp_whole_word = tokenizer.decode(tokens)
                if temp_whole_word == remaining_text[:len(temp_whole_word)]:
                    prev_idx = wts_idx
                    remaining_text = remaining_text[len(temp_whole_word):]
                    if (not merge_non_space or temp_whole_word.startswith(' ') or not whole_word_timestamps) and \
                            temp_whole_word not in append_punctuations and \
                            not has_prepend:
                        has_prepend = temp_whole_word.strip() in prepend_punctuations
                        whole_word_timestamps.append(dict(word=temp_whole_word, timestamp=max_ts))
                    else:
                        has_prepend = False
                        whole_word_timestamps[-1]['word'] += temp_whole_word
                        whole_word_timestamps[-1]['timestamp'] = max_ts
            if remaining_text:
                failed_idx.append(seg_idx)
                whole_word_timestamps = []
            seg['whole_word_timestamps'] = whole_word_timestamps or None
        else:
            seg['whole_word_timestamps'] = None

    if failed_idx:
        print(f'Failed to add whole-word timestamps to the following segments: {tuple(failed_idx)}')


def load_audio_waveform(audio: str, h: int, w: int) -> np.ndarray:
    """

    Parameters
    ----------
    audio: str:
        Audio path
    h: int
        Height of waveform image
    w: int
        Width of waveform image

    Returns
    -------
    Audio waveform image as a NumPy array, in uint8 dtype.
    """
    try:
        waveform, _ = (
            ffmpeg.input(audio, threads=0)
                .filter('aformat', channel_layouts='mono')
                .filter('highpass', f='200').filter('lowpass', f='3000')
                .filter('showwavespic', s=f'{w}x{h}')
                .output('-', pix_fmt='gray', format='rawvideo')
                .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio in waveform: {e.stderr.decode()}") from e
    else:
        return np.frombuffer(waveform, dtype=np.uint8).reshape(h, w)


def remove_lower_quantile(waveform: np.ndarray,
                          upper_quantile: float = None,
                          lower_quantile: float = None,
                          lower_threshold: float = None) -> np.ndarray:
    """
    Removes lower quantile of amplitude from waveform image
    """
    if upper_quantile is None:
        upper_quantile = 0.85
    if lower_quantile is None:
        lower_quantile = 0.15
    if lower_threshold is None:
        lower_threshold = 0.15
    waveform = deepcopy(waveform)
    wave_sums = waveform.sum(0)
    mx = np.quantile(wave_sums, upper_quantile, 0)
    mn = np.quantile(wave_sums, lower_quantile, 0)
    mn_threshold = (mx - mn) * lower_threshold + mn
    waveform[:, wave_sums < mn_threshold] = 0
    return waveform


def wave_to_ts_filter(waveform: np.ndarray, suppress_middle=True,
                      max_index: (list, int) = None) -> np.ndarray:
    """
    Returns A NumPy array mask of sections with amplitude zero
    """
    assert waveform.ndim <= 2, f'waveform have at most 2 dims but found {waveform.ndim}'
    if waveform.ndim == 1:
        wave_sum = waveform
    else:
        wave_sum = waveform.sum(-2)

    wave_filter = wave_sum.astype(bool)

    if not suppress_middle:
        nonzero_indices = wave_filter.nonzero()[0]
        wave_filter[nonzero_indices[0]:nonzero_indices[-1] + 1] = True
    if max_index is not None:
        wave_filter[max_index + 1:] = False

    return ~wave_filter


# modified version of whisper.transcribe.transcribe
def transcribe_word_level(
        model: "Whisper",
        audio: Union[str, np.ndarray, torch.Tensor],
        *,
        verbose: bool = False,
        temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threshold: Optional[float] = 2.4,
        logprob_threshold: Optional[float] = -1.0,
        no_speech_threshold: Optional[float] = 0.6,
        condition_on_previous_text: bool = True,
        stab=True, top_focus=False, ts_num: int = 10,
        alpha: float = None, print_unstab=False,
        suppress_silence: bool = True,
        suppress_middle: bool = True,
        suppress_word_ts: bool = True,
        remove_background: bool = True,
        prepend_punctuations: Union[List[str], Tuple[str]] = None,
        append_punctuations: Union[List[str], Tuple[str]] = None,
        **decode_options):
    """
    Transcribe an audio file using Whisper

    Parameters
    ----------
    model: Whisper
        The Whisper model instance

    audio: Union[str, np.ndarray, torch.Tensor]
        The path to the audio file to open, or the audio waveform

    verbose: bool
        Whether to display the text (with finalized timestamps) being decoded to the console

    temperature: Union[float, Tuple[float, ...]]
        Temperature for sampling. It can be a tuple of temperatures, which will be successfully used
        upon failures according to either `compression_ratio_threshold` or `logprob_threshold`.

    compression_ratio_threshold: float
        If the gzip compression ratio is above this value, treat as failed

    logprob_threshold: float
        If the average log probability over sampled tokens is below this value, treat as failed

    no_speech_threshold: float
        If the no_speech probability is higher than this value AND the average log probability
        over sampled tokens is below `logprob_threshold`, consider the segment as silent

    condition_on_previous_text: bool
        if True, the previous output of the model is provided as a prompt for the next window;
        disabling may make the text inconsistent across windows, but the model becomes less prone to
        getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.

    stab: bool
        Stabilizing timestamps by cross compare timestamps and using additional top timestamp predictions
        to fill in when appropriate to ensure timestamps are chronological.

    top_focus: bool
        Adhere closely to the top predictions for token timestamps stabilization

    ts_num: int
        Number of top timestamp predictions to save for each word for postprocessing stabilization (default: 10).

    alpha: float
        Amount of noise to add to audio to produce slightly difference results.
        audio_features *= torch.rand_like(audio_features) * alpha + 1

    print_unstab: bool
        Whether to display the text (without stabilize timestamps) being decoded to the console

    suppress_silence: bool
        Suppress timestamp tokens that are marked as silent

    suppress_middle: bool
        Suppress any silent timestamps tokens of middle of the segment instead of only beginning and ending

    suppress_word_ts: bool
        Suppress timestamp tokens of words that are marked as silent

    remove_background: bool
        Whether to remove background noise from waveform so that it is marked silent. Determined by parameters:
            upper_quantile: float
                The upper quantile of amplitude to determine a max amplitude, mx (Default: 0.85)
            lower_quantile: float
                The lower quantile of amplitude to determine a min amplitude, mn (Default: 0.15)
            lower_threshold: float
                Suppressed sections of waveform where amplitude < lower_threshold*(mx-mn) + mn. (Default: 0.15)

    prepend_punctuations: Union[List[str], Tuple[str]]
        Punctuations to prepend to next word (Default: “¿([{)

    append_punctuations: Union[List[str], Tuple[str]]
        Punctuations to append to previous word (Default: .。,，!！?？:：”)]}、)

    decode_options: dict
        Keyword arguments to construct `DecodingOptions` instances

    Returns
    -------
    A dictionary containing the resulting text ("text") and segment-level details ("segments"), and
    the spoken language ("language"), which is detected when `decode_options["language"]` is None.
    """

    if 'no_captions_threshold' in decode_options:
        warnings.warn('no_captions_threshold is deprecated. '
                      'Please use no_speech_threshold instead.', DeprecationWarning, stacklevel=2)
        no_speech_threshold = decode_options.pop('no_captions_threshold')

    dtype = torch.float16 if decode_options.get("fp16", True) else torch.float32
    if model.device == torch.device("cpu"):
        if torch.cuda.is_available():
            warnings.warn("Performing inference on CPU when CUDA is available")
        if dtype == torch.float16:
            warnings.warn("FP16 is not supported on CPU; using FP32 instead")
            dtype = torch.float32

    if dtype == torch.float32:
        decode_options["fp16"] = False

    if 'max_initial_timestamp' not in decode_options:
        decode_options['max_initial_timestamp'] = None

    mel = log_mel_spectrogram(audio)

    if decode_options.get("language", None) is None:
        if verbose:
            print("Detecting language using up to the first 30 seconds. Use `--language` to specify the language")
        segment = pad_or_trim(mel, N_FRAMES).to(model.device).to(dtype)
        _, probs = model.detect_language(segment)
        decode_options["language"] = max(probs, key=probs.get)
        print(f"Detected language: {LANGUAGES[decode_options['language']]}")

    mel = mel.unsqueeze(0)
    language = decode_options["language"]
    task = decode_options.get("task", "transcribe")
    tokenizer = get_tokenizer(model.is_multilingual, language=language, task=task)

    def decode_with_fallback(segment: torch.Tensor, suppress_ts_mask: Tensor = None) \
            -> Union[List[DecodingResult], tuple]:
        temperatures = [temperature] if isinstance(temperature, (int, float)) else temperature
        kwargs = {**decode_options}
        t = temperatures[0]
        if t == 0:
            best_of = kwargs.pop("best_of", None)
        else:
            best_of = kwargs.get("best_of", None)

        options = DecodingOptions(**kwargs, temperature=t)
        results, ts_tokens, ts_logits_ = model.decode(segment, options, ts_num=ts_num, alpha=alpha,
                                                      suppress_ts_mask=suppress_ts_mask,
                                                      suppress_word_ts=suppress_word_ts)

        kwargs.pop("beam_size", None)  # no beam search for t > 0
        kwargs.pop("patience", None)  # no patience for t > 0
        kwargs["best_of"] = best_of  # enable best_of for t > 0
        for t in temperatures[1:]:
            needs_fallback = [
                compression_ratio_threshold is not None
                and result.compression_ratio > compression_ratio_threshold
                or logprob_threshold is not None
                and result.avg_logprob < logprob_threshold
                for result in results
            ]
            if any(needs_fallback):
                options = DecodingOptions(**kwargs, temperature=t)
                retries, r_ts_tokens, r_ts_logits = model.decode(segment[needs_fallback], options,
                                                                 ts_num=ts_num, alpha=alpha,
                                                                 suppress_ts_mask=suppress_ts_mask,
                                                                 suppress_word_ts=suppress_word_ts)
                for retry_index, original_index in enumerate(np.nonzero(needs_fallback)[0]):
                    results[original_index] = retries[retry_index]
                    ts_tokens[original_index] = r_ts_tokens[retry_index]
                    ts_logits_[original_index] = r_ts_logits[retry_index]

        return results, ts_tokens, ts_logits_

    seek = 0
    input_stride = exact_div(
        N_FRAMES, model.dims.n_audio_ctx
    )  # mel frames per output token: 2
    time_precision = (
            input_stride * HOP_LENGTH / SAMPLE_RATE
    )  # time per output token: 0.02 (seconds)
    all_tokens = []
    all_segments = []
    prompt_reset_since = 0

    initial_prompt = decode_options.pop("initial_prompt", None) or []
    if initial_prompt:
        initial_prompt = tokenizer.encode(" " + initial_prompt.strip())
        all_tokens.extend(initial_prompt)

    def _to_list(x: (Tensor, None)):
        if x is None:
            return x
        return x.tolist()

    def add_segment(
            *, offset: float, start: float, end: float, text_tokens: Tensor, result: DecodingResult,
            start_timestamps: list = None, end_timestamps: list = None, word_timestamps: Tensor = None,
            start_ts_logits: list = None, end_ts_logits: list = None, word_ts_logits: Tensor = None
    ):
        no_eot_mask = text_tokens < tokenizer.eot
        text_tokens_no_eot = text_tokens[no_eot_mask]
        text = tokenizer.decode(text_tokens_no_eot)

        if len(text.strip()) == 0:  # skip empty text output
            return

        if word_timestamps is not None:
            assert word_timestamps.shape[0] == text_tokens.shape[0]
            if word_ts_logits is None:
                word_ts_fields = zip(text_tokens_no_eot, word_timestamps[no_eot_mask], repeat(None))
            else:
                assert word_ts_logits.shape[0] == text_tokens.shape[0]
                word_ts_fields = zip(text_tokens_no_eot, word_timestamps[no_eot_mask], word_ts_logits[no_eot_mask])

            word_timestamps = [dict(word=tokenizer.decode([token]),
                                    token=token.item(),
                                    timestamps=timestamps_.tolist(),
                                    timestamp_logits=_to_list(ts_logits_))
                               for token, timestamps_, ts_logits_ in word_ts_fields]

        all_segments.append(
            {
                "id": len(all_segments),
                "seek": seek,
                'offset': offset,  # offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
                "start": start,
                "end": end,
                "text": text,
                "tokens": result.tokens,
                "temperature": result.temperature,
                "avg_logprob": result.avg_logprob,
                "compression_ratio": result.compression_ratio,
                "no_speech_prob": get_new_attrs(result, 'no_caption_prob'),
                "alt_start_timestamps": start_timestamps,
                "start_ts_logits": start_ts_logits,
                "alt_end_timestamps": end_timestamps,
                "end_ts_logits": end_ts_logits,
                "unstable_word_timestamps": word_timestamps,
                'anchor_point': False
            }
        )
        if print_unstab or (verbose and not stab):
            print(f'[{format_timestamp(start)} --> {format_timestamp(end)}] "{text}"')
            if word_timestamps is not None:
                ts_str = (f' ->[{format_timestamp(ts_["timestamps"][0])}] "{ts_["word"].strip()}"' for ts_ in
                          word_timestamps)
                print('\n'.join(ts_str), end='\n\n')

    if suppress_silence:
        ts_scale = HOP_LENGTH / SAMPLE_RATE / time_precision
        wf = load_audio_waveform(audio, 100, int(mel.shape[-1] * ts_scale))

    upper_quantile = decode_options.pop('upper_quantile', 0.85)
    lower_quantile = decode_options.pop('lower_quantile', 0.15)
    lower_threshold = decode_options.pop('lower_threshold', 0.15)

    while seek < mel.shape[-1]:
        timestamp_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
        remaining_duration = float((mel.shape[-1] - seek) * HOP_LENGTH / SAMPLE_RATE)
        segment = pad_or_trim(mel[:, :, seek:], N_FRAMES).to(model.device).to(dtype)
        segment_duration = min(float(segment.shape[-1] * HOP_LENGTH / SAMPLE_RATE), remaining_duration)
        segment_max_ts = segment_duration / time_precision

        if suppress_silence:
            wf_seek = int(seek * ts_scale)
            segment_wf = wf[..., wf_seek:wf_seek + 1501]
            if remove_background:
                segment_wf = remove_lower_quantile(segment_wf.astype(np.float32),
                                                   upper_quantile=upper_quantile,
                                                   lower_quantile=lower_quantile,
                                                   lower_threshold=lower_threshold)
            segment_wf = pad_or_trim(segment_wf, 1501)
            suppress_ts_mask = torch.from_numpy(wave_to_ts_filter(segment_wf,
                                                                  suppress_middle=suppress_middle,
                                                                  max_index=int(segment_max_ts)))

            if suppress_ts_mask.all():  # segment is silent
                seek += segment.shape[-1]  # fast-forward to the next segment boundary
                continue
        else:
            suppress_ts_mask = None

        decode_options["prompt"] = all_tokens[prompt_reset_since:]
        result, finalized_ts_tokens, ts_logits = decode_with_fallback(segment,
                                                                      suppress_ts_mask=suppress_ts_mask)

        result = result[0]
        tokens = torch.tensor(result.tokens)
        finalized_ts_tokens = torch.tensor(finalized_ts_tokens[0])
        ts_logits = torch.tensor(ts_logits[0])

        if no_speech_threshold is not None:
            # no voice activity check
            should_skip = get_new_attrs(result, 'no_caption_prob') > no_speech_threshold
            if logprob_threshold is not None and result.avg_logprob > logprob_threshold:
                # don't skip if the logprob is high enough, despite the no_speech_prob
                should_skip = False

            if should_skip:
                seek += segment.shape[-1]  # fast-forward to the next segment boundary
                continue

        timestamp_tokens: torch.Tensor = tokens.ge(tokenizer.timestamp_begin)
        consecutive = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0].add_(1)
        if len(consecutive) > 0:  # if the output contains two consecutive timestamp tokens
            last_slice = 0
            for current_slice in consecutive:
                sliced_tokens = tokens[last_slice:current_slice]
                sliced_ts_tokens = finalized_ts_tokens[last_slice:current_slice]
                sliced_ts_logits = ts_logits[last_slice:current_slice]
                start_timestamp_position = (
                        sliced_tokens[0].item() - tokenizer.timestamp_begin
                )
                end_timestamp_position = (
                        sliced_tokens[-1].item() - tokenizer.timestamp_begin
                )

                word_ts = timestamp_offset + (sliced_ts_tokens - tokenizer.timestamp_begin) * time_precision

                add_segment(
                    offset=timestamp_offset,
                    start=timestamp_offset + start_timestamp_position * time_precision,
                    end=min(timestamp_offset + end_timestamp_position * time_precision,
                            timestamp_offset + segment_duration),
                    text_tokens=sliced_tokens[1:-1],
                    result=result,
                    start_timestamps=word_ts[0].tolist(),
                    end_timestamps=word_ts[-1].tolist(),
                    word_timestamps=word_ts[1:-1],
                    start_ts_logits=sliced_ts_logits[0].tolist(),
                    end_ts_logits=sliced_ts_logits[-1].tolist(),
                    word_ts_logits=sliced_ts_logits[1:-1]
                )
                last_slice = current_slice
            last_timestamp_position = (
                min(tokens[last_slice - 1].item() - tokenizer.timestamp_begin, segment_max_ts)
            )
            seek += last_timestamp_position * input_stride
            all_tokens.extend(tokens[: last_slice + 1].tolist())
        else:
            duration = segment_duration
            timestamps = tokens[timestamp_tokens.nonzero().flatten()]
            if len(timestamps) > 0:
                # no consecutive timestamps but it has a timestamp; use the last one.
                # single timestamp at the end means no speech after the last timestamp.
                last_timestamp_position = min(timestamps[-1].item() - tokenizer.timestamp_begin, segment_max_ts)
                duration = last_timestamp_position * time_precision

            word_ts = timestamp_offset + (finalized_ts_tokens - tokenizer.timestamp_begin) * time_precision

            add_segment(
                offset=timestamp_offset,
                start=timestamp_offset,
                end=timestamp_offset + duration,
                text_tokens=tokens,
                result=result,
                word_timestamps=word_ts,
                word_ts_logits=ts_logits
            )

            seek += segment.shape[-1]
            all_tokens.extend(tokens.tolist())

        if all_segments:
            all_segments[-1]['anchor_point'] = True
            all_segments[-1]['next_offset'] = float(seek * HOP_LENGTH / SAMPLE_RATE)
        if not condition_on_previous_text or result.temperature > 0.5:
            # do not feed the prompt tokens if a high temperature was used
            prompt_reset_since = len(all_tokens)

    if len(all_segments) > 1 and all_segments[-1]['alt_start_timestamps'] is None:
        all_segments[-1]['alt_start_timestamps'] = all_segments[-2]['alt_end_timestamps']

    if stab:
        all_segments = stabilize_timestamps(all_segments, top_focus=top_focus)
        add_whole_word_ts(tokenizer, all_segments,
                          prepend_punctuations=prepend_punctuations,
                          append_punctuations=append_punctuations)
        if verbose:
            print('\nSTABILIZED\n')
            for seg_ in all_segments:
                print(f'[{format_timestamp(seg_["start"])} --> {format_timestamp(seg_["end"])}] "{seg_["text"]}"')
                if seg_['word_timestamps']:
                    ts_str = (f' ->[{format_timestamp(ts_["timestamp"])}] "{ts_["word"].strip()}"' for ts_ in
                              seg_['word_timestamps'])
                    print('\n'.join(ts_str), end='\n\n')

    return dict(text=tokenizer.decode(all_tokens[len(initial_prompt):]), segments=all_segments, language=language)


class DecodingTaskWordLevel(DecodingTask):

    def __init__(self, *args, **kwargs):
        super(DecodingTaskWordLevel, self).__init__(*args, **kwargs)

    # modified version of whisper.DecodingTask._main_loop
    def _main_loop(self, audio_features: Tensor, tokens: Tensor, ts_num: int = None, alpha: float = None,
                   suppress_ts_mask: Tensor = None, suppress_word_ts: bool = False):
        assert audio_features.shape[0] == tokens.shape[0]
        n_batch = tokens.shape[0]
        sum_logprobs: Tensor = torch.zeros(n_batch, device=audio_features.device)
        no_speech_probs = [np.nan] * n_batch

        ts_num = 5 if ts_num is None else max(ts_num, 1)
        initial_tk_len = tokens.shape[-1]
        ts_tokens = torch.zeros([*tokens.shape[:-1], 1], device=tokens.device, dtype=tokens.dtype)
        ts_logits = torch.zeros_like(ts_tokens)
        try:
            for i in range(self.sample_len):
                if alpha:
                    logits = self.inference.logits(tokens,
                                                   audio_features * (torch.rand_like(audio_features) * alpha + 1))
                else:
                    logits = self.inference.logits(tokens, audio_features)

                if i == 0 and get_new_attrs(self.tokenizer, 'no_captions') is not None:  # save no_speech_probs
                    probs_at_sot = logits[:, self.sot_index].float().softmax(dim=-1)
                    no_speech_probs = probs_at_sot[:, get_new_attrs(self.tokenizer, 'no_captions')].tolist()

                # now we need to consider the logits at the last token only
                logits = logits[:, -1]

                logits_clone = torch.clone(logits)
                if suppress_word_ts and suppress_ts_mask is not None:
                    logits_clone[:, self.tokenizer.timestamp_begin:][:, suppress_ts_mask] = -np.inf
                logits_clone[:, : self.tokenizer.timestamp_begin] = -np.inf
                temp_ts_logits, temp_ts_token = torch.topk(logits_clone, ts_num)
                ts_tokens = torch.cat([ts_tokens, temp_ts_token], -1)
                ts_logits = torch.cat([ts_logits, temp_ts_logits], -1)

                del logits_clone

                # if suppress_ts_mask is not None:
                #     logits[:, self.tokenizer.timestamp_begin:][suppress_ts_mask] = -np.inf

                # apply the logit filters, e.g. for suppressing or applying penalty to
                for logit_filter in self.logit_filters:
                    logit_filter.apply(logits, tokens)

                if suppress_ts_mask is not None:
                    logits[:, self.tokenizer.timestamp_begin:][:, suppress_ts_mask] = -np.inf

                # expand the tokens tensor with the selected next tokens
                tokens, completed = self.decoder.update(tokens, logits, sum_logprobs)

                if completed or tokens.shape[-1] > self.n_ctx:
                    break
        finally:
            self.inference.cleanup_caching()
            new_ts_token_count = tokens.shape[-1] - initial_tk_len
            ts_tokens = ts_tokens[..., 1:].reshape(
                [*tokens.shape[:-1], new_ts_token_count, ts_num])
            ts_logits = ts_logits[..., 1:].reshape(
                [*tokens.shape[:-1], new_ts_token_count, ts_num])

        return tokens, sum_logprobs, no_speech_probs, ts_tokens, ts_logits

    # modified version of whisper.DecodingTask.run
    @torch.no_grad()
    def run(self, mel: Tensor, ts_num: int = None, alpha: float = None, suppress_ts_mask: Tensor = None,
            suppress_word_ts=False) \
            -> Union[List[DecodingResult], Tuple[List[DecodingResult], List[List[int]], List[List[int]]]]:
        self.decoder.reset()
        tokenizer: Tokenizer = self.tokenizer
        n_audio: int = mel.shape[0]

        ts_num = 10 if ts_num is None else max(ts_num, 1)

        audio_features: Tensor = self._get_audio_features(mel)  # encoder forward pass
        tokens: Tensor = torch.tensor([self.initial_tokens]).expand(n_audio, -1)

        # detect language if requested, overwriting the language token
        languages, language_probs = self._detect_language(audio_features, tokens)
        if self.options.task == "lang_id":
            return [
                DecodingResult(audio_features=features, language=language, language_probs=probs)
                for features, language, probs in zip(audio_features, languages, language_probs)
            ]

        # repeat the audio & text tensors by the group size, for beam search or best-of-n sampling
        audio_features = audio_features.repeat_interleave(self.n_group, dim=0)
        tokens = tokens.repeat_interleave(self.n_group, dim=0).to(audio_features.device)

        # call the main sampling loop
        tokens, sum_logprobs, no_speech_probs, ts_tokens, ts_logits = self._main_loop(audio_features, tokens,
                                                                                      ts_num=ts_num, alpha=alpha,
                                                                                      suppress_ts_mask=suppress_ts_mask,
                                                                                      suppress_word_ts=suppress_word_ts)

        # reshape the tensors to have (n_audio, n_group) as the first two dimensions
        audio_features = audio_features[:: self.n_group]
        no_speech_probs = no_speech_probs[:: self.n_group]
        assert audio_features.shape[0] == len(no_speech_probs) == n_audio

        tokens = tokens.reshape(n_audio, self.n_group, -1)
        ts_tokens = ts_tokens.reshape(n_audio, self.n_group, -1, ts_num)
        ts_logits = ts_logits.reshape(n_audio, self.n_group, -1, ts_num)
        sum_logprobs = sum_logprobs.reshape(n_audio, self.n_group)

        # get the final candidates for each group, and slice between the first sampled token and EOT
        tokens, sum_logprobs = self.decoder.finalize(tokens, sum_logprobs)
        tokens: List[List[Tensor]] = [
            [t[self.sample_begin: (t == tokenizer.eot).nonzero()[0, 0]] for t in s] for s in tokens
        ]
        ts_tokens: List[List[Tensor]] = [[t[:len(tokens[i][j])] for j, t in enumerate(s)] for i, s in
                                         enumerate(ts_tokens)]
        ts_logits: List[List[Tensor]] = [[t[:len(tokens[i][j])] for j, t in enumerate(s)] for i, s in
                                         enumerate(ts_logits)]

        # select the top-ranked sample in each group
        selected = self.sequence_ranker.rank(tokens, sum_logprobs)
        tokens: List[List[int]] = [t[i].tolist() for i, t in zip(selected, tokens)]
        ts_tokens: List[List[int]] = [t[i].tolist() for i, t in zip(selected, ts_tokens)]
        ts_logits: List[List[int]] = [t[i].tolist() for i, t in zip(selected, ts_logits)]
        texts: List[str] = [tokenizer.decode(t).strip() for t in tokens]

        sum_logprobs: List[float] = [lp[i] for i, lp in zip(selected, sum_logprobs)]
        avg_logprobs: List[float] = [lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)]

        fields = (texts, languages, tokens, audio_features, avg_logprobs, no_speech_probs)
        if len(set(map(len, fields))) != 1:
            raise RuntimeError(f"inconsistent result lengths: {list(map(len, fields))}")

        return [
                   DecodingResult(
                       audio_features=features,
                       language=language,
                       tokens=tokens,
                       text=text,
                       avg_logprob=avg_logprob,
                       **(dict(no_caption_prob=no_speech_prob) if hasattr(DecodingResult, 'no_caption_prob') else dict(
                           no_speech_prob=no_speech_prob)),
                       temperature=self.options.temperature,
                       compression_ratio=compression_ratio(text),
                   )
                   for text, language, tokens, features, avg_logprob, no_speech_prob in zip(*fields)
               ], ts_tokens, ts_logits


# modified version of whisper.decoding.decode
@torch.no_grad()
def decode_word_level(model: "Whisper", mel: Tensor, options: DecodingOptions = DecodingOptions(),
                      ts_num: int = None, alpha: float = None, suppress_ts_mask: Tensor = None,
                      suppress_word_ts=False) -> \
        Union[DecodingResult, List[DecodingResult], tuple]:
    """
    Performs decoding of 30-second audio segment(s), provided as Mel spectrogram(s).

    Parameters
    ----------
    model: Whisper
        the Whisper model instance

    mel: torch.Tensor, shape = (80, 3000) or (*, 80, 3000)
        A tensor containing the Mel spectrogram(s)

    options: DecodingOptions
        A dataclass that contains all necessary options for decoding 30-second segments

    ts_num: int
        Number of additional top timestamp predictions to save for each word for postprocessing stabilization (default: 5).

    alpha: float
        Amount of noise to add to audio to produce slightly difference results.
        audio_features *= torch.rand_like(audio_features) * alpha + 1

    suppress_ts_mask: (list, Tensor)
        Mask suppress to timestamp token(s) for decoding

    suppress_word_ts: bool
        Use suppress_ts_mask to suppress timestamp tokens of words

    Returns
    -------
    result: Union[DecodingResult, List[DecodingResult]]
        The result(s) of decoding contained in `DecodingResult` dataclass instance(s)
    """
    single = mel.ndim == 2
    if single:
        mel = mel.unsqueeze(0)

    result, ts_tokens, ts_logits = DecodingTaskWordLevel(model, options).run(mel, ts_num=ts_num,
                                                                             alpha=alpha,
                                                                             suppress_ts_mask=suppress_ts_mask,
                                                                             suppress_word_ts=suppress_word_ts)

    if single:
        result = result[0]

    return result, ts_tokens, ts_logits


def modify_model(model: whisper.model.Whisper):
    model.decode = MethodType(decode_word_level, model)
    model.transcribe = MethodType(transcribe_word_level, model)

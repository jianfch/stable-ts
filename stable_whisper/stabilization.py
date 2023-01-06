import warnings
from copy import deepcopy
from itertools import chain
from typing import Union, List, Tuple
import numpy as np
from whisper.tokenizer import Tokenizer


MIN_DUR = 0.02


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


def group_word_timestamps(res: (dict, List[dict]), one_group=True, combine_compound=False,
                          ts_key: str = None, min_dur: float = None):
    if ts_key is None:
        ts_key = 'whole_word_timestamps'

    if min_dur is None:
        min_dur = MIN_DUR

    assert min_dur > 0, 'min_dur must be greater than 0'

    def group_ts(ts_: List[dict], start) -> List[dict]:
        group0: List[dict] = []
        for w_ts_i, w_ts in enumerate(ts_):
            curr_end = w_ts['timestamp']
            if group0:
                curr_start = group0[-1]['end']
                if not combine_compound or w_ts['word'].startswith(' '):
                    if group0[-1]['end'] - group0[-1]['start'] >= min_dur:
                        curr_dur = curr_end - curr_start
                        prev_word_len = len(group0[-1]['text'])
                        is_last = w_ts == ts_[-1]
                        next_dur = max(min_dur, curr_dur) if is_last else (ts_[w_ts_i + 1]['timestamp'] - curr_end)
                        next_word_len = prev_word_len if is_last else len(ts_[w_ts_i + 1]['word'])
                        if curr_dur >= min_dur or \
                                not is_last or \
                                (next_dur < min_dur) or \
                                next_dur < curr_dur or \
                                next_word_len < prev_word_len:
                            group0.append(dict(start=curr_start,
                                               end=curr_end,
                                               text=w_ts['word']))
                            continue

                group0[-1]['end'] = max(curr_start, curr_end)
                group0[-1]['text'] += w_ts['word']
            else:
                group0.append(dict(start=start,
                                   end=curr_end,
                                   text=w_ts['word']))

        return group0

    def group_ts_final(first_group: List[dict]) -> List[dict]:

        group1: List[dict] = []
        prev_ts_dict_dur = 0
        for i, ts_dict in enumerate(first_group):
            ni = i + 1
            curr_ts_dict_dur = ts_dict['end'] - ts_dict['start']
            next_ts_dict_dur = first_group[ni]['end'] - first_group[ni]['start'] if ni < len(first_group) else 0
            merge_with_prev = curr_ts_dict_dur < min_dur and (
                    prev_ts_dict_dur < next_ts_dict_dur or ni == len(first_group))

            if merge_with_prev and group1:
                group1[-1]['end'] = ts_dict['end']
                group1[-1]['text'] += ts_dict['text']
            else:
                group1.append(ts_dict)

        return group1

    segs: List[dict] = res['segments'] if isinstance(res, dict) else res
    assert set(ts_key in seg for seg in segs) == {True}, f'input contains missing {ts_key}'

    grouped = (group_ts(seg[ts_key], seg['start']) for seg in segs)
    return group_ts_final(list(chain.from_iterable(grouped))) if one_group else list(grouped)


def tighten_timestamps(res: dict, end_at_last_word=True, end_before_period=False, start_at_first_word=False) -> dict:
    res = deepcopy(res)
    if not any((end_at_last_word, end_before_period, start_at_first_word)):
        return res
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
        warnings.warn('No segments found, whole-word timestamps cannot be added.', stacklevel=2)
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
                    if ((not merge_non_space or temp_whole_word.startswith(' ') or not whole_word_timestamps) and
                            temp_whole_word not in append_punctuations and
                            not has_prepend) or not len(whole_word_timestamps):
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
        warnings.warn(f'Failed to add whole-word timestamps to the following segments: {tuple(failed_idx)}',
                      stacklevel=2)

import warnings
import numpy as np
from whisper.tokenizer import Tokenizer, get_tokenizer
from itertools import chain
from typing import List, Union, Tuple
from copy import deepcopy
from .stabilization import stabilize_timestamps


def merge_segments(*segments: dict) -> dict:
    """
    Merge segments preserving chronology.
    """
    if len(segments) == 1:
        return segments[0]

    def get(key):
        return [i[key] for i in segments]

    def cat_ls(key):
        return list(chain.from_iterable(get(key)))

    def cat_token():
        if len(set(tuple(i['tokens']) for i in segments)) == len(segments):
            return cat_ls('tokens')
        return segments[0].get('tokens')

    def weighted_avg(key):
        word_count = [len(i['word_timestamps']) for i in segments]
        total_words = sum(word_count)
        return sum(count / total_words * seg[key] for seg, count in zip(segments, word_count))

    merged_segment = dict(
        id=segments[0]['id'],
        seek=segments[0]['seek'],
        offset=segments[0]['offset'],
        start=segments[0]['start'],
        end=segments[-1]['end'],
        text=''.join(get('text')),
        tokens=cat_token(),
        temperature=max(get('temperature')),
        avg_logprob=weighted_avg('avg_logprob'),
        compression_ratio=weighted_avg('compression_ratio'),
        no_speech_prob=weighted_avg('no_speech_prob'),
        alt_start_timestamps=segments[0]['alt_start_timestamps'],
        start_ts_logits=segments[0]['start_ts_logits'],
        alt_end_timestamps=segments[-1]['alt_end_timestamps'],
        end_ts_logits=segments[-1]['end_ts_logits'],
        unstable_word_timestamps=cat_ls('unstable_word_timestamps'),
        anchor_point=any(get('anchor_point')),
        word_timestamps=cat_ls('word_timestamps'),
        whole_word_timestamps=cat_ls('whole_word_timestamps')
    )
    if any('next_offset' in seg for seg in segments):
        if 'next_offset' in segments[-1]:
            merged_segment['next_offset'] = segments[-1]['next_offset']
        else:
            merged_segment['next_offset'] = segments[-1]['end']

    return merged_segment


def merge_no_dur_segments(segments: List[dict]):
    """
    Merge any segments with zero duration with an adjacent segment.
    """
    if len(segments) == 1:
        return
    zero_duration = [i for i, seg in enumerate(segments) if seg['end'] - seg['start'] == 0]
    if zero_duration:
        zero_duration_groups = []
        for i in zero_duration:
            if zero_duration_groups and zero_duration_groups[-1][-1] == i - 1:
                zero_duration_groups[-1].append(i)
            else:
                zero_duration_groups.append([i])
        for i, idxs in enumerate(zero_duration_groups):
            if idxs[0] != 0:
                next_idx = idxs[-1] + 1
                if len(segments) >= next_idx:
                    prev_shorter = True
                else:
                    prev_idx = idxs[0] - 1
                    prev_dur = segments[prev_idx]['end'] - segments[prev_idx]['start']
                    next_dur = segments[next_idx]['end'] - segments[next_idx]['start']
                    if prev_dur == next_dur:
                        prev_shorter = len(segments[prev_idx]['text']) < len(segments[next_idx]['text'])
                    else:
                        prev_shorter = prev_dur < next_dur
            else:
                prev_shorter = False

            if prev_shorter:
                zero_duration_groups[i] = [idxs[0] - 1] + idxs
            else:
                zero_duration_groups[i].append(idxs[-1] + 1)

        for idxs in reversed(zero_duration_groups):
            i = idxs[0]
            new_seg = merge_segments(*[segments.pop(i) for _ in idxs])
            segments.insert(i, new_seg)
        for i in range(len(segments)):
            segments[i]['id'] = i


def refine_word_level_ts(segments: Union[List[dict], dict],
                         *,
                         tokenizer: Union[Tokenizer, bool] = True,
                         ts_num: int = None,
                         segment_indices: List[int] = None,
                         stab_options: dict = None,
                         whole_word_options: dict = None,
                         average: bool = False) -> Union[List[dict], dict]:
    """
    Refined word timestamps with [more_word_timestamps] of [segments] from results.

    Parameters
    ----------
    segments: Union[List[dict], dict]
        [segments] from results or the result.
    tokenizer: Union[Tokenizer, bool]
        An instance of the tokenizer used for get decode of the segments results
        or boolean for whether the tokenizer is multilingual. (Default: True)
    ts_num: int
        Number of top timestamp predictions to save for each word for postprocessing stabilization.
        (Default use the amount found in the segments or 10 if none is found)
    segment_indices: List[int]
        Indices of the segment to refine timestamps. None means all segments will be refined.
    stab_options: dict
        Arguments for stable_whisper.stabilization.stabilize_timestamps.
        If any segments are missing [word_timestamps] stabilization will be redone then
        timestamps are stabilized again after the segments are update with new unstable timestamps from refinement.
    whole_word_options: dict
        Arguments for stable_whisper.stabilization.add_whole_word_ts.
        If any segments are contains [whole_word_timestamps] or if this argument is not empty,
        [whole_word_timestamps] will be recomputed after refinement.
    average: bool
        Whether to average the original word timestamps with the refined timestamps. (Default: False)

    Returns
    -------
    The original results/segment updated with refined word timestamps.
    """

    ori_results = None
    if isinstance(segments, dict):
        if 'segments' not in segments:
            raise KeyError(f'segments not found in input')
        if len(segments['segments']) == 0:
            return segments
        ori_results = segments
        segments = ori_results['segments']

    if len(segments) == 0:
        return segments

    has_more_ts = set('more_word_timestamps' in seg for seg in segments)
    if len(has_more_ts) == 2:
        warnings.warn('Some segments are missing [more_word_timestamps]. Those segments will be skipped.',
                      stacklevel=2)
    elif has_more_ts == {False}:
        raise NotImplementedError('Cannot refine segments that all are missing [more_word_timestamps].')

    if isinstance(tokenizer, bool):
        tokenizer = get_tokenizer(tokenizer)

    def _filter_max_min(vals, mn, mx, other=None):
        for val_idx, val in enumerate(vals):
            mask = np.logical_and(mx > val, val > mn)
            new_val = val[mask]
            if new_val.shape[0]:
                vals[val_idx] = new_val
                if other is not None:
                    other[val_idx] = other[val_idx][mask]
        if other is None:
            return vals
        return vals, other

    def _refine_sequential(more_wts: Union[List[List[float]], List[np.ndarray]],
                           more_wts_logits: Union[List[List[float]], List[np.ndarray]] = None,
                           imin: float = None,
                           imax: float = None) -> Tuple[List[List[float]], List[List[float]]]:

        if isinstance(more_wts[0], list):
            more_wts = list(map(np.array, more_wts))

        if isinstance(more_wts_logits[0], list):
            more_wts_logits = list(map(np.array, more_wts_logits))

        if imin is None:
            imin = more_wts[0][:ts_num].min()
        if imax is None:
            imax = more_wts[-1][:ts_num].max()

        more_wts = deepcopy(more_wts)
        more_wts_logits = deepcopy(more_wts_logits)

        prev_i = []
        while len(prev_i) < len(more_wts):
            vs = [np.inf if wts_i in prev_i else np.var(wts[:ts_num]) for wts_i, wts in enumerate(more_wts)]
            target_i = vs.index(min(vs))
            prev_i.append(target_i)
            mid = more_wts[target_i][:ts_num].mean()
            more_wts[:target_i], more_wts_logits[:target_i] = \
                _filter_max_min(more_wts[:target_i], imin, mid, more_wts_logits[:target_i])
            more_wts[target_i+1:], more_wts_logits[target_i+1:] = \
                _filter_max_min(more_wts[target_i+1:], mid, imax, more_wts_logits[target_i+1:])

        return more_wts, more_wts_logits

    if any(not seg.get('word_timestamps') for seg in segments):
        segments = stabilize_timestamps(segments, **(stab_options or {}))

    segments = deepcopy(segments)
    # merge_no_dur_segments(segments)

    add_whole_words = whole_word_options is not None

    for idx, seg in enumerate(segments):
        if segment_indices is not None and idx not in segment_indices:
            continue

        if 'whole_word_timestamps' in seg and not add_whole_words:
            add_whole_words = True

        curr_ts_num = len(seg['unstable_word_timestamps'][0]['timestamps']) \
            if ts_num is None or seg.get('unstable_word_timestamps') else (ts_num or 10)

        start = seg['start']
        end = seg['end']

        seg['more_word_timestamps'], seg['more_word_ts_logits'] = \
            _refine_sequential(seg['more_word_timestamps'], seg.get('more_word_ts_logits'), imin=start, imax=end)

        for i, unstab_ts in enumerate(seg['unstable_word_timestamps']):
            unstab_ts['timestamps'] = seg['more_word_timestamps'][i][:curr_ts_num].tolist()
            unstab_ts['timestamp_logits'] = seg['more_word_ts_logits'][i][:curr_ts_num].tolist()

        seg['refined'] = True

    for idx, seg in enumerate(stabilize_timestamps(segments, **(stab_options or {}))):
        new_wtss = np.array([w['timestamp'] for w in seg['word_timestamps']])
        old_wtss = np.array([w['timestamp'] for w in segments[idx]['word_timestamps']])
        if len(new_wtss) < 2 or len(old_wtss) < 2:
            continue

        final_wtss = new_wtss if average or (new_wtss[1]-seg['end'] <= old_wtss[1]-seg['end']) else old_wtss
        for wi, fw in enumerate(final_wtss):
            if average:
                segments[idx]['word_timestamps'][wi]['timestamp'] = \
                    (fw + segments[idx]['word_timestamps'][wi]['timestamp']) / 2
            else:
                segments[idx]['word_timestamps'][wi]['timestamp'] = fw
            segments[idx]['more_word_timestamps'][wi] = segments[idx]['more_word_timestamps'][wi].tolist()
            if segments[idx].get('more_word_ts_logits') is not None:
                segments[idx]['more_word_ts_logits'][wi] = segments[idx]['more_word_ts_logits'][wi].tolist()

    if add_whole_words:
        from .stabilization import add_whole_word_ts
        add_whole_word_ts(tokenizer, segments, **(whole_word_options or {}))

    if ori_results is not None:
        ori_results['segments'] = segments
        return ori_results

    return segments

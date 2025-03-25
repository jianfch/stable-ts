import warnings
import re
import torch
import numpy as np
from typing import Union, List, Tuple, Optional, Callable
from copy import deepcopy
from itertools import chain

from tqdm import tqdm

from .stabilization import suppress_silence, get_vad_silence_func, VAD_SAMPLE_RATES
from .stabilization.nonvad import audio2timings
from .stabilization.utils import filter_timings
from .text_output import *
from .utils import str_to_valid_type, format_timestamp, UnsortedException
from .audio.utils import audio_to_tensor_resample
from .default import get_min_word_dur, get_append_punctuations, get_prepend_punctuations


__all__ = ['WhisperResult', 'Segment', 'WordTiming']


def _combine_attr(obj: object, other_obj: object, attr: str):
    if (val := getattr(obj, attr)) is not None:
        other_val = getattr(other_obj, attr)
        if isinstance(val, list):
            if other_val is None:
                setattr(obj, attr, None)
            else:
                val.extend(other_val)
        else:
            new_val = None if other_val is None else ((val + other_val) / 2)
            setattr(obj, attr, new_val)


def _increment_attr(obj: object, attr: str, val: Union[int, float]):
    if (curr_val := getattr(obj, attr, None)) is not None:
        setattr(obj, attr, curr_val + val)


def _round_timestamp(ts: Union[float, None]):
    if not ts:
        return ts
    return round(ts, 3)


def _store_content(obj: object, content: Union[Callable, object]) -> str:
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    key = content.__repr__().replace('_', '-')
    if not key.startswith('<') and not key.endswith('>'):
        key = f'<{key}>'
    if not hasattr(obj, '_content_cache'):
        setattr(obj, '_content_cache', {})
    getattr(obj, '_content_cache')[key] = content
    return key


def _get_content(obj: object, content: Union[Callable, object, str], strict=True) -> Union[Callable, object, str]:
    if isinstance(content, str) and content.startswith('<') and content.endswith('>'):
        curr_variable = {'<True>': True, '<False>': False}.get(content, None)
        if curr_variable is None and hasattr(obj, '_content_cache'):
            curr_variable = getattr(obj, '_content_cache').get(content, None)
        if curr_variable is None:
            if strict:
                raise NameError(f'{content.replace("-", "_")} not found')
            return content
        content = curr_variable
    return content


class WordTiming:

    def __init__(
            self,
            word: str,
            start: float,
            end: float,
            probability: Optional[float] = None,
            tokens: Optional[List[int]] = None,
            left_locked: bool = False,
            right_locked: bool = False,
            segment_id: Optional[int] = None,
            id: Optional[int] = None,
            segment: Optional['Segment'] = None,
            round_ts: bool = True,
            ignore_unused_args: bool = False
    ):
        if not ignore_unused_args and segment_id is not None:
            warnings.warn('The parameter ``segment_id`` is ignored. '
                          'Specify the current segment instance with ``segment``.',
                          stacklevel=2)
        self.round_ts = round_ts
        self.word = word
        self._start = self.round(start)
        self._end = self.round(end)
        self.probability = probability
        self.tokens = tokens
        self.left_locked = left_locked
        self.right_locked = right_locked
        self.segment = segment
        self.id = id

    def __repr__(self):
        return f'WordTiming(start={self.start}, end={self.end}, word="{self.word}")'

    def __len__(self):
        return len(self.word)

    def __add__(self, other: 'WordTiming'):
        self_copy = WordTiming(
            word=self.word + other.word,
            start=min(self.start, other.start),
            end=max(self.end, other.end),
            probability=self.probability,
            tokens=None if self.tokens is None else self.tokens.copy(),
            left_locked=self.left_locked or other.left_locked,
            right_locked=self.right_locked or other.right_locked,
            id=self.id,
            segment=self.segment
        )
        _combine_attr(self_copy, other, 'probability')
        _combine_attr(self_copy, other, 'tokens')

        return self_copy

    def __deepcopy__(self, memo=None):
        return self.copy(copy_tokens=True)

    def __copy__(self):
        return self.copy()

    def copy(
            self,
            keep_segment: bool = False,
            copy_tokens: bool = False
    ):
        return WordTiming(
            word=self.word,
            start=self.start,
            end=self.end,
            probability=self.probability,
            tokens=None if (self.tokens is None) else (self.tokens.copy() if copy_tokens else self.tokens),
            left_locked=self.left_locked,
            right_locked=self.right_locked,
            id=self.id,
            segment=self.segment if keep_segment else None,
            round_ts=self.round_ts
        )

    def round(self, timestamp: float) -> float:
        if not self.round_ts:
            return timestamp
        return _round_timestamp(timestamp)

    def to_display_str(self):
        return f"[{format_timestamp(self.start)}] -> [{format_timestamp(self.end)}] \"{self.word}\""

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @start.setter
    def start(self, val):
        self._start = self.round(val)

    @end.setter
    def end(self, val):
        self._end = self.round(val)

    @property
    def segment_id(self):
        return None if self.segment is None else self.segment.id

    @property
    def duration(self):
        return self.round(self.end - self.start)

    def round_all_timestamps(self):
        warnings.warn('``.round_all_timestamps()`` is deprecated and will be removed in future versions. '
                      'Use ``.round_ts=True`` to round timestamps by default instead.',
                      stacklevel=2)
        self.round_ts = True

    def offset_time(self, offset_seconds: float):
        self.start = self.start + offset_seconds
        self.end = self.end + offset_seconds

    def to_dict(self):
        return dict(
            word=self.word,
            start=self.start,
            end=self.end,
            probability=self.probability,
            tokens=None if self.tokens is None else self.tokens.copy()
        )

    def lock_left(self):
        self.left_locked = True

    def lock_right(self):
        self.right_locked = True

    def lock_both(self):
        self.lock_left()
        self.lock_right()

    def unlock_both(self):
        self.left_locked = False
        self.right_locked = False

    def suppress_silence(self,
                         silent_starts: np.ndarray,
                         silent_ends: np.ndarray,
                         min_word_dur: Optional[float] = None,
                         nonspeech_error: float = 0.3,
                         keep_end: Optional[bool] = True):
        suppress_silence(self, silent_starts, silent_ends, min_word_dur, nonspeech_error, keep_end)
        return self

    def rescale_time(self, scale_factor: float):
        self.start = self.start * scale_factor
        self.end = self.end * scale_factor

    def clamp_max(self, max_dur: float, clip_start: bool = False, verbose: bool = False):
        if self.duration > max_dur:
            if clip_start:
                new_start = round(self.end - max_dur, 3)
                if verbose:
                    print(f'Start: {self.start} -> {new_start}\nEnd: {self.end}\nText:"{self.word}"\n')
                self.start = new_start

            else:
                new_end = round(self.start + max_dur, 3)
                if verbose:
                    print(f'Start: {self.start}\nEnd: {self.end} -> {new_end}\nText:"{self.word}"\n')
                self.end = new_end

    def set_segment(self, segment: 'Segment'):
        warnings.warn('``.set_segment(current_segment_instance)`` is deprecated and will be removed in future versions.'
                      ' Use ``.segment = current_segment`` instead.',
                      stacklevel=2)
        self.segment = segment

    def get_segment(self) -> Union['Segment', None]:
        """
        Return instance of :class:`stable_whisper.result.Segment` that this instance is a part of.
        """
        warnings.warn('``.get_segment()`` will be removed in future versions. Use ``.segment`` instead.',
                      stacklevel=2)
        return self.segment


def _words_by_lock(words: List[WordTiming], only_text: bool = False, include_single: bool = False):
    """
    Return a nested list of words such that each sublist contains words that are locked together.
    """
    all_words = []
    for word in words:
        if len(all_words) == 0 or not (all_words[-1][-1].right_locked or word.left_locked):
            all_words.append([word])
        else:
            all_words[-1].append(word)
    if only_text:
        all_words = list(map(lambda ws: list(map(lambda w: w.word, ws)), all_words))
    if not include_single:
        all_words = [ws for ws in all_words if len(ws) > 1]
    return all_words


class Segment:

    def __init__(
            self,
            start: Optional[float] = None,
            end: Optional[float] = None,
            text: Optional[str] = None,
            seek: Optional[float] = None,
            tokens: List[int] = None,
            temperature: Optional[float] = None,
            avg_logprob: Optional[float] = None,
            compression_ratio: Optional[float] = None,
            no_speech_prob: Optional[float] = None,
            words: Optional[Union[List[WordTiming], List[dict]]] = None,
            id: Optional[int] = None,
            result: Optional["WhisperResult"] = None,
            round_ts: bool = True,
            ignore_unused_args: bool = False
    ):
        if words:
            if ignore_unused_args:
                start = end = text = tokens = None
            else:
                if (start or end) is not None:
                    warnings.warn('Arguments for ``start`` and ``end`` will be ignored '
                                  'and the ``start`` and ``end`` will taken from the first and last ``words``.',
                                  stacklevel=2)
                if text is not None:
                    warnings.warn('The argument for ``text`` will be ignored '
                                  'and it will always be the concatenation of text in ``words``',
                                  stacklevel=2)
                if tokens is not None:
                    warnings.warn('The argument for ``tokens`` will be ignored '
                                  'and it will always be the concatenation of tokens in ``words``',
                                  stacklevel=2)
        self.round_ts = round_ts
        self._default_start = self.round(start) if start else 0.0
        self._default_end = self.round(end) if end else 0.0
        self._default_text = text or ''
        self._default_tokens = tokens or []
        self.seek = seek
        self.temperature = temperature
        self.avg_logprob = avg_logprob
        self.compression_ratio = compression_ratio
        self.no_speech_prob = no_speech_prob
        self.words = words
        if self.words and isinstance(words[0], dict):
            self.words = [
                WordTiming(
                    **word,
                    segment=self,
                    round_ts=self.round_ts,
                    ignore_unused_args=True
                ) for word in self.words
            ]
        self.id = id
        self._reversed_text = False
        self.result = result

    def __repr__(self):
        return f'Segment(start={self.start}, end={self.end}, text="{self.text}")'

    def __getitem__(self, index: int) -> WordTiming:
        if self.words is None:
            raise ValueError('segment contains no words')
        return self.words[index]

    def __delitem__(self, index: int):
        if self.words is None:
            raise ValueError('segment contains no words')
        del self.words[index]
        self.reassign_ids(index)

    def __deepcopy__(self, memo=None):
        return self.copy(copy_words=True, copy_tokens=True)

    def __copy__(self):
        return self.copy()

    def copy(
            self,
            new_words: Optional[List[WordTiming]] = None,
            keep_result: bool = False,
            copy_words: bool = False,
            copy_tokens: bool = False
    ):
        if new_words is None:
            if self.has_words:
                words = [w.copy(copy_tokens=copy_tokens) for w in self.words] if copy_words else self.words
            else:
                words = None
            def_start = self._default_start
            def_end = self._default_end
            def_text = self._default_text
            def_tokens = self._default_tokens
        else:
            words = [w.copy(copy_tokens=copy_tokens) for w in new_words] if copy_words else new_words
            def_start = def_end = def_text = def_tokens = None
        new_seg = Segment(
            start=def_start,
            end=def_end,
            text=def_text,
            seek=self.seek,
            tokens=def_tokens,
            temperature=self.temperature,
            avg_logprob=self.avg_logprob,
            compression_ratio=self.compression_ratio,
            no_speech_prob=self.no_speech_prob,
            words=words,
            id=self.id,
            result=self.result if keep_result else None,
            round_ts=self.round_ts,
            ignore_unused_args=True
        )
        return new_seg

    def round(self, timestamp: float) -> float:
        if not self.round_ts:
            return timestamp
        return _round_timestamp(timestamp)

    def to_display_str(self, only_segment: bool = False):
        line = f'[{format_timestamp(self.start)} --> {format_timestamp(self.end)}] "{self.text}"'
        if self.has_words and not only_segment:
            line += '\n' + '\n'.join(
                f"-{w.to_display_str()}" for w in self.words
            ) + '\n'
        return line

    @property
    def has_words(self):
        return bool(self.words)

    @property
    def ori_has_words(self):
        return self.words is not None

    @property
    def start(self):
        if self.has_words:
            return self.words[0].start
        return self._default_start

    @property
    def end(self):
        if self.has_words:
            return self.words[-1].end
        return self._default_end

    @start.setter
    def start(self, val):
        if self.has_words:
            self.words[0].start = val
            return
        self._default_start = self.round(val)

    @end.setter
    def end(self, val):
        if self.has_words:
            self.words[-1].end = val
            return
        self._default_end = self.round(val)

    @property
    def text(self) -> str:
        if self.has_words:
            return ''.join(word.word for word in self.words)
        return self._default_text

    @property
    def tokens(self) -> List[int]:
        if self.has_words and self.words[0].tokens:
            return list(chain.from_iterable(word.tokens for word in self.words))
        return self._default_tokens

    @property
    def duration(self):
        return self.end - self.start

    def word_count(self):
        if self.has_words:
            return len(self.words)
        return -1

    def char_count(self):
        if self.has_words:
            return sum(len(w) for w in self.words)
        return len(self.text)

    def add(self, other: 'Segment', copy_words: bool = False, newline: bool = False, reassign_ids: bool = True):
        if self.ori_has_words == other.ori_has_words:
            words = (self.words + other.words) if self.ori_has_words else None
        else:
            self_state = 'with' if self.ori_has_words else 'without'
            other_state = 'with' if other.ori_has_words else 'without'
            raise ValueError(f"Can't merge segment {self_state} words and a segment {other_state} words.")

        self_copy = self.copy(words, copy_words=copy_words)
        _combine_attr(self_copy, other, 'temperature')
        _combine_attr(self_copy, other, 'avg_logprob')
        _combine_attr(self_copy, other, 'compression_ratio')
        _combine_attr(self_copy, other, 'no_speech_prob')

        self_copy._default_end = other._default_end
        self_copy._default_text += other._default_text
        self_copy._default_tokens += other._default_tokens

        if newline:
            if self_copy.has_words:
                if not self_copy.words[len(self.words)-1].word.endswith('\n'):
                    self_copy.words[len(self.words)-1].word += '\n'
            else:
                if self_copy.text[len(self.text)-1] != '\n':
                    self_copy._default_text = self_copy.text[:len(self.text)] + '\n' + self_copy.text[len(self.text):]
        if reassign_ids:
            self_copy.reassign_ids()
        return self_copy

    def __add__(self, other: 'Segment'):
        return self.add(other, copy_words=True)

    def _word_operations(self, operation: str, *args, **kwargs):
        if self.has_words:
            for w in self.words:
                getattr(w, operation)(*args, **kwargs)

    def round_all_timestamps(self):
        warnings.warn('``.round_all_timestamps()`` is deprecated and will be removed in future versions. '
                      'Use ``.round_ts=True`` to round timestamps by default instead.',
                      stacklevel=2)
        self.round_ts = True

    def offset_time(self, offset_seconds: float):
        if self.seek is not None:
            self.seek += offset_seconds
        if self.has_words:
            self._word_operations('offset_time', offset_seconds)
        else:
            self.start = self.start + offset_seconds
            self.end = self.end + offset_seconds

    def add_words(self, index0: int, index1: int, inplace: bool = False):
        if self.has_words:
            new_word = self.words[index0] + self.words[index1]
            if inplace:
                i0, i1 = sorted([index0, index1])
                self.words[i0] = new_word
                del self.words[i1]
            return new_word

    def rescale_time(self, scale_factor: float):
        if self.seek is not None:
            self.seek *= scale_factor
        if self.has_words:
            self._word_operations('rescale_time', scale_factor)
        else:
            self.start = self.start * scale_factor
            self.end = self.end * scale_factor

    def apply_min_dur(self, min_dur: float, inplace: bool = False):
        """
        Merge any word with adjacent word if its duration is less than ``min_dur``.
        """
        segment = self if inplace else deepcopy(self)
        if not self.has_words:
            return segment
        max_i = len(segment.words) - 1
        if max_i == 0:
            return segment
        for i in reversed(range(len(segment.words))):
            if max_i == 0:
                break
            if segment.words[i].duration < min_dur:
                if i == max_i:
                    segment.add_words(i-1, i, inplace=True)
                elif i == 0:
                    segment.add_words(i, i+1, inplace=True)
                else:
                    if segment.words[i+1].duration < segment.words[i-1].duration:
                        segment.add_words(i-1, i, inplace=True)
                    else:
                        segment.add_words(i, i+1, inplace=True)
                max_i -= 1
        return segment

    def _to_reverse_text(
            self,
            prepend_punctuations: Optional[str] = None,
            append_punctuations: Optional[str] = None,
    ):
        """
        Return a copy with words reversed order per segment.
        """
        warnings.warn('``_to_reverse_text()`` is deprecated and will be removed in future versions.',
                      category=DeprecationWarning, stacklevel=2)
        prepend_punctuations = get_prepend_punctuations(prepend_punctuations)
        if prepend_punctuations and ' ' not in prepend_punctuations:
            prepend_punctuations += ' '
        append_punctuations = get_append_punctuations(append_punctuations)
        self_copy = self.copy(copy_words=True)
        has_prepend = bool(prepend_punctuations)
        has_append = bool(append_punctuations)
        if has_prepend or has_append:
            word_objs = (
                self_copy.words
                if self_copy.has_words else
                [WordTiming(w, 0, 1, 0) for w in self_copy.text.split(' ')]
            )
            for word in word_objs:
                new_append = ''
                if has_prepend:
                    for _ in range(len(word)):
                        char = word.word[0]
                        if char in prepend_punctuations:
                            new_append += char
                            word.word = word.word[1:]
                        else:
                            break
                new_prepend = ''
                if has_append:
                    for _ in range(len(word)):
                        char = word.word[-1]
                        if char in append_punctuations:
                            new_prepend += char
                            word.word = word.word[:-1]
                        else:
                            break
                word.word = f'{new_prepend}{word.word}{new_append[::-1]}'
            self_copy._default_text = ''.join(w.word for w in reversed(word_objs))

        return self_copy

    def to_dict(self, reverse_text: Union[bool, tuple] = False):
        if reverse_text:
            warnings.warn('``reverse_text=True`` is deprecated and will be removed in future versions. '
                          'RTL text playback issues are caused by the video player incorrectly parsing tags '
                          '(note: tags come from ``segment_level=True + word_level=True``).')
            segment = self._to_reverse_text(*(reverse_text if reverse_text else []))
        else:
            segment = self

        seg_dict = dict(
            start=segment.start,
            end=segment.end,
            text=segment.text,
            seek=segment.seek,
            tokens=None if segment.tokens is None else segment.tokens.copy(),
            temperature=segment.temperature,
            avg_logprob=segment.avg_logprob,
            compression_ratio=segment.compression_ratio,
            no_speech_prob=segment.no_speech_prob,
        )

        if segment.has_words:
            seg_dict['words'] = [w.to_dict() for w in segment.words]
        elif segment.ori_has_words:
            seg_dict['words'] = []
        if reverse_text:
            seg_dict['reversed_text'] = True
        return seg_dict

    def words_by_lock(self, only_text: bool = True, include_single: bool = False):
        return _words_by_lock(self.words, only_text=only_text, include_single=include_single)

    @property
    def left_locked(self):
        if self.has_words:
            return self.words[0].left_locked
        return False

    @property
    def right_locked(self):
        if self.has_words:
            return self.words[-1].right_locked
        return False

    def lock_left(self):
        if self.has_words:
            self.words[0].lock_left()

    def lock_right(self):
        if self.has_words:
            self.words[-1].lock_right()

    def lock_both(self):
        self.lock_left()
        self.lock_right()

    def unlock_all_words(self):
        self._word_operations('unlock_both')

    def reassign_ids(self, start: Optional[int] = None):
        if self.has_words:
            for i, word in enumerate(self.words[start:], start or 0):
                word.segment = self
                word.id = i

    def update_seg_with_words(self):
        warnings.warn('Attributes that required updating are now properties based on the ``words`` except for ``id``. '
                      '``update_seg_with_words()`` is deprecated and will be removed in future versions. '
                      'Use ``.reassign_ids()`` to manually update ids',
                      stacklevel=2)
        self.reassign_ids()

    def suppress_silence(self,
                         silent_starts: np.ndarray,
                         silent_ends: np.ndarray,
                         min_word_dur: Optional[float] = None,
                         word_level: bool = True,
                         nonspeech_error: float = 0.3,
                         use_word_position: bool = True):
        min_word_dur = get_min_word_dur(min_word_dur)
        if self.has_words:
            ending_punctuations = get_append_punctuations()
            words = self.words if word_level or len(self.words) == 1 else [self.words[0], self.words[-1]]
            for i, w in enumerate(words, 1):
                if use_word_position:
                    keep_end = not (w.word[-1] in ending_punctuations or i == len(words))
                else:
                    keep_end = None
                w.suppress_silence(silent_starts, silent_ends, min_word_dur, nonspeech_error, keep_end)
        else:
            suppress_silence(self,
                             silent_starts,
                             silent_ends,
                             min_word_dur,
                             nonspeech_error)

        return self

    def get_locked_indices(self):
        locked_indices = [i
                          for i, (left, right) in enumerate(zip(self.words[1:], self.words[:-1]))
                          if left.left_locked or right.right_locked]
        return locked_indices

    def get_gaps(self, as_ndarray=False):
        if self.has_words:
            s_ts = np.array([w.start for w in self.words])
            e_ts = np.array([w.end for w in self.words])
            gap = s_ts[1:] - e_ts[:-1]
            return gap if as_ndarray else gap.tolist()
        return []

    def get_gap_indices(self, max_gap: float = 0.1):  # for splitting
        if not self.has_words or len(self.words) < 2:
            return []
        if max_gap is None:
            max_gap = 0
        indices = (self.get_gaps(True) > max_gap).nonzero()[0].tolist()
        return sorted(set(indices) - set(self.get_locked_indices()))

    def get_punctuation_indices(self, punctuation: Union[List[str], List[Tuple[str, str]], str]):  # for splitting
        if not self.has_words or len(self.words) < 2:
            return []
        if isinstance(punctuation, str):
            punctuation = [punctuation]
        indices = []
        for p in punctuation:
            if isinstance(p, str):
                for i, s in enumerate(self.words[:-1]):
                    if s.word.endswith(p):
                        indices.append(i)
                    elif i != 0 and s.word.startswith(p):
                        indices.append(i-1)
            else:
                ending, beginning = p
                indices.extend([i for i, (w0, w1) in enumerate(zip(self.words[:-1], self.words[1:]))
                                if w0.word.endswith(ending) and w1.word.startswith(beginning)])

        return sorted(set(indices) - set(self.get_locked_indices()))

    def _get_special_period_indices(self, extra_indices: Optional[List[int]] = None) -> List[int]:
        indices = [
            i for i, word in enumerate(self.words)
            if re.search('^[A-Z0-9]', word.word) is not None and
            not word.word.endswith('..') and
            len(re.sub('[.A-Z0-9]', '', word.word)) < 3
        ]
        if extra_indices:
            indices = sorted(set(indices + extra_indices))
        return indices

    def _get_even_split_indices(
            self,
            cumsum: np.ndarray,
            parts_per_split: np.ndarray,
            splits: np.ndarray,
            excluded_indices: Optional[List[int]] = None
    ):
        if excluded_indices:
            cumsum_idxs = np.arange(len(cumsum))
            max_word_i = len(self.words) - 1
            for i in sorted(set(excluded_indices)):
                if i == max_word_i:
                    break
                cumsum[i] = cumsum[i + 1]
                cumsum_idxs[i] = cumsum_idxs[i + 1]
            indices = [
                cumsum_idxs[np.abs(cumsum - (i * parts_per_split)).argmin()]
                for i in range(1, int(splits))
            ]
            indices = sorted(set(indices))
        else:
            indices = [
                np.abs(cumsum - (i * parts_per_split)).argmin()
                for i in range(1, int(splits))
            ]

        return indices

    def get_length_indices(
            self,
            max_chars: int = None,
            max_words: int = None,
            even_split: bool = True,
            include_lock: bool = False,
            ignore_special_periods: bool = False
    ):
        # for splitting
        if not self.has_words or (max_chars is None and max_words is None):
            return []
        assert max_chars != 0 and max_words != 0, \
            f'max_chars and max_words must be greater 0, but got {max_chars} and {max_words}'
        if len(self.words) < 2:
            return []
        indices = []
        locked_indices = self.get_locked_indices() if include_lock else []
        if ignore_special_periods:
            locked_indices = self._get_special_period_indices(locked_indices)
        if even_split:
            char_count = -1 if max_chars is None else sum(map(len, self.words))
            word_count = -1 if max_words is None else len(self.words)
            exceed_chars = max_chars is not None and char_count > max_chars
            exceed_words = max_words is not None and word_count > max_words
            if exceed_chars:
                splits = np.ceil(char_count / max_chars)
                chars_per_split = char_count / splits
                cum_char_count = np.cumsum([len(w.word) for w in self.words[:-1]])
                indices = self._get_even_split_indices(
                    cum_char_count,
                    chars_per_split,
                    splits,
                    locked_indices
                )
                if max_words is not None:
                    exceed_words = any(j-i+1 > max_words for i, j in zip([0]+indices, indices+[len(self.words)]))

            if exceed_words:
                splits = np.ceil(word_count / max_words)
                words_per_split = word_count / splits
                cum_word_count = np.array(range(1, len(self.words)+1))
                indices = self._get_even_split_indices(
                    cum_word_count,
                    words_per_split,
                    splits,
                    locked_indices
                )

        else:
            curr_words = 0
            curr_chars = 0
            for i, word in enumerate(self.words):
                curr_words += 1
                curr_chars += len(word)
                if i != 0:
                    if (
                            max_chars is not None and curr_chars > max_chars
                            or
                            max_words is not None and curr_words > max_words
                    ) and i-1 not in locked_indices:
                        indices.append(i-1)
                        curr_words = 1
                        curr_chars = len(word)
        return indices

    def get_duration_indices(
            self,
            max_dur: float,
            even_split: bool = True,
            include_lock: bool = False,
            ignore_special_periods: bool = False
    ):
        if not self.has_words or (total_duration := np.sum([w.duration for w in self.words])) <= max_dur:
            return []
        locked_indices = self.get_locked_indices() if include_lock else []
        if ignore_special_periods:
            locked_indices = self._get_special_period_indices(locked_indices)
        if even_split:
            splits = np.ceil(total_duration / max_dur)
            dur_per_split = total_duration / splits
            cum_dur = np.cumsum([w.duration for w in self.words[:-1]])
            indices = self._get_even_split_indices(
                cum_dur,
                dur_per_split,
                splits,
                locked_indices
            )
        else:
            indices = []
            curr_total_dur = 0.0
            for i, word in enumerate(self.words):
                curr_total_dur += word.duration
                if i != 0:
                    if curr_total_dur > max_dur and i - 1 not in locked_indices:
                        indices.append(i - 1)
                        curr_total_dur = word.duration
        return indices

    def split(self, indices: List[int], reassign_ids: bool = True):
        if len(indices) == 0:
            return []
        if indices[-1] != len(self.words) - 1:
            indices.append(len(self.words) - 1)
        seg_copies = []
        prev_i = 0
        for i in indices:
            i += 1
            new_words = self.words[prev_i:i]
            if new_words:
                new_seg = self.copy(new_words, copy_words=False)
                seg_copies.append(new_seg)
                if reassign_ids:
                    new_seg.reassign_ids()
            prev_i = i
        return seg_copies

    def set_result(self, result: 'WhisperResult'):
        warnings.warn('``.set_result(current_result_instance)`` is deprecated and will be removed in future versions. '
                      'Use ``.result = current_result_instance`` instead.',
                      stacklevel=2)
        self.result = result

    def get_result(self) -> Union['WhisperResult', None]:
        """
        Return outer instance of :class:`stable_whisper.result.WhisperResult` that ``self`` is a part of.
        """
        warnings.warn('``.get_result()`` will be removed in future versions. Use ``.result`` instead.',
                      stacklevel=2)
        return self.result

    def convert_to_segment_level(self):
        if not self.has_words:
            return
        self._default_text = self.text
        self._default_start = self.start
        self._default_end = self.end
        self._default_tokens = self.tokens
        self.words = None


class WhisperResult:

    def __init__(
            self,
            result: Union[str, dict, list],
            force_order: bool = False,
            check_sorted: Union[bool, str] = True,
            show_unsorted: bool = True
    ):
        result, self.path = self._standardize_result(result)
        self.ori_dict = result.get('ori_dict') or result
        self.language = self.ori_dict.get('language')
        self._regroup_history = result.get('regroup_history', '')
        self._nonspeech_sections = result.get('nonspeech_sections', [])
        segments = (result.get('segments', self.ori_dict.get('segments')) or {}).copy()
        self.segments = [Segment(**s, ignore_unused_args=True) for s in segments] if segments else []
        self._forced_order = force_order
        if self._forced_order:
            self.force_order()
        self.raise_for_unsorted(check_sorted, show_unsorted)
        self.remove_no_word_segments(any(seg.has_words for seg in self.segments))
        self._ignore_special_periods = False

    def __getitem__(self, index: int) -> Segment:
        return self.segments[index]

    def __delitem__(self, index: int):
        del self.segments[index]
        self.reassign_ids(True, start=index)

    @property
    def duration(self):
        if not self.segments:
            return 0.0
        return _round_timestamp(self.segments[-1].end - self.segments[0].start)

    @staticmethod
    def _standardize_result(result: Union[str, dict, List[dict], List[List[dict]]]) -> Tuple[dict, Union[str, None]]:
        path = None
        if isinstance(result, str):
            path = result
            result = load_result(path)
        if isinstance(result, dict):
            return result, path
        if not isinstance(result, list):
            raise TypeError(f'Expect result to be list but got {type(result)}')
        if not result or not result[0]:
            return {}, path
        if isinstance(result[0], list):
            if not isinstance(result[0][0], dict):
                raise NotImplementedError(f'Got list of list of {type(result[0])} but expects list of list of dict')
            result = dict(
                segments=[
                    dict(
                        start=words[0]['start'],
                        end=words[-1]['end'],
                        text=''.join(w['word'] for w in words),
                        words=words
                    )
                    for words in result if words
                ]
            )

        elif isinstance(result[0], dict):
            result = dict(segments=result)
        else:
            raise NotImplementedError(f'Got list of {type(result[0])} but expects list of list/dict')
        return result, path

    def force_order(self):
        prev_ts_end = 0
        timestamps = self.all_words_or_segments()
        for i, ts in enumerate(timestamps, 1):
            if ts.start < prev_ts_end:
                ts.start = prev_ts_end
            if ts.start > ts.end:
                if prev_ts_end > ts.end:
                    warnings.warn('Multiple consecutive timestamps are out of order. Some parts will have no duration.')
                    ts.start = ts.end
                    for j in range(i-2, -1, -1):
                        if timestamps[j].end > ts.end:
                            timestamps[j].end = ts.end
                        if timestamps[j].start > ts.end:
                            timestamps[j].start = ts.end
                else:
                    if ts.start != prev_ts_end:
                        ts.start = prev_ts_end
                    else:
                        ts.end = ts.start if i == len(timestamps) else timestamps[i].start
            prev_ts_end = ts.end

    def raise_for_unsorted(self, check_sorted: Union[bool, str] = True, show_unsorted: bool = True):
        if check_sorted is False:
            return
        all_parts = self.all_words_or_segments()
        if not all_parts:
            return
        is_word = isinstance(all_parts[0], WordTiming)
        timestamps = np.array(list(chain.from_iterable((p.start, p.end) for p in all_parts)))
        if len(timestamps) > 1 and (unsorted_mask := timestamps[:-1] > timestamps[1:]).any():
            if show_unsorted:
                def get_part_info(idx):
                    curr_part = all_parts[idx]
                    seg_id = curr_part.segment_id if is_word else curr_part.id
                    word_id_str = f'Word ID: {curr_part.id}\n' if is_word else ''
                    return (
                        f'Segment ID: {seg_id}\n{word_id_str}'
                        f'Start: {curr_part.start}\nEnd: {curr_part.end}\n'
                        f'Text: "{curr_part.word if is_word else curr_part.text}"'
                    ), curr_part.start, curr_part.end

                for i, unsorted in enumerate(unsorted_mask, 2):
                    if unsorted:
                        word_id = i//2-1
                        part_info, start, end = get_part_info(word_id)
                        if i % 2 == 1:
                            next_info, next_start, _ = get_part_info(word_id+1)
                            part_info += f'\nConflict: end ({end}) > next start ({next_start})\n{next_info}'
                        else:
                            part_info += f'\nConflict: start ({start}) > end ({end})'
                        print(part_info, end='\n\n')

            data = self.to_dict()
            if check_sorted is True:
                raise UnsortedException(data=data)
            warnings.warn('Timestamps are not in ascending order. '
                          'If data is produced by Stable-ts, please submit an issue with the saved data.')
            save_as_json(data, check_sorted)

    def update_all_segs_with_words(self):
        warnings.warn('Attributes that required updating are now properties based on the ``words`` except for ``id``. '
                      '``update_all_segs_with_words()`` is deprecated and will be removed in future versions. '
                      'Use ``.reassign_ids()`` to manually update ids',
                      stacklevel=2)
        self.reassign_ids()

    def update_nonspeech_sections(self, silent_starts, silent_ends):
        self._nonspeech_sections = [
            dict(start=round(s, 3), end=round(e, 3)) for s, e in zip(silent_starts, silent_ends)
        ]

    def add_segments(
            self,
            index0: int,
            index1: int,
            inplace: bool = False,
            lock: bool = False,
            newline: bool = False,
            reassign_ids: bool = True
    ):
        new_seg = self.segments[index0].add(
            self.segments[index1],
            copy_words=False,
            newline=newline,
            reassign_ids=reassign_ids
        )
        if lock and self.segments[index0].has_words:
            lock_idx = len(self.segments[index0].words)
            new_seg.words[lock_idx - 1].lock_right()
            if lock_idx < len(new_seg.words):
                new_seg.words[lock_idx].lock_left()
        if inplace:
            i0, i1 = sorted([index0, index1])
            self.segments[i0] = new_seg
            del self.segments[i1]
            if reassign_ids:
                self.reassign_ids(True)
        return new_seg

    def rescale_time(self, scale_factor: float):
        for s in self.segments:
            s.rescale_time(scale_factor)

    def apply_min_dur(self, min_dur: float, inplace: bool = False):
        """
        Merge any word/segment with adjacent word/segment if its duration is less than ``min_dur``.
        """
        result = self if inplace else deepcopy(self)
        max_i = len(result.segments) - 1
        if max_i == 0:
            return result
        for i in reversed(range(len(result.segments))):
            if max_i == 0:
                break
            if result.segments[i].duration < min_dur:
                if i == max_i:
                    result.add_segments(i-1, i, inplace=True, reassign_ids=False)
                elif i == 0:
                    result.add_segments(i, i+1, inplace=True, reassign_ids=False)
                else:
                    if result.segments[i+1].duration < result.segments[i-1].duration:
                        result.add_segments(i-1, i, inplace=True, reassign_ids=False)
                    else:
                        result.add_segments(i, i+1, inplace=True, reassign_ids=False)
                max_i -= 1
        result.reassign_ids()
        for s in result.segments:
            s.apply_min_dur(min_dur, inplace=True)
        return result

    def offset_time(self, offset_seconds: float):
        for s in self.segments:
            s.offset_time(offset_seconds)

    def suppress_silence(
            self,
            silent_starts: np.ndarray,
            silent_ends: np.ndarray,
            min_word_dur: Optional[float] = None,
            word_level: bool = True,
            nonspeech_error: float = 0.3,
            use_word_position: bool = True,
            verbose: bool = True,
    ) -> "WhisperResult":
        """
        Move any start/end timestamps in silence parts of audio to the boundaries of the silence.

        Parameters
        ----------
        silent_starts : numpy.ndarray
            An array starting timestamps of silent sections of audio.
        silent_ends : numpy.ndarray
            An array ending timestamps of silent sections of audio.
        min_word_dur : float or None, default None meaning use ``stable_whisper.default.DEFAULT_VALUES``
            Shortest duration each word is allowed to reach for adjustments.
        word_level : bool, default False
            Whether to settings to word level timestamps.
        nonspeech_error : float, default 0.3
            Relative error of non-speech sections that appear in between a word for adjustments.
        use_word_position : bool, default True
            Whether to use position of the word in its segment to determine whether to keep end or start timestamps if
            adjustments are required. If it is the first word, keep end. Else if it is the last word, keep the start.
        verbose : bool, default True
            Whether to use progressbar to show progress.

        Returns
        -------
        stable_whisper.result.WhisperResult
            The current instance after the changes.
        """
        min_word_dur = get_min_word_dur(min_word_dur)
        max_ts = self.segments[-1].end if self.segments else 0
        with tqdm(total=max_ts, unit='sec', disable=not verbose, desc='Adjustment') as tqdm_pbar:
            for s in self.segments:
                s.suppress_silence(
                    silent_starts,
                    silent_ends,
                    min_word_dur,
                    word_level=word_level,
                    nonspeech_error=nonspeech_error,
                    use_word_position=use_word_position
                )
                if verbose:
                    tqdm_pbar.update(s.end - tqdm_pbar.n)
            tqdm_pbar.update(tqdm_pbar.total - tqdm_pbar.n)

        return self

    def adjust_by_silence(
            self,
            audio: Union[torch.Tensor, np.ndarray, str, bytes],
            vad: Union[bool, dict] = False,
            *,
            verbose: (bool, None) = False,
            sample_rate: int = None,
            vad_onnx: bool = False,
            vad_threshold: float = 0.35,
            q_levels: int = 20,
            k_size: int = 5,
            min_word_dur: Optional[float] = None,
            min_silence_dur: Optional[float] = None,
            word_level: bool = True,
            nonspeech_error: float = 0.3,
            use_word_position: bool = True

    ) -> "WhisperResult":
        """
        Adjust timestamps base on detected speech gaps.

        This is method combines :meth:`stable_whisper.result.WhisperResult.suppress_silence` with silence detection.

        Parameters
        ----------
        audio : str or numpy.ndarray or torch.Tensor or bytes
            Path/URL to the audio file, the audio waveform, or bytes of audio file.
        vad : bool or dict, default False
            Whether to use Silero VAD to generate timestamp suppression mask.
            Instead of ``True``, using a dict of keyword arguments will load the VAD with the arguments.
            Silero VAD requires PyTorch 1.12.0+. Official repo, https://github.com/snakers4/silero-vad.
        verbose : bool or None, default False
            Displays all info if ``True``. Displays progressbar if ``False``. Display nothing if ``None``.
            If ``vad = True`` and ``False``, mute messages about hitting local caches.
            Note that the message about first download cannot be muted.
        sample_rate : int, default None, meaning ``whisper.audio.SAMPLE_RATE``, 16kHZ
            The sample rate of ``audio``.
        vad_threshold : float, default 0.35
            Threshold for detecting speech with Silero VAD. Low threshold reduces false positives for silence detection.
        q_levels : int, default 20
            Quantization levels for generating timestamp suppression mask; ignored if ``vad = true``.
            Acts as a threshold to marking sound as silent.
            Fewer levels will increase the threshold of volume at which to mark a sound as silent.
        k_size : int, default 5
            Kernel size for avg-pooling waveform to generate timestamp suppression mask; ignored if ``vad = true``.
            Recommend 5 or 3; higher sizes will reduce detection of silence.
        min_word_dur : float or None, default None meaning use ``stable_whisper.default.DEFAULT_VALUES``
            Shortest duration each word is allowed to reach from adjustments.
        min_silence_dur : float, optional
            Shortest duration of silence allowed for silence suppression.
        word_level : bool, default True
            Whether to settings to word level timestamps.
        nonspeech_error : float, default 0.3
            Relative error of non-speech sections that appear in between a word for adjustments.
        use_word_position : bool, default True
            Whether to use position of the word in its segment to determine whether to keep end or start timestamps if
            adjustments are required. If it is the first word, keep end. Else if it is the last word, keep the start.

        Returns
        -------
        stable_whisper.result.WhisperResult
            The current instance after the changes.

        Notes
        -----
        This operation is already performed by :func:`stable_whisper.whisper_word_level.transcribe_stable` /
        :func:`stable_whisper.whisper_word_level.transcribe_minimal`/
        :func:`stable_whisper.non_whisper.transcribe_any` / :func:`stable_whisper.alignment.align`
        if ``suppress_silence = True``.
        """
        min_word_dur = get_min_word_dur(min_word_dur)
        if vad is not False:
            audio = audio_to_tensor_resample(audio, sample_rate, VAD_SAMPLE_RATES[0])
            sample_rate = VAD_SAMPLE_RATES[0]
            silent_timings = get_vad_silence_func(
                **(vad if isinstance(vad, dict) else {}),
                vad_onnx=vad_onnx,
                verbose=verbose
            )(audio, speech_threshold=vad_threshold, sr=sample_rate)
        else:
            silent_timings = audio2timings(audio, q_levels=q_levels, k_size=k_size, sr=sample_rate)
        if silent_timings is None:
            return self
        if min_silence_dur:
            silent_timings = filter_timings(silent_timings, min_silence_dur)
        self.suppress_silence(
            *silent_timings,
            min_word_dur=min_word_dur,
            word_level=word_level,
            nonspeech_error=nonspeech_error,
            use_word_position=use_word_position,
            verbose=verbose is not None
        )
        self.update_nonspeech_sections(*silent_timings)
        return self

    def adjust_by_result(
            self,
            other_result: "WhisperResult",
            min_word_dur: Optional[float] = None,
            verbose: bool = False
    ):
        """
        Minimize the duration of words using timestamps of another result.

        Parameters
        ----------
        other_result : "WhisperResult"
            Timing data of the same words in a WhisperResult instance.
        min_word_dur : float or None, default None meaning use ``stable_whisper.default.DEFAULT_VALUES``
            Prevent changes to timestamps if the resultant word duration is less than ``min_word_dur``.
        verbose : bool, default False
            Whether to print out the timestamp changes.
        """
        if not (self.has_words and other_result.has_words):
            raise NotImplementedError('This operation can only be performed on results with word timestamps')
        assert [w.word for w in self.all_words()] == [w.word for w in other_result.all_words()], \
            'The words in [other_result] do not match the current words.'
        min_word_dur = get_min_word_dur(min_word_dur)
        for word, other_word in zip(self.all_words(), other_result.all_words()):
            if word.end > other_word.start:
                new_start = max(word.start, other_word.start)
                new_end = min(word.end, other_word.end)
                if new_end - new_start >= min_word_dur:
                    line = ''
                    if word.start != new_start:
                        if verbose:
                            line += f'[Start:{word.start:.3f}->{new_start:.3f}] '
                        word.start = new_start
                    if word.end != new_end:
                        if verbose:
                            line += f'[End:{word.end:.3f}->{new_end:.3f}]  '
                        word.end = new_end
                    if line:
                        print(f'{line}"{word.word}"')

    def reassign_ids(self, only_segments: bool = False, start: Optional[int] = None):
        for i, s in enumerate(self.segments[start:], start or 0):
            s.id = i
            s.result = self
            if not only_segments:
                s.reassign_ids()

    def remove_no_word_segments(self, ignore_ori=False, reassign_ids: bool = True):
        for i in reversed(range(len(self.segments))):
            if (ignore_ori or self.segments[i].ori_has_words) and not self.segments[i].has_words:
                del self.segments[i]
        if reassign_ids:
            self.reassign_ids()

    def get_locked_indices(self):
        locked_indices = [i
                          for i, (left, right) in enumerate(zip(self.segments[1:], self.segments[:-1]))
                          if left.left_locked or right.right_locked]
        return locked_indices

    def get_gaps(self, as_ndarray=False):
        s_ts = np.array([s.start for s in self.segments])
        e_ts = np.array([s.end for s in self.segments])
        gap = s_ts[1:] - e_ts[:-1]
        return gap if as_ndarray else gap.tolist()

    def get_gap_indices(self, min_gap: float = 0.1):  # for merging
        if len(self.segments) < 2:
            return []
        if min_gap is None:
            min_gap = 0
        indices = (self.get_gaps(True) <= min_gap).nonzero()[0].tolist()
        return sorted(set(indices) - set(self.get_locked_indices()))

    def get_punctuation_indices(self, punctuation: Union[List[str], List[Tuple[str, str]], str]):  # for merging
        if len(self.segments) < 2:
            return []
        if isinstance(punctuation, str):
            punctuation = [punctuation]
        indices = []
        for p in punctuation:
            if isinstance(p, str):
                for i, s in enumerate(self.segments[:-1]):
                    if s.text.endswith(p):
                        indices.append(i)
                    elif i != 0 and s.text.startswith(p):
                        indices.append(i-1)
            else:
                ending, beginning = p
                indices.extend([i for i, (s0, s1) in enumerate(zip(self.segments[:-1], self.segments[1:]))
                                if s0.text.endswith(ending) and s1.text.startswith(beginning)])

        return sorted(set(indices) - set(self.get_locked_indices()))

    def all_words(self):
        return list(chain.from_iterable(s.words for s in self.segments))

    def all_words_or_segments(self):
        return self.all_words() if self.has_words else self.segments

    def all_words_by_lock(self, only_text: bool = True, by_segment: bool = False, include_single: bool = False):
        if by_segment:
            return [
                segment.words_by_lock(only_text=only_text, include_single=include_single)
                for segment in self.segments
            ]
        return _words_by_lock(self.all_words(), only_text=only_text, include_single=include_single)

    def all_tokens(self):
        return list(chain.from_iterable(s.tokens for s in self.all_words()))

    def to_dict(self, keep_orig: bool = True):
        ori_dict = self.ori_dict if keep_orig else {}
        return dict(text=self.text,
                    segments=self.segments_to_dicts(),
                    language=self.language,
                    ori_dict=ori_dict,
                    regroup_history=self._regroup_history,
                    nonspeech_sections=self._nonspeech_sections)

    def segments_to_dicts(self, reverse_text: Union[bool, tuple] = False):
        return [s.to_dict(reverse_text=reverse_text) for s in self.segments]

    def split_segment_by_index(
            self,
            segment: Union[int, Segment],
            indices: Union[int, List[int]],
            reassign_ids: bool = True
    ):
        if not self.has_words:
            return
        if isinstance(indices, int):
            indices = [indices]
        elif not indices:
            return
        oor_indices = [i for i in indices if i < 0 or i > len(segment.words)]
        if oor_indices:
            raise IndexError(f'got out of split range indices: {oor_indices}')
        seg_idx = segment if isinstance(segment, int) else segment.id
        new_segments = self.segments[seg_idx].split(indices, reassign_ids=reassign_ids)
        del self.segments[seg_idx]
        for seg in reversed(new_segments):
            self.segments.insert(seg_idx, seg)
        if reassign_ids:
            self.reassign_ids(True)

    def _remove_special_period_indices(self, indices: List[int], segment_idx: int):
        segment = self.segments[segment_idx]
        for i in range(len(indices)-1, -1, -1):
            word_idx = indices[i]
            if not segment[word_idx].word.endswith('.'):
                continue
            word = segment[word_idx].word.strip()
            if re.search('^[A-Z0-9]', word) is None:
                continue
            if len(re.sub('[.A-Z0-9]', '', word)) < 3:
                indices.pop(i)

    def _split_segments(
            self,
            get_indices,
            args: list = None,
            *,
            lock: bool = False,
            newline: bool = False,
            ignore_special_periods: bool = False
    ):
        if args is None:
            args = []
        no_words = False
        for i in reversed(range(0, len(self.segments))):
            no_words = no_words or not self.segments[i].has_words
            indices = sorted(set(get_indices(self.segments[i], *args)))
            if ignore_special_periods:
                self._remove_special_period_indices(indices, i)
            if not indices:
                continue
            if newline:
                if indices[-1] == len(self.segments[i].words) - 1:
                    del indices[-1]
                    if not indices:
                        continue

                for word_idx in indices:
                    if self.segments[i].words[word_idx].word.endswith('\n'):
                        continue
                    self.segments[i].words[word_idx].word += '\n'
                    if lock:
                        self.segments[i].words[word_idx].lock_right()
                        if word_idx + 1 < len(self.segments[i].words):
                            self.segments[i].words[word_idx+1].lock_left()
            else:
                new_segments = self.segments[i].split(indices, reassign_ids=False)
                if lock:
                    for s in new_segments:
                        if s == new_segments[0]:
                            s.lock_right()
                        elif s == new_segments[-1]:
                            s.lock_left()
                        else:
                            s.lock_both()
                del self.segments[i]
                for s in reversed(new_segments):
                    self.segments.insert(i, s)
        if no_words:
            warnings.warn('Found segment(s) without word timings. These segment(s) cannot be split.')
        self.remove_no_word_segments()

    def _merge_segments(
            self,
            indices: List[int],
            *,
            max_words: int = None,
            max_chars: int = None,
            is_sum_max: bool = False,
            lock: bool = False,
            newline: bool = False
    ):
        if len(indices) == 0:
            return
        for i in reversed(indices):
            seg = self.segments[i]
            if (
                    (
                            max_words and
                            seg.has_words and
                            (
                                    (seg.word_count() + self.segments[i + 1].word_count() > max_words)
                                    if is_sum_max else
                                    (seg.word_count() > max_words and self.segments[i + 1].word_count() > max_words)
                            )
                    ) or
                    (
                            max_chars and
                            (
                                    (seg.char_count() + self.segments[i + 1].char_count() > max_chars)
                                    if is_sum_max else
                                    (seg.char_count() > max_chars and self.segments[i + 1].char_count() > max_chars)
                            )
                    )
            ):
                continue
            self.add_segments(i, i + 1, inplace=True, lock=lock, newline=newline, reassign_ids=False)
        self.remove_no_word_segments()

    def _update_history(self, changes: str):
        if not changes:
            return
        if self._regroup_history:
            self._regroup_history += '_'
        self._regroup_history += changes

    def get_content_by_time(
            self,
            time: Union[float, Tuple[float, float], dict],
            within: bool = False,
            segment_level: bool = False
    ) -> Union[List[WordTiming], List[Segment]]:
        """
        Return content in the ``time`` range.

        Parameters
        ----------
        time : float or tuple of (float, float) or dict
            Range of time to find content. For tuple of two floats, first value is the start time and second value is
            the end time. For a single float value, it is treated as both the start and end time.
        within : bool, default False
            Whether to only find content fully overlaps with ``time`` range.
        segment_level : bool, default False
            Whether to look only on the segment level and return instances of :class:`stable_whisper.result.Segment`
            instead of :class:`stable_whisper.result.WordTiming`.

        Returns
        -------
        list of stable_whisper.result.WordTiming or list of stable_whisper.result.Segment
            List of contents in the ``time`` range. The contents are instances of
            :class:`stable_whisper.result.Segment` if ``segment_level = True`` else
            :class:`stable_whisper.result.WordTiming`.
        """
        if not segment_level and not self.has_words:
            raise ValueError('Missing word timestamps in result. Use ``segment_level=True`` instead.')
        contents = self.segments if segment_level else self.all_words()
        if isinstance(time, (float, int)):
            time = [time, time]
        elif isinstance(time, dict):
            time = [time['start'], time['end']]
        start, end = time

        if within:
            def is_in_range(c):
                return start <= c.start and end >= c.end
        else:
            def is_in_range(c):
                return start <= c.end and end >= c.start

        return [c for c in contents if is_in_range(c)]

    def ignore_special_periods(
            self,
            enable: bool = True
    ) -> "WhisperResult":
        """
        Set ``enable`` as default for ``ignore_special_periods`` in all methods with ``ignore_special_periods``.

        Parameters
        ----------
        enable : bool, default True
            Value to set ``ignore_special_periods`` for methods that has this option.

        Returns
        -------
        stable_whisper.result.WhisperResult
            The current instance after the changes.
        """
        self._ignore_special_periods = enable
        self._update_history(f'isp={int(enable)}')
        return self

    def split_by_gap(
            self,
            max_gap: float = 0.1,
            lock: bool = False,
            newline: bool = False,
            ignore_special_periods: bool = False
    ) -> "WhisperResult":
        """
        Split (in-place) any segment where the gap between two of its words is greater than ``max_gap``.

        Parameters
        ----------
        max_gap : float, default 0.1
            Maximum second(s) allowed between two words if the same segment.
        lock : bool, default False
            Whether to prevent future splits/merges from altering changes made by this method.
        newline: bool, default False
            Whether to insert line break at the split points instead of splitting into separate segments.
        ignore_special_periods : bool, default False
            Whether to avoid splitting at periods that is likely not an indication of the end of a sentence.

        Returns
        -------
        stable_whisper.result.WhisperResult
            The current instance after the changes.
        """
        ignore_special_periods = self._ignore_special_periods or ignore_special_periods
        self._split_segments(
            lambda x: x.get_gap_indices(max_gap),
            lock=lock,
            newline=newline,
            ignore_special_periods=ignore_special_periods
        )
        self._update_history(f'sg={max_gap}+{int(lock)}+{int(newline)}+{int(ignore_special_periods)}')
        return self

    def merge_by_gap(
            self,
            min_gap: float = 0.1,
            max_words: int = None,
            max_chars: int = None,
            is_sum_max: bool = False,
            lock: bool = False,
            newline: bool = False
    ) -> "WhisperResult":
        """
        Merge (in-place) any pair of adjacent segments if the gap between them <= ``min_gap``.

        Parameters
        ----------
        min_gap : float, default 0.1
            Minimum second(s) allow between two segment.
        max_words : int, optional
            Maximum number of words allowed in each segment.
        max_chars : int, optional
            Maximum number of characters allowed in each segment.
        is_sum_max : bool, default False
            Whether ``max_words`` and ``max_chars`` is applied to the merged segment instead of the individual segments
            to be merged.
        lock : bool, default False
            Whether to prevent future splits/merges from altering changes made by this method.
        newline : bool, default False
            Whether to insert a line break between the merged segments.

        Returns
        -------
        stable_whisper.result.WhisperResult
            The current instance after the changes.
        """
        indices = self.get_gap_indices(min_gap)
        self._merge_segments(
            indices,
            max_words=max_words,
            max_chars=max_chars,
            is_sum_max=is_sum_max,
            lock=lock,
            newline=newline
        )
        self._update_history(
            f'mg={min_gap}+{max_words or ""}+{max_chars or ""}+{int(is_sum_max)}+{int(lock)}+{int(newline)}'
        )
        return self

    def split_by_punctuation(
            self,
            punctuation: Union[List[str], List[Tuple[str, str]], str],
            lock: bool = False,
            newline: bool = False,
            min_words: Optional[int] = None,
            min_chars: Optional[int] = None,
            min_dur: Optional[int] = None,
            ignore_special_periods: bool = False
    ) -> "WhisperResult":
        """
        Split (in-place) segments at words that start/end with ``punctuation``.

        Parameters
        ----------
        punctuation : list of str or list of tuple of (str, str) or str
            Punctuation(s) to split segments by.
        lock : bool, default False
            Whether to prevent future splits/merges from altering changes made by this method.
        newline : bool, default False
            Whether to insert line break at the split points instead of splitting into separate segments.
        min_words : int, optional
            Split segments with words >= ``min_words``.
        min_chars : int, optional
            Split segments with characters >= ``min_chars``.
        min_dur : int, optional
            Split segments with duration (in seconds) >= ``min_dur``.
        ignore_special_periods : bool, default False
            Whether to avoid splitting at periods that is likely not an indication of the end of a sentence.

        Returns
        -------
        stable_whisper.result.WhisperResult
            The current instance after the changes.
        """
        def _over_max(x: Segment):
            return (
                    (min_words and len(x.words) >= min_words) or
                    (min_chars and x.char_count() >= min_chars) or
                    (min_dur and x.duration >= min_dur)
            )

        indices = set(s.id for s in self.segments if _over_max(s)) if any((min_words, min_chars, min_dur)) else None

        def _get_indices(x: Segment):
            return x.get_punctuation_indices(punctuation) if indices is None or x.id in indices else []

        ignore_special_periods = self._ignore_special_periods or ignore_special_periods
        self._split_segments(
            _get_indices,
            lock=lock,
            newline=newline,
            ignore_special_periods=ignore_special_periods
        )
        punct_str = '/'.join(p if isinstance(p, str) else '*'.join(p) for p in punctuation)
        self._update_history(
            f'sp={punct_str}+{int(lock)}+{int(newline)}+{min_words or ""}'
            f'+{min_chars or ""}+{min_dur or ""}+{int(ignore_special_periods)}'
        )
        return self

    def merge_by_punctuation(
            self,
            punctuation: Union[List[str], List[Tuple[str, str]], str],
            max_words: int = None,
            max_chars: int = None,
            is_sum_max: bool = False,
            lock: bool = False,
            newline: bool = False
    ) -> "WhisperResult":
        """
        Merge (in-place) any two segments that has specific punctuations inbetween.

        Parameters
        ----------
        punctuation : list of str or list of tuple of (str, str) or str
            Punctuation(s) to merge segments by.
        max_words : int, optional
            Maximum number of words allowed in each segment.
        max_chars : int, optional
            Maximum number of characters allowed in each segment.
        is_sum_max : bool, default False
            Whether ``max_words`` and ``max_chars`` is applied to the merged segment instead of the individual segments
            to be merged.
        lock : bool, default False
            Whether to prevent future splits/merges from altering changes made by this method.
        newline : bool, default False
            Whether to insert a line break between the merged segments.

        Returns
        -------
        stable_whisper.result.WhisperResult
            The current instance after the changes.
        """
        indices = self.get_punctuation_indices(punctuation)
        self._merge_segments(
            indices,
            max_words=max_words,
            max_chars=max_chars,
            is_sum_max=is_sum_max,
            lock=lock,
            newline=newline
        )
        punct_str = '/'.join(p if isinstance(p, str) else '*'.join(p) for p in punctuation)
        self._update_history(
            f'mp={punct_str}+{max_words or ""}+{max_chars or ""}+{int(is_sum_max)}+{int(lock)}+{int(newline)}'
        )
        return self

    def pad(
            self,
            start_pad: Optional[float] = None,
            end_pad: Optional[float] = None,
            max_dur: Optional[float] = None,
            max_end: Optional[float] = None,
            word_level: bool = False
    ) -> "WhisperResult":
        """
        Pad (in-place) timestamps in chronological order.

        Parameters
        ----------
        start_pad : float, optional
            Seconds to pad start timestamps.
            Each start timestamp will be extended no earlier than the end timestamp of the previous word.
        end_pad : float, optional
            Seconds to pad end timestamps.
            Each end timestamp will be extended no later than the start timestamp of the next word or ``max_end``.
        max_dur : float, optional
            Only pad segments or words (``word_level=True``) with duration (in seconds) under or equal to ``max_dur``.
        max_end : float, optional
            Timestamp (in seconds) that padded timestamps cannot exceed.
            Generally used to prevent the last padded end timestamp from exceeding the total duration of the audio.
        word_level : bool, default False
            Whether to pad segment timestamps or word timestamps.

        Returns
        -------
        stable_whisper.result.WhisperResult
            The current instance after the changes.
        """
        if not (start_pad or end_pad):
            warnings.warn(f'No ``start_pad`` or ``end_pad`` given.',
                          stacklevel=2)
            return self
        if word_level and not self.has_words:
            word_level = False
        parts = self.all_words() if word_level else self.segments
        assert not start_pad or start_pad > 0, '``start_pad`` must be positive'
        assert not end_pad or end_pad > 0, '``end_pad`` must be positive'
        assert max_dur is None or max_dur > 0, '``max_dur`` must be greater than 0'
        assert max_end is None or max_end > 0, '``max_end`` must be greater than 0'
        for i, part in enumerate(parts, 1):
            if max_dur and part.end - part.start > max_dur:
                continue
            if start_pad:
                new_start = part.start - start_pad
                new_start = max(0 if i == 1 else parts[i-2].end, new_start)
                part.start = new_start
            if end_pad:
                new_end = part.end + end_pad
                temp_max_end = max_end
                if i != len(parts):
                    temp_max_end = min(max_end, parts[i].start) if max_end else parts[i].start
                if temp_max_end and temp_max_end < new_end:
                    new_end = temp_max_end
                if new_end > part.end:
                    part.end = new_end

        self._update_history(
            f'p={start_pad or ""}+{end_pad or ""}+{max_dur or ""}+{max_end or ""}+{int(word_level)}'
        )
        return self

    def merge_all_segments(
            self,
            record: bool = True
    ) -> "WhisperResult":
        """
        Merge all segments into one segment.

        Parameters
        ----------
        record : bool , default True
            Whether to record this operation in ``regroup_history``.

        Returns
        -------
        stable_whisper.result.WhisperResult
            The current instance after the changes.
        """
        if not self.segments:
            return self
        if self.has_words:
            new_seg = self.segments[0].copy(self.all_words(), keep_result=True, copy_words=False)
        else:
            new_seg = self.segments[0]
            new_seg._default_text = ''.join(s.text for s in self.segments)
            if all(s.tokens is not None for s in self.segments):
                new_seg._default_tokens = list(chain.from_iterable(s.tokens for s in self.segments))
            new_seg.end = self.segments[-1].end
        self.segments = [new_seg]
        self.reassign_ids()
        if record:
            self._update_history('ms')
        return self

    def split_by_length(
            self,
            max_chars: int = None,
            max_words: int = None,
            even_split: bool = True,
            force_len: bool = False,
            lock: bool = False,
            include_lock: bool = False,
            newline: bool = False,
            ignore_special_periods: bool = False
    ) -> "WhisperResult":
        """
        Split (in-place) any segment that exceeds ``max_chars`` or ``max_words`` into smaller segments.

        Parameters
        ----------
        max_chars : int, optional
            Maximum number of characters allowed in each segment.
        max_words : int, optional
            Maximum number of words allowed in each segment.
        even_split : bool, default True
            Whether to evenly split a segment in length if it exceeds ``max_chars`` or ``max_words``.
        force_len : bool, default False
            Whether to force a constant length for each segment except the last segment.
            This will ignore all previous non-locked segment boundaries.
        lock : bool, default False
            Whether to prevent future splits/merges from altering changes made by this method.
        include_lock: bool, default False
            Whether to include previous lock before splitting based on max_words, if ``even_split = False``.
            Splitting will be done after the first non-locked word > ``max_chars`` / ``max_words``.
        newline: bool, default False
            Whether to insert line break at the split points instead of splitting into separate segments.
        ignore_special_periods : bool, default False
            Whether to avoid splitting at periods that is likely not an indication of the end of a sentence.

        Returns
        -------
        stable_whisper.result.WhisperResult
            The current instance after the changes.

        Notes
        -----
        If ``even_split = True``, segments can still exceed ``max_chars``.
        uneven splitting.
        """
        if force_len:
            self.merge_all_segments(record=False)
        ignore_special_periods = self._ignore_special_periods or ignore_special_periods
        self._split_segments(
            lambda x: x.get_length_indices(
                max_chars=max_chars,
                max_words=max_words,
                even_split=even_split,
                include_lock=include_lock
            ),
            lock=lock,
            newline=newline,
            ignore_special_periods=ignore_special_periods
        )
        self._update_history(
            f'sl={max_chars or ""}+{max_words or ""}+{int(even_split)}+{int(force_len)}'
            f'+{int(lock)}+{int(include_lock)}+{int(newline)}+{int(ignore_special_periods)}'
        )
        return self

    def split_by_duration(
            self,
            max_dur: float,
            even_split: bool = True,
            force_len: bool = False,
            lock: bool = False,
            include_lock: bool = False,
            newline: bool = False,
            ignore_special_periods: bool = False
    ) -> "WhisperResult":
        """
        Split (in-place) any segment that exceeds ``max_dur`` into smaller segments.

        Parameters
        ----------
        max_dur : float
            Maximum duration (in seconds) per segment.
        even_split : bool, default True
            Whether to evenly split a segment in length if it exceeds ``max_dur``.
        force_len : bool, default False
            Whether to force a constant length for each segment except the last segment.
            This will ignore all previous non-locked segment boundaries.
        lock : bool, default False
            Whether to prevent future splits/merges from altering changes made by this method.
        include_lock: bool, default False
            Whether to include previous lock before splitting based on max_words, if ``even_split = False``.
            Splitting will be done after the first non-locked word > ``max_dur``.
        newline: bool, default False
            Whether to insert line break at the split points instead of splitting into separate segments.
        ignore_special_periods : bool, default False
            Whether to avoid splitting at periods that is likely not an indication of the end of a sentence.

        Returns
        -------
        stable_whisper.result.WhisperResult
            The current instance after the changes.

        Notes
        -----
        If ``even_split = True``, segments can still exceed ``max_dur``.`
        uneven splitting.
        """
        if force_len:
            self.merge_all_segments(record=False)
        ignore_special_periods = self._ignore_special_periods or ignore_special_periods
        self._split_segments(
            lambda x: x.get_duration_indices(
                max_dur=max_dur,
                even_split=even_split,
                include_lock=include_lock
            ),
            lock=lock,
            newline=newline,
            ignore_special_periods=ignore_special_periods
        )
        self._update_history(
            f'sd={max_dur}+{int(even_split)}+{int(force_len)}'
            f'+{int(lock)}+{int(include_lock)}+{int(newline)}+{int(ignore_special_periods)}'
        )
        return self

    def clamp_max(
            self,
            medium_factor: float = 2.5,
            max_dur: float = None,
            clip_start: Optional[bool] = None,
            verbose: bool = False
    ) -> "WhisperResult":
        """
        Clamp all word durations above certain value.

        This is most effective when applied before and after other regroup operations.

        Parameters
        ----------
        medium_factor : float, default 2.5
            Clamp durations above (``medium_factor`` * medium duration) per segment.
            If ``medium_factor = None/0`` or segment has less than 2 words, it will be ignored and use only ``max_dur``.
        max_dur : float, optional
            Clamp durations above ``max_dur``.
        clip_start : bool or None, default None
            Whether to clamp the start of a word. If ``None``, clamp the start of first word and end of last word per
            segment.
        verbose : bool, default False
            Whether to print out the timestamp changes.

        Returns
        -------
        stable_whisper.result.WhisperResult
            The current instance after the changes.
        """
        if not (medium_factor or max_dur):
            raise ValueError('At least one of following arguments requires non-zero value: medium_factor; max_dur')

        if not self.has_words:
            warnings.warn('Cannot clamp due to missing/no word-timestamps')
            return self

        for seg in self.segments:
            curr_max_dur = None
            if medium_factor and len(seg.words) > 1:
                durations = np.array([word.duration for word in seg.words])
                durations.sort()
                curr_max_dur = medium_factor * durations[len(durations)//2]

            if max_dur and (not curr_max_dur or curr_max_dur > max_dur):
                curr_max_dur = max_dur

            if not curr_max_dur:
                continue

            if clip_start is None:
                seg.words[0].clamp_max(curr_max_dur, clip_start=True, verbose=verbose)
                seg.words[-1].clamp_max(curr_max_dur, clip_start=False, verbose=verbose)
            else:
                for i, word in enumerate(seg.words):
                    word.clamp_max(curr_max_dur, clip_start=clip_start, verbose=verbose)

        self._update_history(f'cm={medium_factor}+{max_dur or ""}+{clip_start or ""}+{int(verbose)}')
        return self

    def lock(
            self,
            startswith: Union[str, List[str]] = None,
            endswith: Union[str, List[str]] = None,
            right: bool = True,
            left: bool = False,
            case_sensitive: bool = False,
            strip: bool = True
    ) -> "WhisperResult":
        """
        Lock words/segments with matching prefix/suffix to prevent splitting/merging.

        Parameters
        ----------
        startswith: str or list of str
            Prefixes to lock.
        endswith: str or list of str
            Suffixes to lock.
        right : bool, default True
            Whether prevent splits/merges with the next word/segment.
        left : bool, default False
            Whether prevent splits/merges with the previous word/segment.
        case_sensitive : bool, default False
            Whether to match the case of the prefixes/suffixes with the words/segments.
        strip : bool, default True
            Whether to ignore spaces before and after both words/segments and prefixes/suffixes.

        Returns
        -------
        stable_whisper.result.WhisperResult
            The current instance after the changes.
        """
        assert startswith is not None or endswith is not None, 'Must specify [startswith] or/and [endswith].'
        startswith = [] if startswith is None else ([startswith] if isinstance(startswith, str) else startswith)
        endswith = [] if endswith is None else ([endswith] if isinstance(endswith, str) else endswith)
        if not case_sensitive:
            startswith = [t.lower() for t in startswith]
            endswith = [t.lower() for t in endswith]
        if strip:
            startswith = [t.strip() for t in startswith]
            endswith = [t.strip() for t in endswith]
        for part in self.all_words_or_segments():
            text = part.word if hasattr(part, 'word') else part.text
            if not case_sensitive:
                text = text.lower()
            if strip:
                text = text.strip()
            for prefix in startswith:
                if text.startswith(prefix):
                    if right:
                        part.lock_right()
                    if left:
                        part.lock_left()
            for suffix in endswith:
                if text.endswith(suffix):
                    if right:
                        part.lock_right()
                    if left:
                        part.lock_left()
        startswith_str = (startswith if isinstance(startswith, str) else '/'.join(startswith)) if startswith else ""
        endswith_str = (endswith if isinstance(endswith, str) else '/'.join(endswith)) if endswith else ""
        self._update_history(
            f'l={startswith_str}+{endswith_str}'
            f'+{int(right)}+{int(left)}+{int(case_sensitive)}+{int(strip)}'
        )
        return self

    def remove_word(
            self,
            word: Union[WordTiming, Tuple[int, int]],
            reassign_ids: bool = True,
            verbose: bool = True,
            record: bool = True
    ) -> 'WhisperResult':
        """
        Remove a word.

        Parameters
        ----------
        word : WordTiming or tuple of (int, int)
            Instance of :class:`stable_whisper.result.WordTiming` or tuple of (segment index, word index).
        reassign_ids : bool, default True
            Whether to reassign segment and word ids (indices) after removing ``word``.
        verbose : bool, default True
            Whether to print detail of the removed word.
        record : bool , default True
            Whether to record this operation in ``regroup_history``.

        Returns
        -------
        stable_whisper.result.WhisperResult
            The current instance after the changes.
        """
        if isinstance(word, WordTiming):
            if self[word.segment_id][word.id] is not word:
                self.reassign_ids()
                if self[word.segment_id][word.id] is not word:
                    raise ValueError('word not in result')
            seg_id, word_id = word.segment_id, word.id
        else:
            seg_id, word_id = map(int, word.split(',')) if isinstance(word, str) else word
        if verbose:
            print(f'Removed: {self[seg_id][word_id].to_dict()}')
        del self.segments[seg_id].words[word_id]
        if not reassign_ids:
            return self
        if self[seg_id].has_words:
            self[seg_id].reassign_ids()
        else:
            self.remove_no_word_segments()
        if record:
            self._update_history(f'rw={seg_id},{word_id}+{int(reassign_ids)}+{int(verbose)}')
        return self

    def remove_segment(
            self,
            segment: Union[Segment, int],
            reassign_ids: bool = True,
            verbose: bool = True,
            record: bool = True
    ) -> 'WhisperResult':
        """
        Remove a segment.

        Parameters
        ----------
        segment : Segment or int
            Instance :class:`stable_whisper.result.Segment` or segment index.
        reassign_ids : bool, default True
            Whether to reassign segment IDs (indices) after removing ``segment``.
        verbose : bool, default True
            Whether to print detail of the removed word.
        record : bool , default True
            Whether to record this operation in ``regroup_history``.

        Returns
        -------
        stable_whisper.result.WhisperResult
            The current instance after the changes.
        """
        if isinstance(segment, Segment):
            if self[segment.id] is not segment:
                self.reassign_ids()
                if self[segment.id] is not segment:
                    raise ValueError('segment not in result')
            segment = segment.id
        if verbose:
            print(f'Removed: [id:{self[segment].id}] {self[segment].to_display_str(True)}')
        del self.segments[segment]
        if not reassign_ids:
            return self
        self.reassign_ids(True, start=segment)
        if record:
            self._update_history(f'rs={segment}+{int(reassign_ids)}+{int(verbose)}')
        return self

    def remove_repetition(
            self,
            max_words: int = 1,
            case_sensitive: bool = False,
            strip: bool = True,
            ignore_punctuations: str = "\"',.?!",
            extend_duration: bool = True,
            verbose: bool = True
    ) -> 'WhisperResult':
        """
        Remove words that repeat consecutively.

        Parameters
        ----------
        max_words : int
            Maximum number of words to look for consecutively.
        case_sensitive : bool, default False
            Whether the case of words need to match to be considered as repetition.
        strip : bool, default True
            Whether to ignore spaces before and after each word.
        ignore_punctuations : bool, default '"',.?!'
            Ending punctuations to ignore.
        extend_duration: bool, default True
            Whether to extend the duration of the previous word to cover the duration of the repetition.
        verbose: bool, default True
            Whether to print detail of the removed repetitions.

        Returns
        -------
        stable_whisper.result.WhisperResult
            The current instance after the changes.
        """
        if not self.has_words:
            return self

        for count in range(1, max_words + 1):
            all_words = self.all_words()
            if len(all_words) < 2:
                return self
            all_words_str = [w.word for w in all_words]
            if strip:
                all_words_str = [w.strip() for w in all_words_str]
            if ignore_punctuations:
                ptn = f'[{ignore_punctuations}]+$'
                all_words_str = [re.sub(ptn, '', w) for w in all_words_str]
            if not case_sensitive:
                all_words_str = [w.lower() for w in all_words_str]
            next_i = None
            changes = []
            for i in reversed(range(count*2, len(all_words_str)+1)):
                if next_i is not None:
                    if next_i != i:
                        continue
                    else:
                        next_i = None
                s = i - count
                if all_words_str[s - count:s] != all_words_str[s:i]:
                    continue
                next_i = s
                if extend_duration:
                    all_words[s-1].end = all_words[i-1].end
                temp_changes = []
                for j in reversed(range(s, i)):
                    if verbose:
                        temp_changes.append(f'- {all_words[j].to_dict()}')
                    self.remove_word(all_words[j], False, verbose=False, record=False)
                if temp_changes:
                    changes.append(
                        f'Remove: [{format_timestamp(all_words[s].start)} -> {format_timestamp(all_words[i-1].end)}] '
                        + ''.join(_w.word for _w in all_words[s:i]) + '\n'
                        + '\n'.join(reversed(temp_changes)) + '\n'
                    )
                for i0, i1 in zip(range(s - count, s), range(s, i)):
                    if len(all_words[i0].word) < len(all_words[i1].word):
                        all_words[i1].start = all_words[i0].start
                        all_words[i1].end = all_words[i0].end
                        _sid, _wid = all_words[i0].segment_id, all_words[i0].id
                        self.segments[_sid].words[_wid] = all_words[i1]

            if changes:
                print('\n'.join(reversed(changes)))

            self.remove_no_word_segments(reassign_ids=False)
        self.reassign_ids()
        self._update_history(
            f'rp={max_words}+{int(case_sensitive)}+{int(strip)}'
            f'+{ignore_punctuations}+{int(extend_duration)}+{int(verbose)}'
        )
        return self

    def remove_words_by_str(
            self,
            words: Union[str, List[str], None],
            case_sensitive: bool = False,
            strip: bool = True,
            ignore_punctuations: str = "\"',.?!",
            min_prob: float = None,
            filters: Callable = None,
            verbose: bool = True
    ) -> 'WhisperResult':
        """
        Remove words that match ``words``.

        Parameters
        ----------
        words : str or list of str or None
            A word or list of words to remove.``None`` for all words to be passed into ``filters``.
        case_sensitive : bool, default False
            Whether the case of words need to match to be considered as repetition.
        strip : bool, default True
            Whether to ignore spaces before and after each word.
        ignore_punctuations : bool, default '"',.?!'
            Ending punctuations to ignore.
        min_prob : float, optional
            Acts as the first filter the for the words that match ``words``. Words with probability < ``min_prob`` will
            be removed if ``filters`` is ``None``, else pass the words into ``filters``. Words without probability will
            be treated as having probability < ``min_prob``.
        filters : Callable, optional
            A function that takes an instance of :class:`stable_whisper.result.WordTiming` as its only argument.
            This function is custom filter for the words that match ``words`` and were not caught by ``min_prob``.
        verbose:
            Whether to print detail of the removed words.

        Returns
        -------
        stable_whisper.result.WhisperResult
            The current instance after the changes.
        """
        if not self.has_words:
            return self
        if isinstance(words, str):
            words = [words]
        elif words == 0:
            words = None
        filters = _get_content(self, filters)
        all_words = self.all_words()
        all_words_str = [w.word for w in all_words]
        if strip:
            all_words_str = [w.strip() for w in all_words_str]
            words = [w.strip() for w in words]
        if ignore_punctuations:
            ptn = f'[{ignore_punctuations}]+$'
            all_words_str = [re.sub(ptn, '', w) for w in all_words_str]
            words = [re.sub(ptn, '', w) for w in words]
        if not case_sensitive:
            all_words_str = [w.lower() for w in all_words_str]
            words = [w.lower() for w in words]

        changes = []
        for i, w in reversed(list(enumerate(all_words_str))):
            if not (words is None or any(w == _w for _w in words)):
                continue
            if (
                    (min_prob is None or all_words[i].probability is None or min_prob > all_words[i].probability) and
                    (filters is None or filters(all_words[i]))
            ):
                if verbose:
                    changes.append(f'Removed: {all_words[i].to_dict()}')
                self.remove_word(all_words[i], False, verbose=False, record=False)
        if changes:
            print('\n'.join(reversed(changes)))
        self.remove_no_word_segments()
        words = 0 if words is None else '/'.join(words)
        self._update_history(
            f'rws={words}+{int(case_sensitive)}+{int(strip)}'
            f'+{ignore_punctuations}+{min_prob}+{_store_content(self, filters)}+{int(verbose)}'
        )
        return self

    def fill_in_gaps(
            self,
            other_result: Union['WhisperResult', str],
            min_gap: float = 0.1,
            case_sensitive: bool = False,
            strip: bool = True,
            ignore_punctuations: str = "\"',.?!",
            verbose: bool = True
    ) -> 'WhisperResult':
        """
        Fill in segment gaps larger than ``min_gap`` with content from ``other_result`` at the times of gaps.

        Parameters
        ----------
        other_result : WhisperResult or str
            Another transcription result as an instance of :class:`stable_whisper.result.WhisperResult` or path to the
            JSON of the result.
        min_gap : float, default 0.1
            The minimum seconds of a gap between segments that must be exceeded to be filled in.
        case_sensitive : bool, default False
            Whether to consider the case of the first and last word of the gap to determine overlapping words to remove
            before filling in.
        strip : bool, default True
            Whether to ignore spaces before and after the first and last word of the gap to determine overlapping words
            to remove before filling in.
        ignore_punctuations : bool, default '"',.?!'
            Ending punctuations to ignore in the first and last word of the gap to determine overlapping words to
            remove before filling in.
        verbose:
            Whether to print detail of the filled content.

        Returns
        -------
        stable_whisper.result.WhisperResult
            The current instance after the changes.
        """
        if len(self.segments) < 2:
            return self
        other_result = _get_content(self, other_result)
        if isinstance(other_result, str):
            other_path = other_result
            other_result = WhisperResult(other_result)
        else:
            other_path = _store_content(self, other_result)

        if strip:
            def strip_space(w):
                return w.strip()
        else:
            def strip_space(w):
                return w

        if ignore_punctuations:
            ptn = f'[{ignore_punctuations}]+$'

            def strip_punctuations(w):
                return re.sub(ptn, '', strip_space(w))
        else:
            strip_punctuations = strip_space

        if case_sensitive:
            _strip = strip_punctuations
        else:
            def _strip(w):
                return strip_punctuations(w).lower()

        seg_pairs = list(enumerate(zip(self.segments[:-1], self.segments[1:])))
        seg_pairs.insert(0, (-1, (None, self.segments[0])))
        seg_pairs.append((seg_pairs[-1][0]+1, (self.segments[-1], None)))

        changes = []
        for i, (seg0, seg1) in reversed(seg_pairs):
            first_word = None if seg0 is None else seg0.words[-1]
            last_word = None if seg1 is None else seg1.words[0]
            start = (other_result[0].start if first_word is None else first_word.end)
            end = other_result[-1].end if last_word is None else last_word.start
            if end - start <= min_gap:
                continue
            gap_words = other_result.get_content_by_time((start, end))
            if first_word is not None and gap_words and _strip(first_word.word) == _strip(gap_words[0].word):
                first_word.end = gap_words[0].end
                gap_words = gap_words[1:]
            if last_word is not None and gap_words and _strip(last_word.word) == _strip(gap_words[-1].word):
                last_word.start = gap_words[-1].start
                gap_words = gap_words[:-1]
            if not gap_words:
                continue
            if last_word is not None and last_word.start < gap_words[-1].end:
                last_word.start = gap_words[-1].end
            new_segments = [other_result[gap_words[0].segment_id].copy([])]
            for j, new_word in enumerate(gap_words):
                new_word_copy = new_word.copy(copy_tokens=True)
                if j == 0 and first_word is not None and first_word.end > gap_words[0].start:
                    new_word_copy.start = first_word.end
                if new_segments[-1].id != new_word.segment_id:
                    new_segments.append(other_result[new_word.segment_id].copy([]))
                new_segments[-1].words.append(new_word_copy)
            if verbose:
                changes.append('\n'.join('Added: ' + s.to_display_str(True) for s in new_segments))
            self.segments = self.segments[:i+1] + new_segments + self.segments[i+1:]
        if changes:
            print('\n'.join(reversed(changes)))
        self.reassign_ids()
        self._update_history(
            f'fg={other_path}+{min_gap}+{int(case_sensitive)}+{int(strip)}+{ignore_punctuations}+{int(verbose)}'
        )
        return self

    def adjust_gaps(
            self,
            duration_threshold: float = 0.75,
            one_section: bool = False
    ) -> "WhisperResult":
        """
        Set segment timestamps to the timestamps of overlapping``nonspeech_sections`` within ``duration_threshold``.

        For results with segments split by gaps or punctuations that indicate pauses in speech, the timestamps of these
        gaps are replaced with those in ``nonspeech_sections`` which overlap. These timestamps tend to be more accurate
        if not exact timestamps of the start and end of utterances, especially for audio with no sounds in the gaps.

        Parameters
        ----------
        duration_threshold : float, default 0.75
            Only ``nonspeech_sections`` with duration >= ``duration_threshold`` * duration of the longest section that
            overlaps the existing gap between segments are considered for that gap. The Lower the value the more
            ``nonspeech_sections`` are considered.
        one_section : bool, default False
            Whether to use only one section of ``nonspeech_sections`` for each gap between segments.
            Recommend ``True`` when there are no sounds in the gaps.

        Returns
        -------
        stable_whisper.result.WhisperResult
            The current instance after the changes.

        Notes
        -----
        The reliability of this approach hinges on the accuracy of the ``nonspeech_sections`` which is computed when
        ``vad`` is ``False`` or ``True`` for transcription and alignment. The accuracy increases with cleaner audio.
        Results produced by ``vad=True`` will work better with less clean audio (i.e. audio with non-speech sounds).

        """
        if duration_threshold > 1:
            raise ValueError(f'``duration_threshold`` must be at most 1.0 but got {duration_threshold}')

        ns_idx = 0
        for seg_idx in range(-1, len(self.segments)):
            curr_part = None if seg_idx == -1 else self.segments[seg_idx]
            next_part = None if curr_part is self.segments[-1] else self.segments[seg_idx + 1]
            curr_start = curr_end = next_start = next_end = None
            if self.has_words:
                if curr_part is None:
                    word_duration = np.median([word.duration for word in next_part]) * 2
                    curr_start = curr_end = max(next_part.start - word_duration, 0)
                if next_part is None:
                    med_duration = np.median([word.duration for word in curr_part]) * 2
                    next_start = next_end = curr_part.end + med_duration
                if curr_part is not None:
                    curr_part = curr_part[-1]
                if next_part is not None:
                    next_part = next_part[0]
            else:
                if curr_part is None:
                    curr_start = curr_end = max(next_part.start - next_part.duration, 0)
                if next_part is None:
                    next_start = next_end = curr_part.end + curr_part.duration

            if curr_start is None:
                curr_start = curr_part.start
            if curr_end is None:
                curr_end = curr_part.end
            if next_start is None:
                next_start = next_part.start
            if next_end is None:
                next_end = next_part.end

            nonspeech_sections: List[Tuple[float, float]] = []
            for ns_idx in range(ns_idx, len(self.nonspeech_sections)):
                section = self.nonspeech_sections[ns_idx]
                nonspeech_start = section['start']
                nonspeech_end = section['end']
                valid_start = curr_start < (nonspeech_end if curr_part is None else nonspeech_start)
                valid_end = (nonspeech_start if next_part is None else nonspeech_end) < next_end
                if valid_start and valid_end:
                    nonspeech_sections.append((nonspeech_start, nonspeech_end))
                if next_start < nonspeech_start:
                    break
            if not nonspeech_sections:
                continue
            nonspeech_durations = np.array([end - start for start, end in nonspeech_sections])
            duration_indices = np.argsort(nonspeech_durations)
            nonspeech_durations = nonspeech_durations[duration_indices]
            nonspeech_duration_pct = nonspeech_durations / nonspeech_durations[-1]
            valid_nonspeech_mask = nonspeech_duration_pct >= duration_threshold
            if not np.any(valid_nonspeech_mask):
                continue
            duration_indices = duration_indices[valid_nonspeech_mask]

            curr_scores = np.array([abs(nonspeech_sections[i][0] - curr_end) for i in duration_indices])
            next_scores = np.array([abs(nonspeech_sections[i][1] - next_start) for i in duration_indices])

            if one_section:
                score_idx_idx = np.argmin(curr_scores + next_scores)
                best_curr_score_idx = duration_indices[score_idx_idx]
                best_next_score_idx = duration_indices[score_idx_idx]
            else:
                best_curr_score_idx = duration_indices[np.argmin(curr_scores)]
                best_next_score_idx = duration_indices[np.argmin(next_scores)]
                if best_curr_score_idx > best_next_score_idx:
                    score_idx_idx = np.argmin(curr_scores + next_scores)
                    best_curr_score_idx = duration_indices[score_idx_idx]
                    best_next_score_idx = duration_indices[score_idx_idx]

            new_end = nonspeech_sections[best_curr_score_idx][0]
            if curr_part is not None and curr_start < new_end:
                curr_part.end = new_end

            new_start = nonspeech_sections[best_next_score_idx][1]
            if next_part is not None and new_start < next_end:
                next_part.start = new_start
        self._update_history(f'ag={duration_threshold}+{int(one_section)}')
        return self

    def convert_to_segment_level(self) -> "WhisperResult":
        """
        Remove all word-level data.

        Converting from word-level to segment-level is useful when the word-level data restricts certain operations that
        need to be performed at the segment-level or when the segment-level results of operations are more desirable.

        Returns
        -------
        stable_whisper.result.WhisperResult
            The current instance after the changes.

        Notes
        -----
        Since many regrouping operations require word-timestamps, it is recommended to only use this method after the
        result has been regrouped into its final segments.

        """
        for seg in self.segments:
            seg.convert_to_segment_level()
        self._update_history('csl')
        return self

    def custom_operation(
            self,
            key: str,
            operator: Union[str, Callable],
            value,
            method: Union[str, Callable],
            word_level: Optional[bool] = None
    ) -> "WhisperResult":
        """
        Perform custom operation on words/segments that satisfy specified conditions.

        Parameters
        ----------
        key : str
            Attribute of words/segments to get value for comparing with ``value``.
            If ``key`` is empty string, directly use the instance of ``WordTiming``/``Segment`` object for comparison.
            All " " in ``key`` will be replaced with "_".
            Start with "len=" to get length of the attribute.
        operator : str or Callable
            Operator or function for comparing vaLue of ``key`` and ``value``.
            Built-in operators: ==, >, >=, <, <=, is, in, start, end
            Function must return a boolean and accept two arguments: vaLue of ``key``, ``value``.
        value : any
            Value to compare with vaLue of ``key``.
            To express multiple values as a string:
                -start string with "all=" or "any=" followed with values separated by ','
                -use "\" to escape values with "," in values
                -these values can only be conbination of str, int, float
                -start with "all=": all values are required to meet the condition to be satisfied
                -start with "any=": only require one of the values to meet the conditions to be satified
        method : str or Callable
            Operation or function to check if words/segments meet conditions.
            Built-in operations:
                mergeleft, mergeright, merge, lockright, lockleft, lock, splitright, splitleft, split, remove
            Function must accept three arguments: this instance of `WhisperResult`, segment index, word index.
        word_level : bool, default None, meaning True if this instance has word-timestamps
            Whether to operate on words. Operate on segments if ``False``.

        Returns
        -------
        stable_whisper.result.WhisperResult
            The current instance after the changes.

        Examples
        --------
        Split on the right of words that end with "." or "," or "?":
        >>>result.custom_operation('word', 'end', 'any=.,\,,?', 'splitright', word_level=True)

        Merge with next segment if the current segment is less than 15 chars:
        >>>result.custom_operation('len=text', '<', 15, 'mergeright', word_level=False)

        If there is split after the word " Mr." or " Ms." or " Mrs.", merge with next segment:
        >>>result.custom_operation('word', 'in', (' Mr.', ' Ms.', ' Mrs.'), 'mergeright', word_level=True)
        same as:
        >>>result.custom_operation('word', '==', 'any= Mr., Ms., Mrs', 'mergeright', word_level=True)

        Replace 'y' with 'ie' for capitalized words:
        >>>def is_match(word, value):
        >>>    text = word.word.strip().lower()
        >>>    return text == 'one' and word.probability < value
        >>>def replace_one(result, seg_index, word_index):
        >>>    word = result[seg_index][word_index]
        >>>    word.word = word.word.lower().replace('one', '1')
        >>>result.custom_operation('', is_match, 0.5, replace_one, word_level=True)

        """
        if self.has_words:
            if word_level is None:
                word_level = True
        elif word_level:
            raise ValueError(f'result is missing word timestamps and not compatible with ``word_level=True``')

        value = _get_content(self, value, strict=False)

        methods = (
            'mergeleft', 'mergeright', 'merge',
            'lockright', 'lockleft', 'lock',
            'splitright', 'splitleft', 'split',
            'remove'
        )
        method = _get_content(self, method)
        is_builtin_method = isinstance(method, str)
        if is_builtin_method:
            if method not in methods:
                raise ValueError(f"invalid method: '{method}'. Valid methods: {method}")
        elif not callable(method):
            raise TypeError(f"'{type(method)}' object is not callable")
        key = key.replace(' ', '_')
        operator = _get_content(self, operator)
        if isinstance(operator, str):
            operators = {
                '==': lambda a, b: a == b,
                '>': lambda a, b: a > b,
                '>=': lambda a, b: a >= b,
                '<': lambda a, b: a < b,
                '<=': lambda a, b: a <= b,
                'is': lambda a, b: a is b,
                'in': lambda a, b: a in b,
                'start': str.startswith,
                'end': str.endswith
            }
            if operator not in operators:
                raise ValueError(f"invalid operator: '{operators}'. Valid operators: {tuple(operators.keys())}")
            operator_str = operator
            operator = operators[operator]
        else:
            operator_str = _store_content(self, operator)

        method_str = method
        if not is_builtin_method:
            method_str = _store_content(self, method)
        elif method.startswith('merge'):

            def get_left_indices(result: "WhisperResult", seg_idx: int, word_idx: int = None):
                if seg_idx == 0 or (word_idx is not None and word_idx != 0):
                    return []
                return [(seg_idx - 1, seg_idx)]

            def get_right_indices(result: "WhisperResult", seg_idx: int, word_idx: int = None):
                if (
                        seg_idx + 1 >= len(result.segments) or
                        (word_idx is not None and word_idx != len(result[seg_idx].words) -1)
                ):
                    return []
                return [(seg_idx, seg_idx + 1)]

            if method == 'mergeright':
                get_indices = get_right_indices
            elif method == 'mergeleft':
                get_indices = get_left_indices
            else:
                def get_indices(*args):
                    return get_right_indices(*args) + get_left_indices(*args)

            def method(result: "WhisperResult", seg_idx: int, word_idx: int = None):
                indices = get_indices(result, seg_idx, word_idx)
                for idxs in indices:
                    result.add_segments(*idxs, inplace=True, reassign_ids=False)

        elif method.startswith('lock'):
            if method == 'lockright':
                def method(result: "WhisperResult", seg_idx: int, word_idx: int = None):
                    if word_idx is None:
                        result[seg_idx].lock_right()
                    else:
                        result[seg_idx][word_idx].lock_right()
            elif method == 'lockleft':
                def method(result: "WhisperResult", seg_idx: int, word_idx: int = None):
                    if word_idx is None:
                        result[seg_idx].lock_left()
                    else:
                        result[seg_idx][word_idx].lock_left()
            else:
                def method(result: "WhisperResult", seg_idx: int, word_idx: int = None):
                    if word_idx is None:
                        result[seg_idx].lock_right()
                        result[seg_idx].lock_left()
                    else:
                        result[seg_idx][word_idx].lock_right()
                        result[seg_idx][word_idx].lock_left()

        elif method.startswith('split'):
            if word_level is None:
                raise ValueError('Segment-level result is not compatible with split actions.')
            elif not word_level:
                raise ValueError('``word_level=False`` is not compatible with split actions.')

            if method == 'splitright':
                def method(result: "WhisperResult", seg_idx: int, word_idx: int = None):
                    if word_idx == len(result[seg_idx].words) + 1:
                        return
                    result.split_segment_by_index(result[seg_idx], word_idx, reassign_ids=False)
            elif method == 'splitleft':
                def method(result: "WhisperResult", seg_idx: int, word_idx: int = None):
                    if word_idx == 0:
                        return
                    result.split_segment_by_index(result[seg_idx], word_idx - 1, reassign_ids=False)
            else:
                def method(result: "WhisperResult", seg_idx: int, word_idx: int = None):
                    indices = []
                    if word_idx != 0:
                        indices.append(word_idx - 1)
                    if word_idx < len(result[seg_idx].words) + 1:
                        indices.append(word_idx)
                    result.split_segment_by_index(result[seg_idx], indices, reassign_ids=False)

        else:  # remove
            def method(result: "WhisperResult", seg_idx: int, word_idx: int = None):
                if word_idx is None:
                    result.remove_segment(result[seg_idx], reassign_ids=False, record=False)
                else:
                    result.remove_word(result[seg_idx][word_idx], reassign_ids=False, record=False)

        if key.startswith('len='):
            key_ = key[4:]

            def get_value(obj):
                return len(getattr(obj, key_))
        elif key == '':
            def get_value(obj):
                return obj
        else:
            def get_value(obj):
                return getattr(obj, key)

        if isinstance(value, str) and (value.startswith('all=') or value.startswith('any=')):
            check = any if value.startswith('any=') else all
            values = [val.replace('\\,', ',') for val in re.split(r'(?<!\\),', value[4:])]

            def is_satisfied(obj):
                return check(operator(get_value(obj), val) for val in values)
        else:
            def is_satisfied(obj):
                return operator(get_value(obj), value)

        if word_level:
            for segment_index in range(len(self.segments)-1, -1, -1):
                for word_index in range(len(self.segments[segment_index].words)-1, -1, -1):
                    word = self[segment_index][word_index]
                    if not is_satisfied(word):
                        continue
                    method(self, segment_index, word_index)
        else:
            for segment_index in range(len(self.segments) - 1, -1, -1):
                segment = self[segment_index]
                if not is_satisfied(segment):
                    continue
                method(self, segment_index, None)

        self.reassign_ids()
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            if isinstance(value, bool):
                value = f'<{value}>'
            else:
                value = _store_content(self, value)
        self._update_history(
            f'co={key.replace("_", " ")}+{operator_str}+{value}+{method_str}+{int(word_level)}'
        )
        return self

    def regroup(
            self,
            regroup_algo: Union[str, bool] = None,
            verbose: bool = False,
            only_show: bool = False
    ) -> "WhisperResult":
        """
        Regroup (in-place) words into segments.

        Parameters
        ----------
        regroup_algo: str or bool, default 'da'
             String representation of a custom regrouping algorithm or ``True`` use to the default algorithm 'da'.
        verbose : bool, default False
            Whether to show all the methods and arguments parsed from ``regroup_algo``.
        only_show : bool, default False
            Whether to show the all methods and arguments parsed from ``regroup_algo`` without running the methods

        Returns
        -------
        stable_whisper.result.WhisperResult
            The current instance after the changes.

        Notes
        -----
        Syntax for string representation of custom regrouping algorithm.
            Method keys:
                sg: split_by_gap
                sp: split_by_punctuation
                sl: split_by_length
                sd: split_by_duration
                mg: merge_by_gap
                mp: merge_by_punctuation
                ms: merge_all_segment
                cm: clamp_max
                l: lock
                us: unlock_all_segments
                da: default algorithm (isp_cm_sp=.* /。/?/？_sg=.5_sp=,* /，++++50_sl=70_cm)
                rw: remove_word
                rs: remove_segment
                rp: remove_repetition
                rws: remove_words_by_str
                fg: fill_in_gaps
                p: pad
                ag: adjust_gaps
                csl: convert_to_segment_level
                co: custom_operation
                isp: ignore_special_periods
            Metacharacters:
                = separates a method key and its arguments (not used if no argument)
                _ separates method keys (after arguments if there are any)
                + separates arguments for a method key
                / separates an argument into list of strings
                * separates an item in list of strings into a nested list of strings
            Notes:
            -arguments are parsed positionally
            -if no argument is provided, the default ones will be used
            -use 1 or 0 to represent True or False
            -use a space to in place of _ for ``key`` of ``custom_operation`` to avoid conflict with the metacharacter
            -use strings "<True>" or "<False>" to represent True or False for ``value`` of ``custom_operation``
            Example 1:
                merge_by_gap(.2, 10, lock=True)
                mg=.2+10+++1
                Note: [lock] is the 5th argument hence the 2 missing arguments inbetween the three + before 1
            Example 2:
                split_by_punctuation([('.', ' '), '。', '?', '？'], True)
                sp=.* /。/?/？+1
            Example 3:
                merge_all_segments().split_by_gap(.5).merge_by_gap(.15, 3)
                ms_sg=.5_mg=.15+3
            Examples 4:
                custom_operation('compression_ratio', '>', 3.0, 'remove', word_level=False)
                co=compression ratio+>+3.0+remove+0
        """
        if regroup_algo is False:
            return self
        if regroup_algo is None or regroup_algo is True:
            regroup_algo = 'da'

        for method, kwargs, msg in self.parse_regroup_algo(regroup_algo, include_str=verbose or only_show):
            if msg:
                print(msg)
            if not only_show:
                method(**kwargs)

        return self

    def parse_regroup_algo(self, regroup_algo: str, include_str: bool = True) -> List[Tuple[Callable, dict, str]]:
        methods = dict(
            sg=self.split_by_gap,
            sp=self.split_by_punctuation,
            sl=self.split_by_length,
            sd=self.split_by_duration,
            mg=self.merge_by_gap,
            mp=self.merge_by_punctuation,
            ms=self.merge_all_segments,
            cm=self.clamp_max,
            us=self.unlock_all_segments,
            l=self.lock,
            rw=self.remove_word,
            rs=self.remove_segment,
            rp=self.remove_repetition,
            rws=self.remove_words_by_str,
            fg=self.fill_in_gaps,
            p=self.pad,
            ag=self.adjust_gaps,
            csl=self.convert_to_segment_level,
            co=self.custom_operation,
            isp=self.ignore_special_periods,
        )
        if not regroup_algo:
            return []

        calls = regroup_algo.split('_')
        if 'da' in calls:
            default_calls = 'isp_cm_sp=.* /。/?/？_sg=.5_sp=,* /，++++50_sl=70_cm'.split('_')
            calls = chain.from_iterable(default_calls if method == 'da' else [method] for method in calls)
        operations = []
        for method in calls:
            method, args = method.split('=', maxsplit=1) if '=' in method else (method, '')
            if method not in methods:
                raise NotImplementedError(f'{method} is not one of the available methods: {tuple(methods.keys())}')
            args = [] if len(args) == 0 else list(map(str_to_valid_type, args.split('+')))
            kwargs = {k: v for k, v in zip(methods[method].__code__.co_varnames[1:], args) if v is not None}
            if include_str:
                kwargs_str = ', '.join(f'{k}="{v}"' if isinstance(v, str) else f'{k}={v}' for k, v in kwargs.items())
                op_str = f'{methods[method].__name__}({kwargs_str})'
            else:
                op_str = None
            operations.append((methods[method], kwargs, op_str))

        return operations

    def find(self, pattern: str, word_level=True, flags=None) -> "WhisperResultMatches":
        """
        Find segments/words and timestamps with regular expression.

        Parameters
        ----------
        pattern : str
            RegEx pattern to search for.
        word_level : bool, default True
            Whether to search at word-level.
        flags : optional
            RegEx flags.

        Returns
        -------
        stable_whisper.result.WhisperResultMatches
            An instance of :class:`stable_whisper.result.WhisperResultMatches` with word/segment that match ``pattern``.
        """
        return WhisperResultMatches(self).find(pattern, word_level=word_level, flags=flags)

    @property
    def text(self):
        return ''.join(s.text for s in self.segments)

    @property
    def regroup_history(self):
        # same syntax as ``regroup_algo`` for :meth:``result.WhisperResult.regroup`
        return self._regroup_history

    @property
    def nonspeech_sections(self):
        return self._nonspeech_sections

    def show_regroup_history(self):
        """
        Print details of all regrouping operations that been performed on data.
        """
        if not self._regroup_history:
            print('Result has no history.')
        for *_, msg in self.parse_regroup_algo(self._regroup_history):
            print(f'.{msg}')

    def __len__(self):
        return len(self.segments)

    def unlock_all_segments(self):
        for s in self.segments:
            s.unlock_all_words()
        return self

    def set_current_as_orig(self, keep_orig: bool = False):
        """
        Overwrite all values in ``ori_dict`` with current values.
        """
        self.ori_dict = self.to_dict(keep_orig=keep_orig)

    def reset(self):
        """
        Restore all values to that at initialization in ``ori_dict``.
        """
        self.language = self.ori_dict.get('language')
        self._regroup_history = ''
        segments = self.ori_dict.get('segments')
        self.segments = [Segment(**s, ignore_unused_args=True) for s in segments] if segments else []
        if self._forced_order:
            self.force_order()
        self.remove_no_word_segments(any(seg.has_words for seg in self.segments))

    @property
    def has_words(self):
        return bool(self.segments) and all(seg.has_words for seg in self.segments)

    to_srt_vtt = result_to_srt_vtt
    to_ass = result_to_ass
    to_tsv = result_to_tsv
    to_txt = result_to_txt
    save_as_json = save_as_json


class SegmentMatch:

    def __init__(
            self,
            segments: Union[List[Segment], Segment],
            _word_indices: List[List[int]] = None,
            _text_match: str = None
    ):
        self.segments = [segments] if isinstance(segments, Segment) else segments
        self.word_indices = [] if _word_indices is None else _word_indices
        self.words = [self.segments[i].words[j] for i, indices in enumerate(self.word_indices) for j in indices]
        if len(self.words) != 0:
            self.text = ''.join(
                self.segments[i].words[j].word
                for i, indices in enumerate(self.word_indices)
                for j in indices
            )
        else:
            self.text = ''.join(seg.text for seg in self.segments)
        self.text_match = _text_match

    @property
    def start(self):
        return (
            self.words[0].start
            if len(self.words) != 0 else
            (self.segments[0].start if len(self.segments) != 0 else None)
        )

    @property
    def end(self):
        return (
            self.words[-1].end
            if len(self.words) != 0 else
            (self.segments[-1].end if len(self.segments) != 0 else None)
        )

    def __len__(self):
        return len(self.segments)

    def __repr__(self):
        return self.__dict__.__repr__()

    def __str__(self):
        return self.__dict__.__str__()


class WhisperResultMatches:
    """
    RegEx matches for WhisperResults.
    """
    # Use WhisperResult.find() instead of instantiating this class directly.
    def __init__(
            self,
            matches: Union[List[SegmentMatch], WhisperResult],
            _segment_indices: List[List[int]] = None
    ):
        if isinstance(matches, WhisperResult):
            self.matches = list(map(SegmentMatch, matches.segments))
            self._segment_indices = [[i] for i in range(len(matches.segments))]
        else:
            self.matches = matches
            assert _segment_indices is not None
            assert len(self.matches) == len(_segment_indices)
            assert all(len(match.segments) == len(_segment_indices[i]) for i, match in enumerate(self.matches))
            self._segment_indices = _segment_indices

    @property
    def segment_indices(self):
        return self._segment_indices

    def _curr_seg_groups(self) -> List[List[Tuple[int, Segment]]]:
        seg_groups, curr_segs = [], []
        curr_max = -1
        for seg_indices, match in zip(self._segment_indices, self.matches):
            for i, seg in zip(sorted(seg_indices), match.segments):
                if i > curr_max:
                    curr_segs.append((i, seg))
                    if i - 1 != curr_max:
                        seg_groups.append(curr_segs)
                        curr_segs = []
                    curr_max = i

        if curr_segs:
            seg_groups.append(curr_segs)
        return seg_groups

    def find(self, pattern: str, word_level=True, flags=None) -> "WhisperResultMatches":
        """
        Find segments/words and timestamps with regular expression.

        Parameters
        ----------
        pattern : str
            RegEx pattern to search for.
        word_level : bool, default True
            Whether to search at word-level.
        flags : optional
            RegEx flags.

        Returns
        -------
        stable_whisper.result.WhisperResultMatches
            An instance of :class:`stable_whisper.result.WhisperResultMatches` with word/segment that match ``pattern``.
        """

        seg_groups = self._curr_seg_groups()
        matches: List[SegmentMatch] = []
        match_seg_indices: List[List[int]] = []
        if word_level:
            if not all(all(seg.has_words for seg in match.segments) for match in self.matches):
                warnings.warn('Cannot perform word-level search with segment(s) missing word timestamps.')
                word_level = False

        for segs in seg_groups:
            if word_level:
                idxs = list(chain.from_iterable(
                    [(i, j)]*len(word.word) for (i, seg) in segs for j, word in enumerate(seg.words)
                ))
                text = ''.join(word.word for (_, seg) in segs for word in seg.words)
            else:
                idxs = list(chain.from_iterable([(i, None)]*len(seg.text) for (i, seg) in segs))
                text = ''.join(seg.text for (_, seg) in segs)
            assert len(idxs) == len(text)
            for curr_match in re.finditer(pattern, text, flags=flags or 0):
                start, end = curr_match.span()
                curr_idxs = idxs[start: end]
                curr_seg_idxs = sorted(set(i[0] for i in curr_idxs))
                if word_level:
                    curr_word_idxs = [
                        sorted(set(j for i, j in curr_idxs if i == seg_idx))
                        for seg_idx in curr_seg_idxs
                    ]
                else:
                    curr_word_idxs = None
                matches.append(SegmentMatch(
                    segments=[s for i, s in segs if i in curr_seg_idxs],
                    _word_indices=curr_word_idxs,
                    _text_match=curr_match.group()
                ))
                match_seg_indices.append(curr_seg_idxs)
        return WhisperResultMatches(matches, match_seg_indices)

    def __len__(self):
        return len(self.matches)

    def __bool__(self):
        return self.__len__() != 0

    def __getitem__(self, idx):
        return self.matches[idx]

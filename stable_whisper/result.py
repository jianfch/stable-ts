import warnings
import re
import numpy as np
from typing import Union, List, Tuple
from dataclasses import dataclass
from copy import deepcopy
from itertools import chain

from .stabilization import suppress_silence
from .text_output import *


__all__ = ['WhisperResult', 'Segment']


@dataclass
class WordTiming:
    word: str
    start: float
    end: float
    probability: float
    tokens: List[int] = None
    left_locked: bool = False
    right_locked: bool = False

    def __len__(self):
        return len(self.word)

    def __add__(self, other: 'WordTiming'):
        assert self.start <= other.start or self.end <= other.end

        self_copy = deepcopy(self)

        self_copy.start = min(self_copy.start, other.start)
        self_copy.end = max(other.end, self_copy.end)
        self_copy.word += other.word
        self_copy.probability = (other.probability + self_copy.probability) / 2
        self_copy.tokens.extend(other.tokens)
        self_copy.left_locked = self_copy.left_locked or other.left_locked
        self_copy.right_locked = self_copy.right_locked or other.right_locked

        return self_copy

    @property
    def duration(self):
        return self.end - self.start

    def offset_time(self, offset_seconds: float):
        self.start = self.start + offset_seconds
        self.end = self.end + offset_seconds

    def to_dict(self):
        dict_ = deepcopy(self.__dict__)
        dict_.pop('left_locked')
        dict_.pop('right_locked')
        return dict_

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
                         min_word_dur: float = 0.1):
        suppress_silence(self, silent_starts, silent_ends, min_word_dur)
        return self

    def rescale_time(self, scale_factor: float):
        self.start = round(self.start * scale_factor, 3)
        self.end = round(self.end * scale_factor, 3)


@dataclass
class Segment:
    seek: float
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    id: int = None
    words: Union[List[WordTiming], List[dict]] = None
    ori_has_words: bool = None

    @property
    def has_words(self):
        return bool(self.words)

    @property
    def duration(self):
        return self.end - self.start

    def word_count(self):
        if self.has_words:
            return len(self.words)

    def char_count(self):
        if self.has_words:
            return sum(len(w) for w in self.words)
        return len(self.text)

    def __post_init__(self):
        if self.has_words:
            self.words: List[WordTiming] = \
                [WordTiming(**word) if isinstance(word, dict) else word for word in self.words]
        if self.ori_has_words is None:
            self.ori_has_words = self.has_words

    def __add__(self, other: 'Segment'):
        assert self.start <= other.start or self.end <= other.end

        self_copy = deepcopy(self)

        self_copy.start = min(self_copy.start, other.start)
        self_copy.end = max(other.end, self_copy.end)
        self_copy.text += other.text
        self_copy.tokens.extend(other.tokens)
        if self_copy.has_words:
            self_copy.words.extend(other.words)

        self_copy.temperature = (other.temperature + self_copy.temperature) / 2
        self_copy.avg_logprob = (other.avg_logprob + self_copy.avg_logprob) / 2
        self_copy.compression_ratio = (other.compression_ratio + self_copy.compression_ratio) / 2
        self_copy.no_speech_prob = (other.no_speech_prob + self_copy.no_speech_prob) / 2

        return self_copy

    def offset_time(self, offset_seconds: float):
        self.seek = self.seek + offset_seconds
        self.start = self.start + offset_seconds
        self.end = self.end + offset_seconds
        if self.has_words:
            for w in self.words:
                w.offset_time(offset_seconds)

    def add_words(self, index0: int, index1: int, inplace: bool = False):
        new_word = self.words[index0] + self.words[index1]
        if inplace:
            i0, i1 = sorted([index0, index1])
            self.words[i0] = new_word
            del self.words[i1]
        return new_word

    def rescale_time(self, scale_factor: float):
        self.seek = round(self.seek * scale_factor, 3)
        self.start = round(self.start * scale_factor, 3)
        self.end = round(self.end * scale_factor, 3)
        if self.has_words:
            for w in self.words:
                w.rescale_time(scale_factor)
        self.update_seg_with_words()

    def apply_min_dur(self, min_dur: float, inplace: bool = False):
        """
        Any duration is less than [min_dur] will be merged with adjacent word.
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
                    if segment.words[i-1].duration < segment.words[i-1].duration:
                        segment.add_words(i-1, i, inplace=True)
                    else:
                        segment.add_words(i, i+1, inplace=True)
                max_i -= 1
        return segment

    def _to_reverse_text(
            self,
            prepend_punctuations: str = None,
            append_punctuations: str = None
    ):
        """

        Returns
        -------
        A copy with words reversed order per segment

        """
        if prepend_punctuations is None:
            prepend_punctuations = "\"'“¿([{-"
        if prepend_punctuations and ' ' not in prepend_punctuations:
            prepend_punctuations += ' '
        if append_punctuations is None:
            append_punctuations = "\"'.。,，!！?？:：”)]}、"
        self_copy = deepcopy(self)
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
            self_copy.text = ''.join(w.word for w in reversed(word_objs))

        return self_copy

    def to_dict(self, reverse_text: Union[bool, tuple] = False):
        if reverse_text:
            seg_dict = (
                (self._to_reverse_text(*reverse_text)
                 if isinstance(reverse_text, tuple) else
                 self._to_reverse_text()).__dict__
            )
        else:
            seg_dict = deepcopy(self.__dict__)
        seg_dict.pop('ori_has_words')
        if self.has_words:
            seg_dict['words'] = [w.to_dict() for w in seg_dict['words']]
        elif self.ori_has_words:
            seg_dict['words'] = []
        else:
            seg_dict.pop('words')
        if self.id is None:
            seg_dict.pop('id')
        if reverse_text:
            seg_dict['reversed_text'] = True
        return seg_dict

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
        if self.has_words:
            for w in self.words:
                w.unlock_both()

    def update_seg_with_words(self):
        if self.has_words:
            self.start = self.words[0].start
            self.end = self.words[-1].end
            if self.words[0].tokens:
                self.tokens = [t for w in self.words for t in w.tokens]
            self.text = ''.join(w.word for w in self.words)

    def suppress_silence(self,
                         silent_starts: np.ndarray,
                         silent_ends: np.ndarray,
                         min_word_dur: float = 0.1,
                         word_level: bool = True):
        if self.has_words:
            words = self.words if word_level or len(self.words) == 1 else [self.words[0], self.words[-1]]
            for w in words:
                w.suppress_silence(silent_starts, silent_ends)
            self.update_seg_with_words()
        else:
            suppress_silence(self,
                             silent_starts,
                             silent_ends,
                             max((self.end - self.start) * .75, min_word_dur))

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

    def get_length_indices(self, max_chars: int = None, max_words: int = None):  # for splitting
        if max_chars is None and max_words is None:
            return []
        assert max_chars != 0 and max_words != 0, \
            f'max_chars and max_words must be greater 0, but got {max_chars} and {max_words}'
        indices = []
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
                ):
                    indices.append(i-1)
                    curr_words = 1
                    curr_chars = len(word)
        return indices

    def split(self, indices: List[int]):
        if len(indices) == 0:
            return []
        if indices[-1] != len(self.words) - 1:
            indices.append(len(self.words) - 1)
        seg_copies = []
        prev_i = 0
        for i in indices:
            i += 1
            c = deepcopy(self)
            c.words = c.words[prev_i:i]
            c.update_seg_with_words()
            seg_copies.append(c)
            prev_i = i
        return seg_copies


class WhisperResult:

    def __init__(self, result: (str, dict)):
        if isinstance(result, str):
            self.path = result
            result = load_result(self.path)
        self.ori_dict = result.get('ori_dict') or result
        self.language = self.ori_dict.get('language')
        segments = self.ori_dict.get('segments')
        self.segments: List[Segment] = [Segment(**s) for s in segments] if segments else []
        self.remove_no_word_segments()

    def add_segments(self, index0: int, index1: int, inplace: bool = False, lock: bool = False):
        new_seg = self.segments[index0] + self.segments[index1]
        new_seg.update_seg_with_words()
        if lock and self.segments[index0].has_words:
            lock_idx = len(self.segments[index0].words)
            new_seg.words[lock_idx - 1].lock_right()
            if lock_idx < len(new_seg.words):
                new_seg.words[lock_idx].lock_left()
        if inplace:
            i0, i1 = sorted([index0, index1])
            self.segments[i0] = new_seg
            del self.segments[i1]
        return new_seg

    def rescale_time(self, scale_factor: float):
        for s in self.segments:
            s.rescale_time(scale_factor)

    def apply_min_dur(self, min_dur: float, inplace: bool = False):
        """
        Any duration is less than [min_dur] will be merged with adjacent word/segments.
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
                    result.add_segments(i-1, i, inplace=True)
                elif i == 0:
                    result.add_segments(i, i+1, inplace=True)
                else:
                    if result.segments[i-1].duration < result.segments[i-1].duration:
                        result.add_segments(i-1, i, inplace=True)
                    else:
                        result.add_segments(i, i+1, inplace=True)
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
            min_word_dur: float = 0.1,
            word_level: bool = True
    ):
        """

        Snap any start/end timestamps in silence parts of audio to the boundaries of the silence.

        Parameters
        ----------
        silent_starts: np.ndarray
            start timestamps of silent sections of audio

        silent_ends: np.ndarray
            start timestamps of silent sections of audio

        min_word_dur: float
            only allow changes on timestamps that results in word duration greater than this value. (default: 0.1)

        word_level: bool
            whether to settings to word level timestamps (default: False)

        """
        for s in self.segments:
            s.suppress_silence(silent_starts, silent_ends, min_word_dur, word_level=word_level)

        return self

    def reassign_ids(self):
        for i, s in enumerate(self.segments):
            s.id = i

    def remove_no_word_segments(self, ignore_ori=False):
        for i in reversed(range(len(self.segments))):
            if (ignore_ori or self.segments[i].ori_has_words) and not self.segments[i].has_words:
                del self.segments[i]
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

    def to_dict(self):
        return dict(text=self.text,
                    segments=self.segments_to_dicts(),
                    language=self.language,
                    ori_dict=self.ori_dict)

    def segments_to_dicts(self, reverse_text: Union[bool, tuple] = False):
        return [s.to_dict(reverse_text=reverse_text) for s in self.segments]

    def _split_segments(self, get_indices, args: list = None, *, lock: bool = False):
        if args is None:
            args = []
        no_words = False
        for i in reversed(range(0, len(self.segments))):
            no_words = not self.segments[i].has_words
            indices = get_indices(self.segments[i], *args)
            if indices:
                new_segments = self.segments[i].split(indices)
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

    def _merge_segments(self, indices: List[int],
                        *, max_words: int = None, max_chars: int = None, is_sum_max: bool = False, lock: bool = False):
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
            self.add_segments(i, i + 1, inplace=True, lock=lock)
        self.remove_no_word_segments()

    def split_by_gap(
            self,
            max_gap: float = 0.1,
            lock: bool = False
    ):
        """

        Split (in-place) any segment into multiple segments
        where the duration in between two  words > [max_gap]

        Parameters
        ----------
        max_gap: float
            The point between any two words greater than this value (seconds) will be split. (Default: 0.1)
        lock: bool
            Whether to prevent future splits/merges from altering changes made by this method. (Default: False)

        """
        self._split_segments(lambda x: x.get_gap_indices(max_gap), lock=lock)
        return self

    def merge_by_gap(
            self,
            min_gap: float = 0.1,
            max_words: int = None,
            max_chars: int = None,
            is_sum_max: bool = False,
            lock: bool = False
    ):
        """

        Merge (in-place) any pair of adjacent segments if the duration in between the pair <= [min_gap]

        Parameters
        ----------
        min_gap: float
            Any gaps below or equal to this value (seconds) will be merged. (Default: 0.1)
        max_words: int
            Maximum number of words allowed. (Default: None)
        max_chars: int
            Maximum number of characters allowed. (Default: None)
        is_sum_max: bool
            Whether [max_words] and [max_chars] is applied to the merged segment
            instead of the individual segments to be merged. (Default: False)
        lock: bool
            Whether to prevent future splits/merges from altering changes made by this method. (Default: False)

        """
        indices = self.get_gap_indices(min_gap)
        self._merge_segments(indices,
                             max_words=max_words, max_chars=max_chars, is_sum_max=is_sum_max, lock=lock)
        return self

    def split_by_punctuation(
            self,
            punctuation: Union[List[str], List[Tuple[str, str]], str],
            lock: bool = False
    ):
        """

        Split (in-place) any segment at words that starts/ends with specified punctuation(s)

        Parameters
        ----------
        punctuation: Union[List[str], List[Tuple[str, str]], str]
            Punctuation(s) to split segments by.
        lock: bool
            Whether to prevent future splits/merges from altering changes made by this method. (Default: False)

        """
        self._split_segments(lambda x: x.get_punctuation_indices(punctuation), lock=lock)
        return self

    def merge_by_punctuation(
            self, punctuation: Union[List[str], List[Tuple[str, str]], str],
            max_words: int = None,
            max_chars: int = None,
            is_sum_max: bool = False,
            lock: bool = False
    ):
        """

        Merge (in-place) any two segments that has specified punctuation(s) inbetween them

        Parameters
        ----------
        punctuation: Union[List[str], str]
            Punctuation(s) to merge segments by.
        max_words: int
            Maximum number of words allowed. (Default: None)
        max_chars: int
            Maximum number of characters allowed. (Default: None)
        is_sum_max: bool
            Whether [max_words] and [max_chars] is applied to the merged segment
            instead of all the individual segments to be merged. (Default: False)
        lock: bool
            Whether to prevent future splits/merges from altering changes made by this method. (Default: False)

        """
        indices = self.get_punctuation_indices(punctuation)
        self._merge_segments(indices,
                             max_words=max_words, max_chars=max_chars, is_sum_max=is_sum_max, lock=lock)
        return self

    def merge_all_segments(self):
        """
        Merge all segments into one segment.
        """
        self._merge_segments(list(range(len(self.segments) - 1)))
        return self

    def split_by_length(
            self,
            max_chars: int = None,
            max_words: int = None,
            force_len: bool = False,
            lock: bool = False
    ):
        """

        Split (in-place) any segment in segments that do not exceed the specified length

        Parameters
        ----------
        max_chars: int
            Maximum number of character allowed in each segment.
        max_words: int
            Maximum number of words allowed in each segment.
        force_len: bool
            Maintain a relatively constant length for each segment. (Default: False)
            This will ignore all previous non-locked segment boundaries (e.g. boundaries set by `regroup()`).
        lock: bool
            Whether to prevent future splits/merges from altering changes made by this method. (Default: False)

        """
        if force_len:
            self.merge_all_segments()
        self._split_segments(lambda x: x.get_length_indices(max_chars=max_chars, max_words=max_words), lock=lock)
        return self

    def regroup(self):
        """
        Regroup (in-place) all words into segments with more natural boundaries without locking.
        """
        return (
            self
            .split_by_punctuation([('.', ' '), '。', '?', '？', ',', '，'])
            .split_by_gap(.5)
            .merge_by_gap(.3, max_words=3)
            .split_by_punctuation([('.', ' '), '。', '?', '？'])
        )

    def find(self, pattern: str, word_level=True, flags=None) -> "WhisperResultMatches":
        """

        Find segments/words and timestamps with regular expression.

        Parameters
        ----------
        pattern: str
            RegEx pattern to search for.
        word_level: bool
            Whether to search at word-level
        flags:
            RegEx flags.

        Returns
        -------
        An instance of WhisperResultMatches class to allow for continuous chaining of this method.
        """
        return WhisperResultMatches(self).find(pattern, word_level=word_level, flags=flags)

    @property
    def text(self):
        return ''.join(s.text for s in self.segments)

    def __len__(self):
        return len(self.segments)

    def unlock_all_segments(self):
        for s in self.segments:
            s.unlock_all_words()
        return self

    def reset(self):
        self.language = self.ori_dict.get('language')
        segments = self.ori_dict.get('segments')
        self.segments: List[Segment] = [Segment(**s) for s in segments] if segments else []

    to_srt_vtt = result_to_srt_vtt
    to_ass = result_to_ass
    to_tsv = result_to_tsv
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
    RegEx matches for WhisperResults
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
        pattern: str
            RegEx pattern to search for.
        word_level: bool
            Whether to search at word-level
        flags:
            RegEx flags.

        Returns
        -------
        An instance of WhisperResultMatches class to allow for continuous chaining of this method.
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

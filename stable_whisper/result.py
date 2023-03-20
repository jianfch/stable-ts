import warnings
import numpy as np
from typing import Union, List
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

    def rescale_time(self, scale_factor: float):
        self.seek = round(self.seek * scale_factor, 3)
        self.start = round(self.start * scale_factor, 3)
        self.end = round(self.end * scale_factor, 3)
        if self.has_words:
            for w in self.words:
                w.rescale_time(scale_factor)
        self.update_seg_with_words()

    def to_dict(self):
        seg_dict = deepcopy(self.__dict__)
        seg_dict.pop('ori_has_words')
        if self.has_words:
            seg_dict['words'] = [w.to_dict() for w in self.words]
        elif self.ori_has_words:
            seg_dict['words'] = []
        else:
            seg_dict.pop('words')
        if self.id is None:
            seg_dict.pop('id')
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
        if not self.words:
            return []
        if max_gap is None:
            max_gap = 0
        indices = (self.get_gaps(True) > max_gap).nonzero()[0].tolist()
        return sorted(set(indices) - set(self.get_locked_indices()))

    def get_punctuation_indices(self, punctuation: Union[List[str], str]):  # for splitting
        if not self.words:
            return []
        if isinstance(punctuation, str):
            punctuation = [punctuation]
        indices = []
        for p in punctuation:
            indices.extend([i for i, w in enumerate(self.words[1:])
                            if w.word.startswith(p)])
            indices.extend([i for i, w in enumerate(self.words[:-1])
                            if w.word.endswith(p)])

        return sorted(set(indices) - set(self.get_locked_indices()))

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
        if min_gap is None:
            min_gap = 0
        indices = (self.get_gaps(True) <= min_gap).nonzero()[0].tolist()
        return sorted(set(indices) - set(self.get_locked_indices()))

    def get_punctuation_indices(self, punctuation: Union[List[str], str]):  # for merging
        if isinstance(punctuation, str):
            punctuation = [punctuation]
        indices = []
        for p in punctuation:
            indices.extend([i for i, s in enumerate(self.segments[1:])
                            if s.text.startswith(p)])
            indices.extend([i for i, s in enumerate(self.segments[:-1])
                            if s.text.endswith(p)])

        return sorted(set(indices) - set(self.get_locked_indices()))

    def all_words(self):
        return list(chain.from_iterable(s.words for s in self.segments))

    def to_dict(self):
        return dict(text=self.text,
                    segments=self.segments_to_dicts(),
                    language=self.language,
                    ori_dict=self.ori_dict)

    def segments_to_dicts(self):
        return [s.to_dict() for s in self.segments]

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

    def _merge_segments(self, get_indices, args: list = None,
                        *, max_words: int = None, max_chars: int = None, lock: bool = False):
        if args is None:
            args = []
        indices = get_indices(*args)
        for i in reversed(indices):
            seg = self.segments[i]
            if (
                    (
                            max_words and
                            seg.has_words and
                            (seg.word_count() + self.segments[i + 1].word_count() > max_words)
                    ) or
                    (
                            max_chars and
                            (seg.char_count() + self.segments[i + 1].char_count() > max_chars)
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
            Whether to prevent future splits/merges from altering changes made by this function. (Default: False)

        """
        self._split_segments(lambda x: x.get_gap_indices(max_gap), lock=lock)
        return self

    def merge_by_gap(
            self,
            min_gap: float = 0.1,
            max_words: int = None,
            max_chars: int = None,
            lock: bool = False
    ):
        """

        Merge (in-place) any pair of adjacent segments if the duration in between the pair <= [min_gap]

        Parameters
        ----------
        min_gap: float
            Any gaps below or equal to this value (seconds) will be merged. (Default: 0.1)
        max_words: int
            Maximum number of words the merged segment is allowed to have. (Default: None)
        max_chars: int
            Maximum number of characters the merged segment is allowed to have. (Default: None)
        lock: bool
            Whether to prevent future splits/merges from altering changes made by this function. (Default: False)

        """
        self._merge_segments(self.get_gap_indices, [min_gap],
                             max_words=max_words, max_chars=max_chars, lock=lock)
        return self

    def split_by_punctuation(
            self,
            punctuation: Union[List[str], str],
            lock: bool = False
    ):
        """

        Split (in-place) any segment at words that starts/ends with specified punctuation(s)

        Parameters
        ----------
        punctuation: Union[List[str], str]
            Punctuation(s) to split segments by.
        lock: bool
            Whether to prevent future splits/merges from altering changes made by this function. (Default: False)

        """
        self._split_segments(lambda x: x.get_punctuation_indices(punctuation), lock=lock)
        return self

    def merge_by_punctuation(
            self, punctuation: Union[List[str], str],
            max_words: int = None,
            max_chars: int = None,
            lock: bool = False
    ):
        """

        Merge (in-place) any two segments that has specified punctuation(s) inbetween them

        Parameters
        ----------
        punctuation: Union[List[str], str]
            Punctuation(s) to merge segments by.
        max_words: int
            Maximum number of words the merged segment is allowed to have. (Default: None)
        max_chars: int
            Maximum number of characters the merged segment is allowed to have. (Default: None)
        lock: bool
            Whether to prevent future splits/merges from altering changes made by this function. (Default: False)

        """
        self._merge_segments(self.get_punctuation_indices, [punctuation],
                             max_words=max_words, max_chars=max_chars, lock=lock)
        return self

    def regroup(self):
        """
        Split segments at periods or question marks,
        then split segments if the gap between two segments more than 0.5 seconds,
        then merge segments if the gap between two segments more than 0.15 seconds. (in-place)

        """
        return (
            self.split_by_punctuation(['.', '。', '?', '？'], True)
            .split_by_gap(.5)
            .merge_by_gap(.15)
            .unlock_all_segments()
        )

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
    save_as_json = save_as_json

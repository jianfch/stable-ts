import os
import sys
import torch
import stable_whisper


def get_load_method():
    if len(sys.argv) >= 2:
        return getattr(stable_whisper, sys.argv[1]), sys.argv[1]
    return stable_whisper.load_model, 'load_model'


def check_result(result, expected_text: str, test_name: str):
    assert result.text == expected_text

    timing_checked = False
    all_words = result.all_words()
    fail_count = 0
    for word in all_words:
        if word.start >= word.end:
            fail_count += 1
        if word.word.strip(" ,") == "americans":
            assert word.start <= 1.8, (word.start, test_name)
            assert word.end >= 1.8, (word.end, test_name)
            timing_checked = True
    fail_rate = fail_count / len(all_words)
    print(f'Fail Count: {fail_count} / {len(all_words)}  ({test_name})\n')
    assert fail_rate < 0.1, (fail_rate, fail_count, test_name)

    assert timing_checked, test_name


def test_align(model_names):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_path = os.path.join(os.path.dirname(__file__), "jfk.flac")
    load, load_name = get_load_method()
    models = [(load(name, device=device), f'{load_name}->{name}') for name in model_names]
    orig_result = models[0][0].transcribe(
        audio_path, language='en', temperature=0.0, word_timestamps=True
    )
    for word in orig_result.all_words():
        word.word = word.word.replace('Americans', 'americans')

    def single_test(m, meth: str, prep, extra_check, **kwargs):
        m, model_type = m
        meth = getattr(m, meth)
        test_name = f'{model_type} + {meth.__name__}(WhisperResult)'
        print(f'Start Test: {test_name}')
        result = meth(audio_path, orig_result, **kwargs)
        check_same_segment_text(orig_result, result)
        check_result(result, orig_result.text, test_name)

        test_name = f'{model_type} + {meth.__name__}(plain-text)'
        print(f'Start Test: {test_name}')
        result = meth(audio_path, prep(orig_result), language=orig_result.language)
        if extra_check:
            extra_check(orig_result, result)
        check_result(result, orig_result.text, test_name)

    def get_text(res):
        return res.text

    def get_segment_dicts(res):
        return [dict(start=s.start, end=s.end, text=s.text) for s in res]

    def check_same_segment_text(res0, res1):
        assert [s.text for s in res0] == [s.text for s in res1], 'mismatch segment text'

    for model in models:
        for method in ('align', 'align_words'):
            options = dict(original_split=True) if method == 'align' else {}
            preprocess = get_text if method == 'align' else get_segment_dicts
            check_seg = None if method == 'align' else check_same_segment_text
            single_test(model, method, preprocess, check_seg, **options)


def test():
    test_align(['tiny', 'tiny.en'])


if __name__ == '__main__':
    test()

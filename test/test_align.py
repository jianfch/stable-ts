import os
import torch
import stable_whisper


def check_result(result, expected_text: str, test_name: str):
    assert result.text == expected_text

    timing_checked = False
    for segment in result:
        for word in segment:
            assert word.start < word.end, (word.start, word.end, test_name)
            if word.word.strip(" ,") == "americans":
                assert word.start <= 1.8, (word.start, test_name)
                assert word.end >= 1.8, (word.end, test_name)
                timing_checked = True

    assert timing_checked, test_name


def test_align(model_names):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_path = os.path.join(os.path.dirname(__file__), "jfk.flac")
    models = [stable_whisper.load_model(name, device=device) for name in model_names]
    orig_result = models[0].transcribe(
        audio_path, language='en', temperature=0.0, word_timestamps=True
    )
    for word in orig_result.all_words():
        word.word = word.word.replace('Americans', 'americans')

    def single_test(m, meth: str, prep, extra_check, **kwargs):
        model_type = 'multilingual-model' if m.is_multilingual else 'en-model'
        meth = getattr(m, meth)
        test_name = f'{model_type} {meth.__name__}(WhisperResult)'
        try:
            result = meth(audio_path, orig_result, **kwargs)
            check_same_segment_text(orig_result, result)
        except Exception as e:
            raise Exception(f'failed test {test_name} -> {e.__class__.__name__}: {e}')
        check_result(result, orig_result.text, test_name)

        test_name = f'{model_type} {meth.__name__}(plain-text)'
        try:
            result = meth(audio_path, prep(orig_result), language=orig_result.language)
            if extra_check:
                extra_check(orig_result, result)
        except Exception as e:
            raise Exception(f'failed test {test_name} -> {e.__class__.__name__}: {e}')
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

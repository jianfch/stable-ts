import os
import sys
import torch
import stable_whisper


def get_load_method():
    if len(sys.argv) >= 2:
        return getattr(stable_whisper, sys.argv[1]), sys.argv[1]
    return stable_whisper.load_model, 'load_model'


def check_result(result, orig_result, test_name: str, expect_change: bool = True):

    timing_checked = False
    for segment in result:
        for word in segment:
            assert word.start < word.end
            if word.word.strip(" ,").lower() == "americans":
                assert word.start <= 1.8, (word.start, test_name)
                assert word.end >= 1.8, (word.end, test_name)
                timing_checked = True

    if expect_change:
        assert any(
            (w0.start, w0.end) != (w1.start, w1.end)
            for w0, w1 in zip(result.all_words(), orig_result.all_words())
        )

    assert timing_checked


def test_refine(model0_name: str, model1_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    load, load_name = get_load_method()
    model0 = load(model0_name, device=device)
    audio_path = os.path.join(os.path.dirname(__file__), "jfk.flac")

    language = "en" if model0_name.endswith(".en") else None
    orig_result = model0.transcribe(
        audio_path, language=language, temperature=0.0, word_timestamps=True
    )

    model1 = stable_whisper.load_model(model1_name, device=device)

    result = model1.refine(audio_path, orig_result, inplace=False)
    check_result(result, orig_result, load_name, True)


def test():
    test_refine('tiny.en', 'tiny')


if __name__ == '__main__':
    test()

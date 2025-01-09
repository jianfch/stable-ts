import os
import torch
import stable_whisper


def check_result(result, expected_text: str):
    assert result.text == expected_text

    timing_checked = False
    for segment in result:
        for word in segment:
            assert word.start < word.end
            if word.word.strip(" ,") == "americans":
                assert word.start <= 1.8, word.start
                assert word.end >= 1.8, word.end
                timing_checked = True

    assert timing_checked


def test_transcribe(model0_name: str, model1_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model0 = stable_whisper.load_model(model0_name, device=device)
    audio_path = os.path.join(os.path.dirname(__file__), "jfk.flac")

    language = "en" if model0_name.endswith(".en") else None
    orig_result = model0.transcribe(
        audio_path, language=language, temperature=0.0, word_timestamps=True
    )
    for word in orig_result.all_words():
        word.word = word.word.replace('Americans', 'americans')

    model1 = stable_whisper.load_model(model1_name, device=device)

    result = model1.align(audio_path, orig_result, original_split=True)
    assert [s.text for s in result] == [s.text for s in orig_result]
    check_result(result, orig_result.text)

    result = model1.align(audio_path, orig_result.text, language=orig_result.language)
    check_result(result, orig_result.text)


def test():
    test_transcribe('tiny', 'tiny.en')


if __name__ == '__main__':
    test()

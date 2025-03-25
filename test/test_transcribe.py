import os
import sys
import torch
import stable_whisper


def get_load_method():
    if len(sys.argv) >= 2:
        return getattr(stable_whisper, sys.argv[1]), sys.argv[1]
    return stable_whisper.load_model, 'load_model'


def check_result(result, test_name: str):
    assert result.language in ('en', 'english'), result.language

    transcription = result.text.lower()
    assert "my fellow americans" in transcription, test_name
    assert "your country" in transcription, test_name
    assert "do for you" in transcription, test_name

    timing_checked = False
    for segment in result:
        for word in segment:
            assert word.start < word.end
            if word.word.strip(" ,").lower() == "americans":
                assert word.start <= 1.8, (word.start, test_name)
                assert word.end >= 1.8, (word.end, test_name)
                timing_checked = True

    assert timing_checked


def test_transcribe(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    load, load_name = get_load_method()
    model = load(model_name, device=device)
    audio_path = os.path.join(os.path.dirname(__file__), "jfk.flac")

    language = "en" if model_name.endswith(".en") else None
    result = model.transcribe(
        audio_path, language=language, temperature=0.0, word_timestamps=True
    )
    check_result(result, f'{load_name}->transcribe')
    if not hasattr(model, 'transcribe_minimal'):
        return
    result = model.transcribe_minimal(
        audio_path, language=language, temperature=0.0, word_timestamps=True
    )
    check_result(result, f'{load_name}->transcribe_minimal')


def test():
    for model in ('tiny', 'tiny.en'):
        test_transcribe(model)


if __name__ == '__main__':
    test()

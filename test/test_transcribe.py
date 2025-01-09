import os
import torch
import stable_whisper


def check_result(result):
    assert result.language == "en"

    transcription = result.text.lower()
    assert "my fellow americans" in transcription
    assert "your country" in transcription
    assert "do for you" in transcription

    timing_checked = False
    for segment in result:
        for word in segment:
            assert word.start < word.end
            if word.word.strip(" ,").lower() == "americans":
                assert word.start <= 1.8, word.start
                assert word.end >= 1.8, word.end
                timing_checked = True

    assert timing_checked


def test_transcribe(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = stable_whisper.load_model(model_name, device=device)
    audio_path = os.path.join(os.path.dirname(__file__), "jfk.flac")

    language = "en" if model_name.endswith(".en") else None
    result = model.transcribe(
        audio_path, language=language, temperature=0.0, word_timestamps=True
    )
    check_result(result)
    result = model.transcribe_minimal(
        audio_path, language=language, temperature=0.0, word_timestamps=True
    )
    check_result(result)


def test():
    for model in ('tiny', 'tiny.en'):
        test_transcribe(model)


if __name__ == '__main__':
    test()

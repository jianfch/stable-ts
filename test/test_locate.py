import os
import torch
import stable_whisper


def test_locate(model_names):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_path = os.path.join(os.path.dirname(__file__), "jfk.flac")
    models = [stable_whisper.load_model(name, device=device) for name in model_names]
    for model in models:
        matches = model.locate(audio_path, 'americans', 'en', mode=0)
        assert len(matches), len(matches)
        words = [word.word.lower().strip(',').strip() for match in matches for word in match]
        assert 'americans' in words, words
        matches = model.locate(audio_path, 'americans', 'en', mode=1)
        assert len(matches), len(matches)
        any(['americans' in match['duration_window_text'].lower() for match in matches])


def test():
    test_locate(['tiny', 'tiny.en'])


if __name__ == '__main__':
    test()

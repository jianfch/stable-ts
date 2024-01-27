import warnings
from .cli import cli
from .original_whisper import transcribe_stable, transcribe_minimal, load_model, modify_model
from .faster_whisper import load_faster_whisper


__all__ = ['load_model', 'modify_model', 'load_faster_whisper']

warnings.filterwarnings('ignore', module='whisper', message='.*Triton.*', category=UserWarning)


if __name__ == '__main__':
    cli()

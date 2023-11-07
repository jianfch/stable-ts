import os
from setuptools import setup


def version() -> str:
    with open(os.path.join(os.path.dirname(__file__), 'stable_whisper/_version.py')) as f:
        return f.read().split('=')[-1].strip().strip('"').strip("'")


def read_me() -> str:
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()


setup(
    name="stable-ts",
    version=version(),
    description="Modifies OpenAI's Whisper to produce more reliable timestamps.",
    long_description=read_me(),
    long_description_content_type='text/markdown',
    python_requires=">=3.8",
    author="Jian",
    url="https://github.com/jianfch/stable-ts",
    license="MIT",
    packages=['stable_whisper'],
    install_requires=[
        "numpy",
        "torch",
        "torchaudio",
        "tqdm",
        "more-itertools",
        "transformers>=4.19.0",
        "ffmpeg-python==0.2.0",
        "openai-whisper==20231106"
    ],
    entry_points={
        "console_scripts": ["stable-ts=stable_whisper.whisper_word_level:cli"],
    },
    include_package_data=False
)

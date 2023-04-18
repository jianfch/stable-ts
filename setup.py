from setuptools import setup


def version() -> str:
    with open('./stable_whisper/_version.py') as f:
        return f.read().split('=')[-1].strip().strip('"').strip("'")


def read_me() -> str:
    with open('README.md', 'r') as f:
        return f.read()


setup(
    name="stable-ts",
    version=version(),
    description="Stabilizing timestamps of OpenAI's Whisper outputs down to word-level.",
    long_description=read_me(),
    long_description_content_type='text/markdown',
    python_requires=">=3.7",
    author="Jian",
    url="https://github.com/jianfch/stable-ts",
    license="MIT",
    packages=['stable_whisper'],
    install_requires=[
      "openai-whisper"
    ],
    include_package_data=False
)

# Stabilizing Timestamps for Whisper

## Description
This script modifies and adds more robust decoding logic on top of OpenAI's Whisper to produce more accurate segment-level timestamps and obtain to word-level timestamps with extra inference.

![image](https://user-images.githubusercontent.com/28970749/192950141-40ac8cbd-ccac-45da-b563-f8144d22c54e.png)

## TODO
- [ ] Add function to stabilize with multiple inferences
- [x] Add word timestamping (previously only token based)

## Setup
#### Option 1: Install Whisper+stable-ts (one line)
```
pip install git+https://github.com/jianfch/stable-ts.git
```
#### Option 2: Install Whisper (repo) and stable-ts (PyPI) separately
1. Install [Whisper](https://github.com/openai/whisper#setup)
2. Check if Whisper is installed correctly by running a quick test
3. Install stable-ts
```python
import whisper
model = whisper.load_model('base')
assert model.transcribe('audio.mp3').get('segments')
```
```commandline
pip install stable-ts
```

### Command-line usage
Transcribe audio then save result as JSON file.
```commandline
stable-ts audio.mp3 -o audio.json
```
Processing JSON file of the results into ASS.
```commandline
stable-ts audio.json -o audio.ass
```
Transcribe multiple audio files then process the results directly into SRT files.
```commandline
stable-ts audio1.mp3 audio2.mp3 audio3.mp3 -o audio1.srt audio2.srt audio3.srt
```
Show all available arguments and help.
```commandline
stable-ts -h
```

### Python usage
```python
import stable_whisper

model = stable_whisper.load_model('base')
# modified model should run just like the regular model but accepts additional parameters
results = model.transcribe('audio.mp3')

# word-level
stable_whisper.results_to_word_srt(results, 'audio.srt')
# sentence/phrase-level
stable_whisper.results_to_sentence_srt(results, 'audio.srt')
```

https://user-images.githubusercontent.com/28970749/202782436-0d56140b-5d52-4f33-b32b-317a19ad32ca.mp4

```python
# sentence/phrase-level & word-level
stable_whisper.results_to_sentence_word_ass(results, 'audio.ass')
```

https://user-images.githubusercontent.com/28970749/202782412-dfa027f8-7073-4023-8ce5-285a2c26c03f.mp4

#### Additional Info
* Although timestamps are chronological, they can still very inaccurate depending on the model, audio, and parameters.
* To produce production ready word-level results, the model needs to be fine-tuned with high quality dataset of audio with word-level timestamp.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments
Includes slight modification of the original work: [Whisper](https://github.com/openai/whisper)

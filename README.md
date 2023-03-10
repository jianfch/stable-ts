# Stabilizing Timestamps for Whisper

## Description
This script modifies and adds more robust decoding logic on top of OpenAI's Whisper to produce more accurate segment-level timestamps and obtain to word-level timestamps without extra inference.

![image](https://user-images.githubusercontent.com/28970749/218944014-b915af81-1cf5-4522-a823-e0f476fcc550.png)


## Update:
The official [Whisper]() repo introduced word-level timestamps in a recent [commit](https://github.com/openai/whisper/commit/500d0fe9668fae5fe2af2b6a3c4950f8a29aa145) which produces more reliable timestamps than method used in this script. 
This script has not been updated to utilize this new version of Whisper yet. It will be updated for next release, version 2.0.0.  

## Setup
```
pip install -U stable-ts
```

To install the lastest commit:
```
pip install -U git+https://github.com/jianfch/stable-ts.git
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
```

https://user-images.githubusercontent.com/28970749/218942894-cb0b91df-1c14-4d2f-9793-d1c8ef20e711.mp4


```python
# the above uses default settings on version 1.1 with large model
# sentence/phrase-level
stable_whisper.results_to_sentence_srt(results, 'audio.srt')
```

https://user-images.githubusercontent.com/28970749/218942942-060610e4-4c96-454d-b00a-8c9a41f4e7de.mp4


```python
# the above uses default settings on version 1.1 with large model
# sentence/phrase-level & word-level
stable_whisper.results_to_sentence_word_ass(results, 'audio.ass')
```
#### Additional Info
* Although timestamps are chronological, they can still very inaccurate depending on the model, audio, and parameters.
* To produce production ready word-level results, the model needs to be fine-tuned with high quality dataset of audio with word-level timestamp.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments
Includes slight modification of the original work: [Whisper](https://github.com/openai/whisper)

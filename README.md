# Stabilizing Timestamps for Whisper

## Description
This script modifies methods of Whisper's model to gain access to the predicted timestamp tokens of each word (token) without needing additional inference. It also stabilizes the timestamps down to the word (token) level to ensure chronology. Additionally, it can suppress gaps in speech for more accurate timestamps.

![image](https://user-images.githubusercontent.com/28970749/192950141-40ac8cbd-ccac-45da-b563-f8144d22c54e.png)

## TODO
- [ ] Add function to stabilize with multiple inferences
- [x] Add word timestamping (previously only token based)

## Dependency
* [Whisper](https://github.com/openai/whisper)

## Setup
#### Option 1: Install Whisper+stable-ts (one line)
```
pip install git+https://github.com/jianfch/stable-ts.git
```
#### Option 2: Install Whisper (repo) and stable-ts (PyPI) separately
1. Install [Whisper](https://github.com/openai/whisper#setup)
2. Check if Whisper is installed correctly by running a quick test
```python
import whisper
model = whisper.load_model('base')
assert model.transcribe('audio.mp3').get('segments')
```
3. Install stable-ts
```commandline
pip install stable-ts
```

### Executing script
```python
from stable_whisper import load_model

model = load_model('base')
# modified model should run just like the regular model but with additional hyperparameters and extra data in results
results = model.transcribe('audio.mp3')
stab_segments = results['segments']
first_segment_word_timestamps = stab_segments[0]['whole_word_timestamps']

# or to get token timestamps that adhere more to the top prediction
from stable_whisper import stabilize_timestamps
stab_segments = stabilize_timestamps(results, top_focus=True)
```

### Generate .srt with stable timestamps
```python
# word-level 
from stable_whisper import results_to_word_srt
# after you get results from modified model
# this treats a word timestamp as end time of the word
# and combines words if their timestamps overlap
results_to_word_srt(results, 'audio.srt')  # combine_compound=True will merge words with no prepended space
```
```python
# sentence/phrase-level
from stable_whisper import results_to_sentence_srt
# after you get results from modified model
results_to_sentence_srt(results, 'audio.srt')
# below is from large model default settings
```

https://user-images.githubusercontent.com/28970749/202782436-0d56140b-5d52-4f33-b32b-317a19ad32ca.mp4


```python
# sentence/phrase-level & word-level
from stable_whisper import results_to_sentence_word_ass
# after you get results from modified model
results_to_sentence_word_ass(results, 'audio.ass')
# below is from large model default settings
```

https://user-images.githubusercontent.com/28970749/202782412-dfa027f8-7073-4023-8ce5-285a2c26c03f.mp4

#### Additional Info
* Since the sentence/segment-level timestamps are predicted directly, they are always more accurate and precise than word/token-level timestamps.
* Although timestamps are chronological, they can still be off sync depending on the model and audio.
* The `unstable_word_timestamps` are left in the results, so you can possibly find better way to utilize them.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments
Includes slight modification of the original work: [Whisper](https://github.com/openai/whisper)

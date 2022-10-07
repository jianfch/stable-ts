# Stabilizing Timestamps for Whisper

## Description
This script modifies methods of Whisper's model to gain access to the predicted timestamp tokens of each word (token) without needing additional inference. It also stabilizes the timestamps down to the word (token) level to ensure chronology.

![image](https://user-images.githubusercontent.com/28970749/192950141-40ac8cbd-ccac-45da-b563-f8144d22c54e.png)

## TODO
- [ ] Add function to stabilize with multiple inferences
- [x] Add word timestamping (previously only token based)

## Dependency
* [Whisper](https://github.com/openai/whisper)

## Setup 
1. Install [Whisper](https://github.com/openai/whisper#setup)
2. Check if Whisper is installed correctly by running a quick test
```python
import whisper
model = whisper.load_model('base', 'cuda')
assert model.transcribe('audio.mp3').get('segments')
```
3. Clone repo
```commandline
git clone https://github.com/jianfch/stable-ts.git
cd stable-ts
```

### Executing script
```python
import whisper
from stable_whisper import modify_model

model = whisper.load_model('base', 'cuda')
modify_model(model)
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
results_to_word_srt(results, 'audio.srt')  #combine_compound=True if compound words are separate
```
```python
# sentence-level
from stable_whisper import results_to_sentence_srt
# after you get results from modified model
results_to_sentence_srt(results, 'audio.srt')
```

#### Additional Info
* Since the sentence/segment-level timestamps are predicted directly, they are always more accurate and precise than word/token-level timestamps.
* Although timestamps are chronological, they can still be off sync depending on the model and audio.
* The `unstable_word_timestamps` are left in the results, so you can possibly find better way to utilize them.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments
Slight modification of the original work:
* [Whisper](https://github.com/openai/whisper)

# Stabilizing Timestamps for Whisper

## Description
This script modifies methods of Whisper's model to gain access to the predicted timestamp tokens of each word(token) without needing additional inference. It also stabilizes the timestamps down to the word(token) level to ensure chronology.

![image](https://user-images.githubusercontent.com/28970749/192950141-40ac8cbd-ccac-45da-b563-f8144d22c54e.png)

## TODO
- [ ] Add function to stabilize with multiple inferences
- [ ] Add word timestamping (it is only token based right now)

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
first_segment_token_timestamps = stab_segment[0]['word_timestamps']

# or to get token timestamps that adhere more to the top prediction
from stable_whisper import stabilize_timestamps
stab_segments = stabilize_timestamps(result, top_focus=True)
```

### Generate .srt with stable timestamps
```python
# token-level 
from stable_whipser import results_to_token_srt
# after you get result from modified model
# this treats the token timestamps as end time of the tokens
results_to_token_srt(result, 'audio.srt')  # will combine tokens if their timestamps overlap
```
```python
# sentence-level
from stable_whipser import results_to_sentence_srt
# after you get result from modified model
results_to_sentence_srt(result, 'audio.srt')
```

#### Additional Info
* The "word" timestamps are actually token timestamps. Since token:word is not always 1:1 (varies by language), you may need to do some additional processing to get individual word timings.
* The timing can still be off sync depending on the model and audio.
* Haven't done any extensive testing to conclude how to interpret the word timestamps. Whether it is beginning/middle/end of the word(token), it's up to you decide how to use the timestamps.
* The `unstable_word_timestamps` are left in the results, so you can possibly find better way to utilize them.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments
Slight modification of the original work:
* [Whisper](https://github.com/openai/whisper)

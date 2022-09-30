# Stabilizing Timestamps for Whisper

## Description
This script modifies methods of Whisper's model to gain access to the predicted timestamp tokens of each word(token) without needing addition inference. It also stabilizes the timestamps down to the word(token) level to ensure chronology.

![image](https://user-images.githubusercontent.com/28970749/192950141-40ac8cbd-ccac-45da-b563-f8144d22c54e.png)

## TODO
- [ ] Add function to stabilize with multiple inferences
- [ ] Add word timestamping (it is only token based right now)

## Dependency
* [Whisper](https://github.com/openai/whisper)

## Setup 
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
results = model.transcribe('audio.mp3')
word_timestamps = results['segments']['word_timestamps']

# gather 7 timestamps tokens per word from prediction instead of the default 5
result = model.transcribe('audio.mp3', ts_num=7) #stab=false to disable stabilization
```

### Generate token-level .srt
```python
from stable_whipser import results_to_srt

# after you get result from model
# this treats the token timestamps as end time of the tokens
results_to_srt(result, 'audio.srt', word_level=True)  # will combine tokens if their timestamps overlap
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

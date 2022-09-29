# Stabilizing Timestamps for Whisper

## Description
This script modifies methods of Whisper's model to gain access to the predicted timestamp tokens of each word without needing addition inference. It also stabilizes the timestamps down to the word level to ensure chronology.

![image](https://user-images.githubusercontent.com/28970749/192950141-40ac8cbd-ccac-45da-b563-f8144d22c54e.png)

## TODO
- [ ] Add ability to stabilize with multiple inferences

## Dependency
* [Whisper](https://github.com/openai/whisper)

### Executing script
```python
import whisper
from stable_whisper import modify_model

model = whisper.load_model('base', 'cuda')
modify_model(model)
results = model.transcribe('audio.mp3')
word_timestamps = results['segments']['word_timestamps']

#gather 7 timestamps tokens per word from prediction instead of the default 5
possibly_results = model.transcribe('audio.mp3', ts_num=7) #stab=false to disable stabilization
```

#### Additional Info
* The timing can still be off sync depending on the model and audio.
* Haven't done any extensive testing to conclude how to interpret the word timestamps. Whether it is beginning/middle/end of the word, it's up to you decide how to use the timestamps.
* The `unstable_word_timestamps` are left in the results, so you can possibly find better way to utilize them.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments
Slight modification of the original work:
* [Whisper](https://github.com/openai/whisper)

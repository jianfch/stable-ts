# Stabilizing Timestamps for Whisper

This script modifies [OpenAI's Whisper](https://github.com/openai/whisper) to produce more reliable timestamps.

![image](./demo/jfk.PNG)


./demo/jfk.mp4


### What's new in 2.0.0 ?
- updated to use Whisper's more reliable word-level timestamps method. 
- the more reliable word timestamps allows regrouping segments word by word.
- can now suppress silence with [Silero VAD](https://github.com/snakers4/silero-vad) (requires PyTorch 1.2.0+)
- non-VAD silencing suppress is also more robust 


./demo/a.mp4


### Features
- more control over the timestamps than default Whisper
- supports direct preprocessing with [Demucs](https://github.com/facebookresearch/demucs) to isolate voice
- support dynamic quantization to decrease memory usage for inference on CPU
- lower memory usage than default Whisper when transcribing very long input audio tracks

## Setup
```
pip install -U stable-ts
```

To install the lastest commit:
```
pip install -U git+https://github.com/jianfch/stable-ts.git
```

### Command-line usage
Transcribe audio then save result as JSON file which contains the original inference results. 
This allows results to be reprocessed different without having to redo inference.
Change `audio.json` to `audio.srt` to process it directly into SRT.
```commandline
stable-ts audio.mp3 -o audio.json
```
Processing JSON file of the results into SRT.
```commandline
stable-ts audio.json -o audio.srt
```
Transcribe multiple audio files then process the results directly into SRT files.
```commandline
stable-ts audio1.mp3 audio2.mp3 audio3.mp3 -o audio1.srt audio2.srt audio3.srt
```

### Python usage
```python
import stable_whisper

model = stable_whisper.load_model('base')
# modified model should run just like the regular model but accepts additional parameters
result = model.transcribe('audio.mp3')
# srt/vtt
result.to_srt_vtt('audio.srt')
# ass
result.to_ass('audio.ass')
# json
result.save_as_json('audio.json')
```

### Regrouping Words
Stable-ts has a preset for regrouping word into different segments. This preset is enabled by `regroup=True`.
But are other built-in regrouping methods that allow you to customize the regrouping logic. 
This preset is just a predefined a combination of those methods.


./demo/xata.mp4


```python
result0 = model.transcribe('audio.mp3', regroup=True) # regroup is True by default
# regroup=True is same as below
result1 = model.transcribe('audio.mp3', regroup=False)
result1.split_by_punctuation(['.', '。', '?', '？'], True).split_by_gap(.5).merge_by_gap(.15).unlock_all_segments()
# result0 == result1
```

### Visualizing Suppression
- Requirement: [Pillow](https://github.com/python-pillow/Pillow) or [opencv-python](https://github.com/opencv/opencv-python)
#### Non-VAD Suppression
![image](./demo/novad.png)
```python
import stable_whisper
# regions on the waveform colored red is where it will be likely be suppressed and marked to as silent
# [q_levels=20] and [k_size=5] are defaults for non-VAD.
stable_whisper.visualize_suppression('audio.mp3', 'image.png', q_levels=20, k_size = 5) 
```
#### VAD Suppression
![image](./demo/vad.png)
```python
# [vad_threshold=0.35] is defaults for VAD.
stable_whisper.visualize_suppression('audio.mp3', 'image.png', vad=True, vad_threshold=0.35)
```

### Encode Comparison 
```python
import stable_whisper

stable_whisper.encode_video_comparison(
    'audio.mp3', 
    ['audio_sub1.srt', 'audio_sub2.srt'], 
    output_videopath='audio.mp4', 
    labels=['Example 1', 'Example 2']
)
```



## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments
Includes slight modification of the original work: [Whisper](https://github.com/openai/whisper)

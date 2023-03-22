# Stabilizing Timestamps for Whisper

This script modifies [OpenAI's Whisper](https://github.com/openai/whisper) to produce more reliable timestamps.

![jfk](https://user-images.githubusercontent.com/28970749/225825244-f3df9607-91ab-4011-a333-7e3ae94da08f.PNG)

https://user-images.githubusercontent.com/28970749/225825286-cdb14d70-566f-454b-a2b3-b61b4b3e09c9.mp4

### What's new in 2.0.0 ?
- updated to use Whisper's more reliable word-level timestamps method. 
- the more reliable word timestamps allows regrouping all words into segments with more natural boundaries.
- can now suppress silence with [Silero VAD](https://github.com/snakers4/silero-vad) (requires PyTorch 1.2.0+)
- non-VAD silence suppression is also more robust 
- see [Quick 1.X → 2.X Guide](#quick-1x--2x-guide)

https://user-images.githubusercontent.com/28970749/225826345-ef7115db-51e4-4b23-aedd-069389b8ae43.mp4

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

### Tips
- for reliable segment timestamps, do not disable word timestamps with `word_timestamps=False` because word timestamps is also used to correct segment timestamps
- use `demucs=True` and `vad=True` for music
- if audio is not transcribing properly compared to whisper, try `mel_first=True` at cost of more memory usuage for long audio tracks

### Quick 1.X → 2.X Guide
- `results_to_sentence_srt(result, 'audio.srt')` → `result.to_srt_vtt('audio.srt', word_level=False)` 
- `results_to_word_srt(result, 'audio.srt')` → `result.to_srt_vtt('output.srt', segment_level=False)`
- `results_to_sentence_word_ass(result, 'audio.srt')` → `result.to_ass('output.ass')`
- there's no need to stabilize segment after inference because they're already stabilized during inference
- `transcribe()` returns a `WhisperResult` object which can be converted to `dict` with `.to_dict()`. e.g `result.to_dict()`

### Regrouping Words
Stable-ts has a preset for regrouping words into different into segments with more natural boundaries. 
This preset is enabled by `regroup=True`. But there are other built-in regrouping methods that allow you to customize the regrouping logic. 
This preset is just a predefined a combination of those methods.

https://user-images.githubusercontent.com/28970749/226504985-3d087539-cfa4-46d1-8eb5-7083f235b429.mp4

```python
result0 = model.transcribe('audio.mp3', regroup=True) # regroup is True by default
# regroup=True is same as below
result1 = model.transcribe('audio.mp3', regroup=False)
(
    result1
    .split_by_punctuation([('.', ' '), '。', '?', '？', ',', '，'])
    .split_by_gap(.5)
    .merge_by_gap(.15, max_words=3)
    .split_by_punctuation([('.', ' '), '。', '?', '？'])
)
# result0 == result1
```

### Visualizing Suppression
- Requirement: [Pillow](https://github.com/python-pillow/Pillow) or [opencv-python](https://github.com/opencv/opencv-python)
#### Non-VAD Suppression

![novad](https://user-images.githubusercontent.com/28970749/225825408-aca63dbf-9571-40be-b399-1259d98f93be.png)

```python
import stable_whisper
# regions on the waveform colored red is where it will be likely be suppressed and marked to as silent
# [q_levels=20] and [k_size=5] are defaults for non-VAD.
stable_whisper.visualize_suppression('audio.mp3', 'image.png', q_levels=20, k_size = 5) 
```
#### VAD Suppression

![vad](https://user-images.githubusercontent.com/28970749/225825446-980924a5-7485-41e1-b0d9-c9b069d605f2.png)

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

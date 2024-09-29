# CHANGELOG

## 2.17.4
* deprecated `vad_onnx` ([b309530](https://github.com/jianfch/stable-ts/commit/b30953022d5f5d80e2ad69735b6b1507c743caad))
* added optional dependencies for Faster Whisper and Hugging Face ([c541169](https://github.com/jianfch/stable-ts/commit/c541169ea1ff98c128a26b9eaf6c945d0614d2ad))
* added `nonspeech_skip` ([888181f](https://github.com/jianfch/stable-ts/commit/888181f139e2ad0b17b89519a45f6f9b5eea0a30))
* fixed [#393](https://github.com/jianfch/stable-ts/issues/393) ([1ee47ce](https://github.com/jianfch/stable-ts/commit/1ee47ce027bb6060d66ccea8637cb6a86a0215c2))
* fixed `stabilization.utils.mask2timing()` to handle edge cases ([e0e7183](https://github.com/jianfch/stable-ts/commit/e0e718365ac3049e3559d2764c698b2ca6818ae2))
* fixed `suppress_silence=False` performing unnecessary compute when `vad=True` ([888181f](https://github.com/jianfch/stable-ts/commit/888181f139e2ad0b17b89519a45f6f9b5eea0a30))
* fixed typos in docstrings ([e0e7183](https://github.com/jianfch/stable-ts/commit/e0e718365ac3049e3559d2764c698b2ca6818ae2))
* updated `refine()` docstring in `README` ([3bc76b9](https://github.com/jianfch/stable-ts/commit/3bc76b98d52bdeae861e29d0e30c972a35392617))
* updated `vad` to accept a `dict` of keyword arguments for loading VAD ([b309530](https://github.com/jianfch/stable-ts/commit/b30953022d5f5d80e2ad69735b6b1507c743caad))

## 2.17.3
* added `pad()` to `result.WhisperResult` ([689fe5e](https://github.com/jianfch/stable-ts/commit/689fe5eaa06b50712dcf46b321cb46d0dd6e0f3c))
* added `newline` to `merge_by_gap()` and `merge_by_punctuation()` ([689fe5e](https://github.com/jianfch/stable-ts/commit/689fe5eaa06b50712dcf46b321cb46d0dd6e0f3c))
* fixed `verbose` for `adjust_by_silence()` ([f53f2ee](https://github.com/jianfch/stable-ts/commit/f53f2ee3d1401cdbdbb9979efe7dc35fce46c2d9))
* fixed adjustment progress bar in `non_whisper.transcribe_any()` ([48d70a8](https://github.com/jianfch/stable-ts/commit/48d70a89fd8715ab83da19a86fc4a4ade8b16be5))
* fixed error from using `tag`/`--tag` when output format is VTT and `word_level=True` ([3997ef1](https://github.com/jianfch/stable-ts/commit/3997ef124382b39a9a281a5090d1ed8334b513eb))
* fixed segment merging methods not working when the result contains only segment-level timestamps ([689fe5e](https://github.com/jianfch/stable-ts/commit/689fe5eaa06b50712dcf46b321cb46d0dd6e0f3c))
* updated `merge_by_gap()` and `merge_by_punctuation()` docstrings with `newline` ([3ab74e7](https://github.com/jianfch/stable-ts/commit/3ab74e7c969adf14abe51244a35a9ab7dd7a1973))

## 2.17.2
* changed SRT to start from index 1 ([9f8db52](https://github.com/jianfch/stable-ts/commit/9f8db520f0e611d010398322debcbb0b199aed13))
* changed `reset()` to be consistent for results produces by all `transcribe()` variants ([864b76c](https://github.com/jianfch/stable-ts/commit/864b76c1d0b8946638dfda6fb6ed577958c5c578))
* fixed [#357](https://github.com/jianfch/stable-ts/issues/357) ([98923ea](https://github.com/jianfch/stable-ts/commit/98923ea5ec7388ae9442923fcda3f4089003cec7))
* fixed `refine()` not working when `verbose` is not `True` ([864b76c](https://github.com/jianfch/stable-ts/commit/864b76c1d0b8946638dfda6fb6ed577958c5c578))
* fixed progress bar warning for `refine()` ([864b76c](https://github.com/jianfch/stable-ts/commit/864b76c1d0b8946638dfda6fb6ed577958c5c578))

## 2.17.1
* fixed [#353](https://github.com/jianfch/stable-ts/issues/353) ([66f8d13](https://github.com/jianfch/stable-ts/commit/66f8d1355d976bbff45b6f2aeda623099d23b5be))
* fixed `align()` error when audio segment contains no detectable nonspeech/silent sections ([6d9a1ef](https://github.com/jianfch/stable-ts/commit/6d9a1efcbbd4dab2cdac2727be9d6f7fcc5dcf06))
* fixed `gap_padding` causing unpredictable gaps or delays in the final timestamps for `align()` ([6d9a1ef](https://github.com/jianfch/stable-ts/commit/6d9a1efcbbd4dab2cdac2727be9d6f7fcc5dcf06))
* updated `align()` ([6d9a1ef](https://github.com/jianfch/stable-ts/commit/6d9a1efcbbd4dab2cdac2727be9d6f7fcc5dcf06))

## 2.17.0
* added `min_silence_dur` to `align()` and all variants of `transcribe()` ([e2f9458](https://github.com/jianfch/stable-ts/commit/e2f9458e540e1d82c7940532bae8813064af9f74))
* added `pad_or_trim()` to `whisper_compatibility` ([c4d42f2](https://github.com/jianfch/stable-ts/commit/c4d42f28137bec5f6e2dab19465ac30fc2222151))
* changed `align()` to ignore  compatibility issues for Fast-Whisper models ([c4d42f2](https://github.com/jianfch/stable-ts/commit/c4d42f28137bec5f6e2dab19465ac30fc2222151))
* changed `align()` to prioritize new timestamps within rounding error ([5ca7ca5](https://github.com/jianfch/stable-ts/commit/5ca7ca5c9e988b74d94d0c1699cd6da11e3273af))
* changed `align()` to prioritize timestamps that least overlap nonspeech timings ([e2f9458](https://github.com/jianfch/stable-ts/commit/e2f9458e540e1d82c7940532bae8813064af9f74))
* changed silence suppression to be less aggressive ([e2f9458](https://github.com/jianfch/stable-ts/commit/e2f9458e540e1d82c7940532bae8813064af9f74))
* changed silence suppression to treat nonspeech sections that overlap a word as individual sections ([5ca7ca5](https://github.com/jianfch/stable-ts/commit/5ca7ca5c9e988b74d94d0c1699cd6da11e3273af))
* dropped Whisper dependency for `stable-ts-whisperless` ([c4d42f2](https://github.com/jianfch/stable-ts/commit/c4d42f28137bec5f6e2dab19465ac30fc2222151))
* fixed `result.WordTIming.suppress_silence()` by undoing changes in [e2f9458](https://github.com/jianfch/stable-ts/commit/e2f9458e540e1d82c7940532bae8813064af9f74) ([0546d76](https://github.com/jianfch/stable-ts/commit/0546d765f6f5167c8cf5c01510d7bd8042cab013))
* fixed discrepancy between `text` and output for `align()` ([e2f9458](https://github.com/jianfch/stable-ts/commit/e2f9458e540e1d82c7940532bae8813064af9f74))
* changed default of `align()` to `presplit=False` on faster-whisper models ([850a19f](https://github.com/jianfch/stable-ts/commit/850a19fd5d9488836f453e1bb64f95687b418e0e))
* updated `README.md` with setup instructions for `stable-ts-whisperless` ([c4d42f2](https://github.com/jianfch/stable-ts/commit/c4d42f28137bec5f6e2dab19465ac30fc2222151))
* updated `use_word_position=True` to also take into account the index of each word ([5ca7ca5](https://github.com/jianfch/stable-ts/commit/5ca7ca5c9e988b74d94d0c1699cd6da11e3273af))

## 2.16.0
* deprecated `suppress_attention` ([5513609](https://github.com/jianfch/stable-ts/commit/5513609b33935192cc54432bd224abd04b535965))
* deprecated `ts_num` and `ts_noise` ([5513609](https://github.com/jianfch/stable-ts/commit/5513609b33935192cc54432bd224abd04b535965))
* added [noisereduce](https://github.com/timsainb/noisereduce) as a supported denoisers ([03bb83b](https://github.com/jianfch/stable-ts/commit/03bb83beb30a4763884efc0bfd5b5bd17ea3c0df))
* added `engine` to `load_model()` ([5513609](https://github.com/jianfch/stable-ts/commit/5513609b33935192cc54432bd224abd04b535965))
* added `extra_models`, to `align()` and `transcribe()` ([5513609](https://github.com/jianfch/stable-ts/commit/5513609b33935192cc54432bd224abd04b535965))
* added `presplit` and `gap_padding` to `align()` ([5513609](https://github.com/jianfch/stable-ts/commit/5513609b33935192cc54432bd224abd04b535965))
* fixed docstring of `adjust_by_silence()` ([5513609](https://github.com/jianfch/stable-ts/commit/5513609b33935192cc54432bd224abd04b535965))
* fixed `dfnet` denoiser model to use specified `device` ([5513609](https://github.com/jianfch/stable-ts/commit/5513609b33935192cc54432bd224abd04b535965))
* fixed error from `progress=True` when `denoiser='noisereduce'`  ([5513609](https://github.com/jianfch/stable-ts/commit/5513609b33935192cc54432bd224abd04b535965))
* fixed incorrect titles when downloading audio with yt-dlp([5513609](https://github.com/jianfch/stable-ts/commit/5513609b33935192cc54432bd224abd04b535965))
* changed `'demucs'` and `'dfnet'` denoisers to denoise in 2 channels when `stream=False`  ([5513609](https://github.com/jianfch/stable-ts/commit/5513609b33935192cc54432bd224abd04b535965))
* improved word timing by making `gap_padding` more effective ([5513609](https://github.com/jianfch/stable-ts/commit/5513609b33935192cc54432bd224abd04b535965))

## 2.15.11
* fixed inaccurate progress bar in `result.WhisperResult.suppress_silence()` ([ad013d7](https://github.com/jianfch/stable-ts/commit/ad013d7f80de2b090ccfe967eb7801c8094cdf8a))
* replaced `update_all_segs_with_words()` in the `refine()` with `reassign_ids()` ([ad013d7](https://github.com/jianfch/stable-ts/commit/ad013d7f80de2b090ccfe967eb7801c8094cdf8a))
* updated `--align` to treat the argument as plain-text if the argument starts with `'text='` ([ad013d7](https://github.com/jianfch/stable-ts/commit/ad013d7f80de2b090ccfe967eb7801c8094cdf8a))

## 2.15.10
* added `--persist` / `-p` to CLI ([177bcc4](https://github.com/jianfch/stable-ts/commit/177bcc4ee34d6fa3753ec6de2362e1567d95b52b))
* added `suppress_attention` to `transcribe()` and `align()` for original Whisper ([177bcc4](https://github.com/jianfch/stable-ts/commit/177bcc4ee34d6fa3753ec6de2362e1567d95b52b))
* fixed `align()` failing to predict nonspeech timings after skipping a nonspeech section ([424f484](https://github.com/jianfch/stable-ts/commit/424f4842d91c0fceb66ac96aba43c41cb30275b3))
* fixed typo ([#324](https://github.com/jianfch/stable-ts/pulls/324)) ([dbee5c5](https://github.com/jianfch/stable-ts/commit/dbee5c5a839be5dc6eb50683d74efcd3099006e2))

## 2.15.9
* changed `WhisperResult` to allow initialization without data ([00ad4b4](https://github.com/jianfch/stable-ts/commit/00ad4b45d314eadedc59b2ede7ab034ef14a5131))
* fixed `Segment.copy()` failing to initialize `WordTiming` when `new_words=None` and `copy_words=False` ([00ad4b4](https://github.com/jianfch/stable-ts/commit/00ad4b45d314eadedc59b2ede7ab034ef14a5131))
* fixed `WhisperResult.duration` to return `0.0` if result contains no segments ([00ad4b4](https://github.com/jianfch/stable-ts/commit/00ad4b45d314eadedc59b2ede7ab034ef14a5131))
* fixed `WhisperResult.has_words` to return `False` if result contains no segments ([00ad4b4](https://github.com/jianfch/stable-ts/commit/00ad4b45d314eadedc59b2ede7ab034ef14a5131))

## 2.15.8
* fixed `Whisper.fill_in_gaps()` ([cbbad76](https://github.com/jianfch/stable-ts/commit/cbbad765891e261a3c90eba463f44d0dc54e5433))
* removed `end` >= `start` requirement for `Segment` ([cbbad76](https://github.com/jianfch/stable-ts/commit/cbbad765891e261a3c90eba463f44d0dc54e5433))
* updated warning message for out of order timestamps ([cbbad76](https://github.com/jianfch/stable-ts/commit/cbbad765891e261a3c90eba463f44d0dc54e5433))

## 2.15.7
* deprecated `Segment.update_seg_with_words()` and `WhisperResult.update_all_segs_with_words()` ([ff89e53](https://github.com/jianfch/stable-ts/commit/ff89e53ab636909b2c184aff28e5e56aeebbca3a))
* changed `start`, `end`, `text`, `tokens` of `Segment` to properties ([ff89e53](https://github.com/jianfch/stable-ts/commit/ff89e53ab636909b2c184aff28e5e56aeebbca3a))
* deprecated and replace `WordTiming.round_all_timestamps()` with `round_ts=True` at initialization ([ff89e53](https://github.com/jianfch/stable-ts/commit/ff89e53ab636909b2c184aff28e5e56aeebbca3a))
* added progress bar for timestamps adjustments ([ff89e53](https://github.com/jianfch/stable-ts/commit/ff89e53ab636909b2c184aff28e5e56aeebbca3a))
* speed up splitting and merging of segments ([ff89e53](https://github.com/jianfch/stable-ts/commit/ff89e53ab636909b2c184aff28e5e56aeebbca3a))
* removed redundant parts of the default regrouping algorithm ([ff89e53](https://github.com/jianfch/stable-ts/commit/ff89e53ab636909b2c184aff28e5e56aeebbca3a))

## 2.15.6
* added `pipeline` to `stable_whisper.load_hf_whisper()` ([c356491](https://github.com/jianfch/stable-ts/commit/c3564915792f17eb5905e56c5ce78ce482a9f89b))
* changed `language`, `task`, `batch_size` to optional parameters for the `WhisperHF.transcribe()` ([c356491](https://github.com/jianfch/stable-ts/commit/c3564915792f17eb5905e56c5ce78ce482a9f89b))
* fixed English models not working for `WhisperHF` ([c356491](https://github.com/jianfch/stable-ts/commit/c3564915792f17eb5905e56c5ce78ce482a9f89b))
* fixed `get_device()` for `'mps'` ([53272cb](https://github.com/jianfch/stable-ts/commit/53272cb0376b2c1e32f0055a5505045681409aac))

## 2.15.5
* `WhisperHF.transcribe()` can now take generation parameters supported by `Transformers` ([133f323](https://github.com/jianfch/stable-ts/commit/133f3235c34aced23c95b10a37df3031b6e4cf0f))
* added logic to replace `None` timestamps returned by Hugging Face Whisper models ([8bbe0c5](https://github.com/jianfch/stable-ts/commit/8bbe0c59bf3125df310596ec0dfecef783b3b437))
* changed `whisper_word_level.hf_whisper.load_hf_pipe()` model loading method([a684fb4](https://github.com/jianfch/stable-ts/commit/a684fb4f5b271079b243b32a9552426340419806))

## 2.15.4
* added DeepFilterNet (https://github.com/Rikorose/DeepFilterNet) as supported denoiser ([3fafd04](https://github.com/jianfch/stable-ts/commit/3fafd04902666675bb0e5786c8a08ee18b68f908))
* added Whisper on Hugging Face Transformers to CLI ([3fafd04](https://github.com/jianfch/stable-ts/commit/3fafd04902666675bb0e5786c8a08ee18b68f908))
* fixed CLI throwing OSError when input is a URL and --output is not specified ([3fafd04](https://github.com/jianfch/stable-ts/commit/3fafd04902666675bb0e5786c8a08ee18b68f908))
* fixed `WhisperHF.transcribe()` unable to load when audio is URL or certain formats ([3fafd04](https://github.com/jianfch/stable-ts/commit/3fafd04902666675bb0e5786c8a08ee18b68f908))

## 2.15.3
* added support for Whisper on Hugging Face Transformers ([9197b5c](https://github.com/jianfch/stable-ts/commit/9197b5c92b31d67b58e2c1bbf8e5283cd7faf4cb))
* fixed non-speech suppression not working properly for `transcribe_any()` ([9197b5c](https://github.com/jianfch/stable-ts/commit/9197b5c92b31d67b58e2c1bbf8e5283cd7faf4cb))

## 2.15.2
* changed default to `dtype=numpy.int32` for all Numpy int arrays ([3886bc6](https://github.com/jianfch/stable-ts/commit/3886bc6932a91784d8229deb3e811f75d8ea122e))

## 2.15.1
* removed `shell=True` in `.audio.utils.get_metadata()` ([e8f72a3](https://github.com/jianfch/stable-ts/commit/e8f72a3b27ddb14202d50f9eaf4af7afccadbaa2))

## 2.15.0
* added "「" to `prepend_punctuations` and "」" to `append_punctuations` ([9968a45](https://github.com/jianfch/stable-ts/commit/9968a4510b2973be40e75d7978c915f3727ed06f))
* added `AudioLoader` class for handling general audio loading ([9968a45](https://github.com/jianfch/stable-ts/commit/9968a4510b2973be40e75d7978c915f3727ed06f))
* added `NonSpeechPredictor` class for handling non-speech detection ([9968a45](https://github.com/jianfch/stable-ts/commit/9968a4510b2973be40e75d7978c915f3727ed06f))
* added `default.py` to hold global default states ([9968a45](https://github.com/jianfch/stable-ts/commit/9968a4510b2973be40e75d7978c915f3727ed06f))
* added `failure_threshold` to `align()` ([9968a45](https://github.com/jianfch/stable-ts/commit/9968a4510b2973be40e75d7978c915f3727ed06f))
* added `stream` to functions that use `AudioLoader` internally ([9968a45](https://github.com/jianfch/stable-ts/commit/9968a4510b2973be40e75d7978c915f3727ed06f))
* added progress bars for VAD and Demucs operations ([9968a45](https://github.com/jianfch/stable-ts/commit/9968a4510b2973be40e75d7978c915f3727ed06f))
* changed text normalization for `align()` ([6d0746c](https://github.com/jianfch/stable-ts/commit/6d0746cd79ece5aea0e08488da1687ef4fe65935))
* changed `WhisperResult` to ignore segments with no words ([6d0746c](https://github.com/jianfch/stable-ts/commit/6d0746cd79ece5aea0e08488da1687ef4fe65935))
* changed `nonspeech_error` default from 0.3 to 0.1 for all functions ([9968a45](https://github.com/jianfch/stable-ts/commit/9968a4510b2973be40e75d7978c915f3727ed06f))
* changed `nonspeech_skip` default from 3.0 to 5.0 for `align()` ([9968a45](https://github.com/jianfch/stable-ts/commit/9968a4510b2973be40e75d7978c915f3727ed06f))
* changed `use_word_position` behavior ([9968a45](https://github.com/jianfch/stable-ts/commit/9968a4510b2973be40e75d7978c915f3727ed06f))
* changed to load Demucs into cache for reuse by default ([9968a45](https://github.com/jianfch/stable-ts/commit/9968a4510b2973be40e75d7978c915f3727ed06f))
* deprecated and replaced `demucs` and `demucs_options` with `denoiser` and `denoiser_options` ([9968a45](https://github.com/jianfch/stable-ts/commit/9968a4510b2973be40e75d7978c915f3727ed06f))
* dropped `ffmpeg-python` dependency ([9968a45](https://github.com/jianfch/stable-ts/commit/9968a4510b2973be40e75d7978c915f3727ed06f))
* dropped dependencies: more-itertools, transformers ([9968a45](https://github.com/jianfch/stable-ts/commit/9968a4510b2973be40e75d7978c915f3727ed06f))
* fixed `align()` producing empty word slices ([6d0746c](https://github.com/jianfch/stable-ts/commit/6d0746cd79ece5aea0e08488da1687ef4fe65935))
* fixed `refine()` exceeding the max token count ([#297](https://github.com/jianfch/stable-ts/issues/297)) ([f6d61c2](https://github.com/jianfch/stable-ts/commit/f6d61c228d5a00f89637422537d36cd358e5b90d))
* fixed issues in `transcribe_any()` caused by unspecified samplerate ([9968a45](https://github.com/jianfch/stable-ts/commit/9968a4510b2973be40e75d7978c915f3727ed06f))
* fixed `vad=True` causing first word of segment to be grouped with previous segment ([9968a45](https://github.com/jianfch/stable-ts/commit/9968a4510b2973be40e75d7978c915f3727ed06f))
* refactored `audio.py`, `stabilization.py`, `whisper_word_level.py` into subpackages ([9968a45](https://github.com/jianfch/stable-ts/commit/9968a4510b2973be40e75d7978c915f3727ed06f))
* removed `demucs_output` ([9968a45](https://github.com/jianfch/stable-ts/commit/9968a4510b2973be40e75d7978c915f3727ed06f))

## 2.14.4
* added `output_demo.mp4` ([395c8a9](https://github.com/jianfch/stable-ts/commit/395c8a9e28c1dc92972ef093e18b3b8257b27a2d))
* fixed `align()` throwing `UnsortedException` ([f9ca03b](https://github.com/jianfch/stable-ts/commit/f9ca03b1b1c16a2cbae8e9231859dbedce10adf7))
* fixed `original_split=True` failing when there are more than one consecutive newlines ([f9ca03b](https://github.com/jianfch/stable-ts/commit/f9ca03b1b1c16a2cbae8e9231859dbedce10adf7))
* fixed (`align()` IndexError)(https://github.com/jianfch/stable-ts/issues/292#issuecomment-1890781467) ([f9ca03b](https://github.com/jianfch/stable-ts/commit/f9ca03b1b1c16a2cbae8e9231859dbedce10adf7))

## 2.14.3
* added `trust_repo=True` for loading Silero-VAD ([a6b2b05](https://github.com/jianfch/stable-ts/commit/a6b2b05568e75b1602a6e23891b59c4a9e218f6b))
* added `'master'` to the branch for loading Silero-VAD  ([a6b2b05](https://github.com/jianfch/stable-ts/commit/a6b2b05568e75b1602a6e23891b59c4a9e218f6b))
* fixed `align()` failing for faster whisper with certain languages ([677f233](https://github.com/jianfch/stable-ts/commit/677f233bedff857b56bfe48e10d095c40d7f6425))
* fixed `result.WhisperResult.apply_min_dur()` and `result.Segment.apply_min_dur()` to work as intended ([be2985e](https://github.com/jianfch/stable-ts/commit/be2985e852aeafe4444dbd81e9230476a662a6f0))
* removed `resampling_method="kaiser_window"` for all calls of `torchaudio.functional.resample()` ([a6b2b05](https://github.com/jianfch/stable-ts/commit/a6b2b05568e75b1602a6e23891b59c4a9e218f6b))

## 2.14.2
* updated `align()` logic ([738fd98](https://github.com/jianfch/stable-ts/commit/738fd98490584c492cf2f7873bdddaf7a0ec9d40))
* added `nonspeech_skip` to `align()` ([738fd98](https://github.com/jianfch/stable-ts/commit/738fd98490584c492cf2f7873bdddaf7a0ec9d40))
* added `show_unsorted` to `result.WhisperResult.__init__()` and `result.WhisperResult.raise_for_unsorted()` ([738fd98](https://github.com/jianfch/stable-ts/commit/738fd98490584c492cf2f7873bdddaf7a0ec9d40))
* added `use_word_position` to methods that support non-speech/silence suppression ([738fd98](https://github.com/jianfch/stable-ts/commit/738fd98490584c492cf2f7873bdddaf7a0ec9d40))
* fixed `result.WhisperResult.force_order()` to handle data with multiple consecutive unsort timestamps ([738fd98](https://github.com/jianfch/stable-ts/commit/738fd98490584c492cf2f7873bdddaf7a0ec9d40))
* fixed empty segment removal to work as intend for `result.WhisperResult` ([ef0a87e](https://github.com/jianfch/stable-ts/commit/ef0a87e036bacaa1b9d1261b82e74ba3118ac075))
* updated `README.md` to directly included the docstrings instead of hyperlinks ([738fd98](https://github.com/jianfch/stable-ts/commit/738fd98490584c492cf2f7873bdddaf7a0ec9d40))
* updated `result.save_as_json()` to include `ensure_ascii=False` as default ([738fd98](https://github.com/jianfch/stable-ts/commit/738fd98490584c492cf2f7873bdddaf7a0ec9d40))
* added `kwargs` to `result.save_as_json()` ([738fd98](https://github.com/jianfch/stable-ts/commit/738fd98490584c492cf2f7873bdddaf7a0ec9d40))
* updated demo videos ([3524aa2](https://github.com/jianfch/stable-ts/commit/3524aa2c9ea8aa8ca89721a0d54580e2c15b9340))

## 2.14.1
* fixed `result.WhisperResult.force_order()` causing IndexError ([0430a31](https://github.com/jianfch/stable-ts/commit/0430a31dc8fe05f4bc4a3cf39c28074d7afc79dd))
* updated `README.md` ([bc4601f](https://github.com/jianfch/stable-ts/commit/bc4601ff354a56fc119867eb6d55ad20ee2b1de7))

## 2.14.0
* added `nonspeech_sections` property to `result.WhisperResult` ([191674b](https://github.com/jianfch/stable-ts/commit/191674beefdddbce026732d5fd93026f85c26772))
* added `nonspeech_error` for silence suppression ([191674b](https://github.com/jianfch/stable-ts/commit/191674beefdddbce026732d5fd93026f85c26772))
* changed `min_word_dur` behavior for silence suppression ([191674b](https://github.com/jianfch/stable-ts/commit/191674beefdddbce026732d5fd93026f85c26772))
* changed silence suppression behavior ([191674b](https://github.com/jianfch/stable-ts/commit/191674beefdddbce026732d5fd93026f85c26772))
* updated `README.md` ([191674b](https://github.com/jianfch/stable-ts/commit/191674beefdddbce026732d5fd93026f85c26772))

## 2.13.7
* fixed `result.WhisperResult.split_by_punctuation()` not working if `min_words`/`min_chars`/`min_dur` are unspecified ([d51edb6](https://github.com/jianfch/stable-ts/commit/d51edb6ad86b06f4582f4c06fcf8a4b6dc8e0bca))

## 2.13.6
* added `show_regroup_history()` to `result.WhisperResult` ([df4a199](https://github.com/jianfch/stable-ts/commit/df4a199a54f58a85f4fefbb19f0902cfd037d1cc))
* added new attribute, `regroup_history`, to `.result.WhisperResult` ([df4a199](https://github.com/jianfch/stable-ts/commit/df4a199a54f58a85f4fefbb19f0902cfd037d1cc))
* added `min_words`, `min_chars`, `min_dur` to `result.WhisperResult.split_by_punctuation()` ([df4a199](https://github.com/jianfch/stable-ts/commit/df4a199a54f58a85f4fefbb19f0902cfd037d1cc))
* updated `README.md` ([e86c571](https://github.com/jianfch/stable-ts/commit/e86c5714e2a56a10ee3e24b8f2e60a1c1c3fe506))

## 2.13.5
* added `get_content_by_time()` to `result.WhisperResult` ([900797a](https://github.com/jianfch/stable-ts/commit/900797a90a35b09fef9fe13574e963dc6a58aa56))
* added `get_result()` to `result.Segment` ([900797a](https://github.com/jianfch/stable-ts/commit/900797a90a35b09fef9fe13574e963dc6a58aa56))
* added `get_segment()` to `result.WordTiming` ([900797a](https://github.com/jianfch/stable-ts/commit/900797a90a35b09fef9fe13574e963dc6a58aa56))
* added `text_ouput.result_to_txt()`/`result.WhisperResult.to_txt()` ([900797a](https://github.com/jianfch/stable-ts/commit/900797a90a35b09fef9fe13574e963dc6a58aa56))
* added editing methods to `result.WhisperResult`: `remove_word()`, `remove_segment()`, `remove_repetition()`, `remove_words_by_str()`, `fill_in_gaps()` ([900797a](https://github.com/jianfch/stable-ts/commit/900797a90a35b09fef9fe13574e963dc6a58aa56))
* added editing methods to list of 'method keys' in `result.WhisperResult.regroup()` ([900797a](https://github.com/jianfch/stable-ts/commit/900797a90a35b09fef9fe13574e963dc6a58aa56))
* changed `result.Segment.to_display_str()` to enclose segment text in double quotes ([900797a](https://github.com/jianfch/stable-ts/commit/900797a90a35b09fef9fe13574e963dc6a58aa56))
* implemented `__getitem__` and `__delitem__`  for `result.Segment` and `result.WhisperResult` ([900797a](https://github.com/jianfch/stable-ts/commit/900797a90a35b09fef9fe13574e963dc6a58aa56))
* updated docstrings of `whisper_word_level.load_model()` and `whisper_word_level.load_faster_whisper()` ([900797a](https://github.com/jianfch/stable-ts/commit/900797a90a35b09fef9fe13574e963dc6a58aa56))

## 2.13.4
* added `result.WhisperResult.split_by_duration()` ([71b9f1f](https://github.com/jianfch/stable-ts/commit/71b9f1fcbd1268f8bfe95bba6a394a2bc2e7339b))
* fixed `newline=True` for `result.WhisperResult._split_segments()` ([71b9f1f](https://github.com/jianfch/stable-ts/commit/71b9f1fcbd1268f8bfe95bba6a394a2bc2e7339b))
* fixed docstring of `result.WhisperResult.split_by_length()` ([71b9f1f](https://github.com/jianfch/stable-ts/commit/71b9f1fcbd1268f8bfe95bba6a394a2bc2e7339b))
* updated Whisper to [v20231117](https://github.com/openai/whisper/commit/e58f28804528831904c3b6f2c0e473f346223433) ([71b9f1f](https://github.com/jianfch/stable-ts/commit/71b9f1fcbd1268f8bfe95bba6a394a2bc2e7339b))

## 2.13.3
* added `--faster_whisper`, `-fw` to CLI ([a038ad1](https://github.com/jianfch/stable-ts/commit/a038ad18c8ed86aafe298b5a0f67c45bf7ffadb2))
* added `--locate`, `-lc` to CLI ([a038ad1](https://github.com/jianfch/stable-ts/commit/a038ad18c8ed86aafe298b5a0f67c45bf7ffadb2))
* changed `alignment.align()` to be compatible with faster-whisper ([a038ad1](https://github.com/jianfch/stable-ts/commit/a038ad18c8ed86aafe298b5a0f67c45bf7ffadb2))
* changed `verbose` behavior for `alignment.locate()` ([a038ad1](https://github.com/jianfch/stable-ts/commit/a038ad18c8ed86aafe298b5a0f67c45bf7ffadb2))
* fixed inconsistent syntax and typo in docstrings ([a038ad1](https://github.com/jianfch/stable-ts/commit/a038ad18c8ed86aafe298b5a0f67c45bf7ffadb2))
* removed assertions for checking timestamp order when using `__add__()` with `result.Segment` or `result.WordTiming` ([a038ad1](https://github.com/jianfch/stable-ts/commit/a038ad18c8ed86aafe298b5a0f67c45bf7ffadb2))

## 2.13.2
* added `newline` to `split_by_gap()`, `split_by_punctuation()`, `split_by_length()` ([b336735](https://github.com/jianfch/stable-ts/commit/b336735ff784bb59690eec8f9f706b0151dda74c))
* added `progress_callback` to `whisper_word_level.load_faster_whisper.faster_transcribe()` ([b336735](https://github.com/jianfch/stable-ts/commit/b336735ff784bb59690eec8f9f706b0151dda74c))
* fixed [#241](https://github.com/jianfch/stable-ts/issues/241) ([5c512a1](https://github.com/jianfch/stable-ts/commit/5c512a1880b937025792d441b98f5a13ab5a735e))
* refactored `_COMPATIBLE_WHISPER_VERSIONS`, `_required_whisper_ver`, `warn_compatibility_issues()` ([b336735](https://github.com/jianfch/stable-ts/commit/b336735ff784bb59690eec8f9f706b0151dda74c))
* updated `README.md` ([3dfbd72](https://github.com/jianfch/stable-ts/commit/3dfbd722a8edb606a2d819fee49ff9c5db4bf0f2))
* updated `--model` for CLI to be compatible with checkpoint paths ([b336735](https://github.com/jianfch/stable-ts/commit/b336735ff784bb59690eec8f9f706b0151dda74c))
*  `merge_all_segments()` with faster logic ([b336735](https://github.com/jianfch/stable-ts/commit/b336735ff784bb59690eec8f9f706b0151dda74c))
* updated `verbose` for `.whisper_word_level.load_faster_whisper.faster_transcribe()` ([b336735](https://github.com/jianfch/stable-ts/commit/b336735ff784bb59690eec8f9f706b0151dda74c))
* updated whisper version to `v20231106` ([b336735](https://github.com/jianfch/stable-ts/commit/b336735ff784bb59690eec8f9f706b0151dda74c))

## 2.13.1
* added `avg_prob_threshold` to `whisper_word_level.transcribe_stable()` ([58ece35](https://github.com/jianfch/stable-ts/commit/58ece35f2975c906ebd44a469055a88e3e816e67))
* added `fast_mode` to `alignment.align()` ([58ece35](https://github.com/jianfch/stable-ts/commit/58ece35f2975c906ebd44a469055a88e3e816e67))
* added `utils.UnsortedException` ([eb00d29](https://github.com/jianfch/stable-ts/commit/eb00d291e54d82d381a967c30385002db0c8b1ae))
* added `word_dur_factor` and `max_word_dur` to `alignment.align()` ([58ece35](https://github.com/jianfch/stable-ts/commit/58ece35f2975c906ebd44a469055a88e3e816e67))
* changed `check_sorted` for `result.WhisperResult` to also accept a path ([eb00d29](https://github.com/jianfch/stable-ts/commit/eb00d291e54d82d381a967c30385002db0c8b1ae))
* changed `clip_start` default to `None` for `result.WhisperResult.clamp_max()` ([58ece35](https://github.com/jianfch/stable-ts/commit/58ece35f2975c906ebd44a469055a88e3e816e67))
* corrected docstrings of `suppress_silence` and `suppress_word_ts` ([58ece35](https://github.com/jianfch/stable-ts/commit/58ece35f2975c906ebd44a469055a88e3e816e67))
* fixed `timing.find_alignment_stable()` returning negative timestamps ([58ece35](https://github.com/jianfch/stable-ts/commit/58ece35f2975c906ebd44a469055a88e3e816e67))

## 2.13.0
* added `alignment.locate()` ([a777206](https://github.com/jianfch/stable-ts/commit/a777206b98f0bb19ebdf36b2e39704dceeb2c04f))
* added `utils.format_timestamp()` and `utils.make_safe()` ([a777206](https://github.com/jianfch/stable-ts/commit/a777206b98f0bb19ebdf36b2e39704dceeb2c04f))
* added `utils.safe_print()` ([a777206](https://github.com/jianfch/stable-ts/commit/a777206b98f0bb19ebdf36b2e39704dceeb2c04f))
* added `demucs`, `demucs_options`, `only_voice_freq` to `alignment.refine()` ([a777206](https://github.com/jianfch/stable-ts/commit/a777206b98f0bb19ebdf36b2e39704dceeb2c04f))
* added `to_display_str()` to `result.Segment` ([a777206](https://github.com/jianfch/stable-ts/commit/a777206b98f0bb19ebdf36b2e39704dceeb2c04f))
* added `demucs_options` to `whisper_word_level.load_faster_whisper.faster_transcribe()` ([a777206](https://github.com/jianfch/stable-ts/commit/a777206b98f0bb19ebdf36b2e39704dceeb2c04f))
* updated `--output` / `-o` ([a777206](https://github.com/jianfch/stable-ts/commit/a777206b98f0bb19ebdf36b2e39704dceeb2c04f))
* changed `audio` to always expected to be 16kHz for `torch.Tensor` or `numpy.ndarray` ([a777206](https://github.com/jianfch/stable-ts/commit/a777206b98f0bb19ebdf36b2e39704dceeb2c04f))
* fixed `alignment.align()` failing if `text` a `result.WhisperResult` without tokens ([a777206](https://github.com/jianfch/stable-ts/commit/a777206b98f0bb19ebdf36b2e39704dceeb2c04f))
* fixed `original_split=True` by replacing line breaks with space ([97a316d](https://github.com/jianfch/stable-ts/commit/97a316db54975edd7f2c180684028a03823981b7))
* fixed `result_to_ass()` failing to return to base color when using `tag` ([83ae509](https://github.com/jianfch/stable-ts/commit/83ae509e3b99051a008188c649a434b1f6fb4b96))
* improved efficiency of segment splitting for `alignment.align()` when `original_split=True` ([a777206](https://github.com/jianfch/stable-ts/commit/a777206b98f0bb19ebdf36b2e39704dceeb2c04f))
* refactored the audio preprocessing into `audio.prep_audio()` ([a777206](https://github.com/jianfch/stable-ts/commit/a777206b98f0bb19ebdf36b2e39704dceeb2c04f))
* removed `_is_whisper_repo_version` from `utils.py` ([a777206](https://github.com/jianfch/stable-ts/commit/a777206b98f0bb19ebdf36b2e39704dceeb2c04f))
* renamed `original_spit` to `original_split` for `alignment.align()` ([a777206](https://github.com/jianfch/stable-ts/commit/a777206b98f0bb19ebdf36b2e39704dceeb2c04f))
* set `action="extend"` for all CLI keyword arguments that take multiple values ([a777206](https://github.com/jianfch/stable-ts/commit/a777206b98f0bb19ebdf36b2e39704dceeb2c04f))
* changed `demucs` to also accept a Demucs model instance([a777206](https://github.com/jianfch/stable-ts/commit/a777206b98f0bb19ebdf36b2e39704dceeb2c04f))
* deprecated `time_scale`, `input_sr`, `demucs_output`, `demucs_device` ([a777206](https://github.com/jianfch/stable-ts/commit/a777206b98f0bb19ebdf36b2e39704dceeb2c04f))
* updated docstrings ([a777206](https://github.com/jianfch/stable-ts/commit/a777206b98f0bb19ebdf36b2e39704dceeb2c04f))

## 2.12.3
* updated `alignment.align()` to raise warning on failure ([b9ac041](https://github.com/jianfch/stable-ts/commit/b9ac0415838aedc6db8c8cf4402af8592f009a8a))
* changed `language` into a required parameter ([b9ac041](https://github.com/jianfch/stable-ts/commit/b9ac0415838aedc6db8c8cf4402af8592f009a8a))
* fixed `alignment.align()` endlessly looping ([b9ac041](https://github.com/jianfch/stable-ts/commit/b9ac0415838aedc6db8c8cf4402af8592f009a8a))

## 2.12.2
* added `original_spit` to `alignment.align()` ([45bd3bc](https://github.com/jianfch/stable-ts/commit/45bd3bcc16991cb9ef8232e05198c878f8b07ec9))
* ignore `DecodingOptions` for alignment ([1fb3009](https://github.com/jianfch/stable-ts/commit/1fb30099642c15912bbd43affdfdfc8af630703b))

## 2.12.1
* changed `abs_dur_change` default to `None` ([dd1452e](https://github.com/jianfch/stable-ts/commit/dd1452ed159ea71e3c703ad077f987fefebd26c0))
* changed `abs_prob_decrease` default to `0.5` ([dd1452e](https://github.com/jianfch/stable-ts/commit/dd1452ed159ea71e3c703ad077f987fefebd26c0))
* changed `alignment.refine()` allow durations to increase ([dd1452e](https://github.com/jianfch/stable-ts/commit/dd1452ed159ea71e3c703ad077f987fefebd26c0))
* changed `rel_prob_decrease` default to `0.3` ([dd1452e](https://github.com/jianfch/stable-ts/commit/dd1452ed159ea71e3c703ad077f987fefebd26c0))
* changed `rel_rel_prob_decrease` to optional ([dd1452e](https://github.com/jianfch/stable-ts/commit/dd1452ed159ea71e3c703ad077f987fefebd26c0))
* changed the usage of original probability in `alignment.refine()` ([dd1452e](https://github.com/jianfch/stable-ts/commit/dd1452ed159ea71e3c703ad077f987fefebd26c0))
* fixed CLI not using `decode_options` ([9aba3dc](https://github.com/jianfch/stable-ts/commit/9aba3dc6655743e051479c479809020fdadf95fe))
* fixed `adjust_by_silence()` throwing `TypeError` ([92d51b9](https://github.com/jianfch/stable-ts/commit/92d51b900f246fb50e1e7be7610b755605cef872))
* updated `README.md` [3643092](https://github.com/jianfch/stable-ts/commit/3643092fcc6993d73b752ad0d246ef454608eeb5))

## 2.12.0
* added `--align` to CLI ([c90ff06](https://github.com/jianfch/stable-ts/commit/c90ff06bc55694034994010e05b5fc2f50070b03))
* added `alignment.refine()` for refining timestamps ([138cb6b](https://github.com/jianfch/stable-ts/commit/138cb6b8ca69835ec2b02aa66935579713a4667e))
* added `--refine` and `--refine_option` to CLI ([138cb6b](https://github.com/jianfch/stable-ts/commit/138cb6b8ca69835ec2b02aa66935579713a4667e))
* added `segment_id` and `id` to `result.WordTiming` ([138cb6b](https://github.com/jianfch/stable-ts/commit/138cb6b8ca69835ec2b02aa66935579713a4667e))
* added description to transcription progress bar ([138cb6b](https://github.com/jianfch/stable-ts/commit/138cb6b8ca69835ec2b02aa66935579713a4667e))
* fixed `align()` not working when `text` is a `result.WhisperResult` ([138cb6b](https://github.com/jianfch/stable-ts/commit/138cb6b8ca69835ec2b02aa66935579713a4667e))
* fixed `transcribe()` throwing error if `suppress_silence=False` ([138cb6b](https://github.com/jianfch/stable-ts/commit/138cb6b8ca69835ec2b02aa66935579713a4667e))
* updated `README.md` ([c90ff06](https://github.com/jianfch/stable-ts/commit/c90ff06bc55694034994010e05b5fc2f50070b03))

## 2.11.7
* fixed `--debug` not showing the first option ([857df9a](https://github.com/jianfch/stable-ts/commit/857df9a0f2756ee99a7eaeeee2610d245bd1c3c5))
* fixed `demucs` and `only_voice_freq` for `transcribe_stable()` ([7f62a9d](https://github.com/jianfch/stable-ts/commit/7f62a9db98db5c63d93daa3397fed900f59b0596))
* fixed `demucs` for `transcribe_minimal()` ([857df9a](https://github.com/jianfch/stable-ts/commit/857df9a0f2756ee99a7eaeeee2610d245bd1c3c5))
* fixed `only_voice_freq` for `transcribe_minimal()` ([7f62a9d](https://github.com/jianfch/stable-ts/commit/7f62a9db98db5c63d93daa3397fed900f59b0596))
* fixed progress bar for faster-whisper ([7f62a9d](https://github.com/jianfch/stable-ts/commit/7f62a9db98db5c63d93daa3397fed900f59b0596))
* updated `transcribe_minimal()` to accept more options ([857df9a](https://github.com/jianfch/stable-ts/commit/857df9a0f2756ee99a7eaeeee2610d245bd1c3c5))
* updated `transcribe_stable()` for faster-whisper models to accept more options ([7f62a9d](https://github.com/jianfch/stable-ts/commit/7f62a9db98db5c63d93daa3397fed900f59b0596))

## 2.11.6
* delete `_demo` directory ([66f4376](https://github.com/jianfch/stable-ts/commit/66f437612561b1359dac40765d12ddf84bb4e3bd))
* fixed [#216](https://github.com/jianfch/stable-ts/issues/216) ([1732ac0](https://github.com/jianfch/stable-ts/commit/1732ac0b8a14000545c7571b45c4a71e9d6ca2ea))

## 2.11.5
* added `'us'` as method key to `WhisperResult.regroup()` ([da33bf5](https://github.com/jianfch/stable-ts/commit/da33bf5e0f1c93e00eda9dcb5d4f635a1f1e1b35))
* added `--demucs_option`, `--model_option`, `--transcribe_option`, `--save_option` to CLI ([da33bf5](https://github.com/jianfch/stable-ts/commit/da33bf5e0f1c93e00eda9dcb5d4f635a1f1e1b35))
* added `--transcribe_method` to CLI ([da33bf5](https://github.com/jianfch/stable-ts/commit/da33bf5e0f1c93e00eda9dcb5d4f635a1f1e1b35))
* added `Segment.words_by_lock()`, `WhisperResult.all_words_by_lock()` ([da33bf5](https://github.com/jianfch/stable-ts/commit/da33bf5e0f1c93e00eda9dcb5d4f635a1f1e1b35))
* added `strip` to `WhisperResult.lock()` ([e98c3d6](https://github.com/jianfch/stable-ts/commit/e98c3d6191fae901fc0ed2fa217785d11c9d649e))
* fixed docstring of `WhisperResult.lock()` ([05bba74](https://github.com/jianfch/stable-ts/commit/05bba74e900fa0aca081ac06ebb5b91e5a50b068))
* improved `--debug` for CLI ([da33bf5](https://github.com/jianfch/stable-ts/commit/da33bf5e0f1c93e00eda9dcb5d4f635a1f1e1b35))
* improved `even_split=True` for `WhisperResult.split_by_length()` ([da33bf5](https://github.com/jianfch/stable-ts/commit/da33bf5e0f1c93e00eda9dcb5d4f635a1f1e1b35))
* updated docstring of `WhisperResult.split_by_length()` ([da33bf5](https://github.com/jianfch/stable-ts/commit/da33bf5e0f1c93e00eda9dcb5d4f635a1f1e1b35))

## 2.11.4
* added `lock()` to `WhisperResult` ([384fc3c](https://github.com/jianfch/stable-ts/commit/384fc3ce40c80279acbac2bc86f671f319cfefb6))
* added `'l'` as method key to `WhisperResult.regroup()` ([384fc3c](https://github.com/jianfch/stable-ts/commit/384fc3ce40c80279acbac2bc86f671f319cfefb6))
* added progress bar to transcription with faster-whisper ([5ac6f5e](https://github.com/jianfch/stable-ts/commit/5ac6f5e11eedddc7bedf728c70eb19b303b4b63c))
* updated `--output_format` to accept multiple formats ([384fc3c](https://github.com/jianfch/stable-ts/commit/384fc3ce40c80279acbac2bc86f671f319cfefb6))
* updated `WhisperResult.reset()` to match its initialization ([384fc3c](https://github.com/jianfch/stable-ts/commit/384fc3ce40c80279acbac2bc86f671f319cfefb6))
* updated `regroup()` to parse `regroup_algo` into dict ([384fc3c](https://github.com/jianfch/stable-ts/commit/384fc3ce40c80279acbac2bc86f671f319cfefb6))

## 2.11.3
* added `check_sorted` to `WhisperResult` ([4054ca1](https://github.com/jianfch/stable-ts/commit/4054ca174d9ad0fffbaf0a1e09cdbfdf3fd1cbca))
* added `check_sorted` to `transcribe_any()` ([07eaf9e](https://github.com/jianfch/stable-ts/commit/07eaf9e34b7996fe46a7afcac0ae8f8d7e73bfa7))
* added `round_all_timestamps()` to `result.Segment` and `result.WordTiming` ([4a7e52b](https://github.com/jianfch/stable-ts/commit/4a7e52bef88552c91249ff05c580e84cd5c168de))
* changed default to `word_timestamps=True` for `faster_transcribe()` ([4a7e52b](https://github.com/jianfch/stable-ts/commit/4a7e52bef88552c91249ff05c580e84cd5c168de))
* changed `raise_for_unsorted()` logic ([4a7e52b](https://github.com/jianfch/stable-ts/commit/4a7e52bef88552c91249ff05c580e84cd5c168de))
* fixed `WhisperResult.force_order()` to work as intended ([4a7e52b](https://github.com/jianfch/stable-ts/commit/4a7e52bef88552c91249ff05c580e84cd5c168de))

## 2.11.2
* fixed `condition_on_previous_text` ([641cce7](https://github.com/jianfch/stable-ts/commit/641cce70baeb611795a9c828846631ee5d394bd4))
* updated Whisper version to `v20230918` ([641cce7](https://github.com/jianfch/stable-ts/commit/641cce70baeb611795a9c828846631ee5d394bd4))

## 2.11.1
* added `token_step` to `align()` ([ac3b38c](https://github.com/jianfch/stable-ts/commit/ac3b38ccd5d5cc28326a754c8137c4b6ec7bd1d1))
* delete `_demo` directory ([b592731](https://github.com/jianfch/stable-ts/commit/b592731d4b5241618fd9f30496037daa0a724f6e))
* fixed [#205](https://github.com/jianfch/stable-ts/issues/205) ([ac3b38c](https://github.com/jianfch/stable-ts/commit/ac3b38ccd5d5cc28326a754c8137c4b6ec7bd1d1))
* updated `README.md` ([d0340ef](https://github.com/jianfch/stable-ts/commit/d0340ef34aa68285a0e4410e889069b5a9cce036), [ffa05a4](https://github.com/jianfch/stable-ts/commit/ffa05a419a45d68fb93466957d756509e65e13b4))

## 2.11.0
* added `Whisper.adjust_by_result()` ([6da3dd8](https://github.com/jianfch/stable-ts/commit/6da3dd849d9f0a2fe9b3b2e7612a1ca58599c3d6))
* added `alignment.align()`  ([6da3dd8](https://github.com/jianfch/stable-ts/commit/6da3dd849d9f0a2fe9b3b2e7612a1ca58599c3d6))
* added `load_faster_whisper()` ([6da3dd8](https://github.com/jianfch/stable-ts/commit/6da3dd849d9f0a2fe9b3b2e7612a1ca58599c3d6))
* fixed `encode_video_comparison()` unable to encode more than two subtitle files ([6da3dd8](https://github.com/jianfch/stable-ts/commit/6da3dd849d9f0a2fe9b3b2e7612a1ca58599c3d6))
* fixed `verbose` not working for `transcribe_minimal()` ([6da3dd8](https://github.com/jianfch/stable-ts/commit/6da3dd849d9f0a2fe9b3b2e7612a1ca58599c3d6))
* refactored compatibility warning into `warn_compatibility_issues()` in `utils.py` ([6da3dd8](https://github.com/jianfch/stable-ts/commit/6da3dd849d9f0a2fe9b3b2e7612a1ca58599c3d6))
* refactored post-inference silence suppress into `WhisperResult.adjust_by_silence()` ([6da3dd8](https://github.com/jianfch/stable-ts/commit/6da3dd849d9f0a2fe9b3b2e7612a1ca58599c3d6))

## 2.10.1
* added `demucs_options` to `transcribe()` ([91cf2b1](https://github.com/jianfch/stable-ts/commit/91cf2b15085ea0b826fb00c517f2132b66cbb651))
* added `ignore_compatibility` to `transcribe()` ([91cf2b1](https://github.com/jianfch/stable-ts/commit/91cf2b15085ea0b826fb00c517f2132b66cbb651))
* changed compatibility warning to distinguish between mismatch version number and repo version ([91cf2b1](https://github.com/jianfch/stable-ts/commit/91cf2b15085ea0b826fb00c517f2132b66cbb651))
* changed heuristic for identifying Whisper version number to avoid false positives ([91cf2b1](https://github.com/jianfch/stable-ts/commit/91cf2b15085ea0b826fb00c517f2132b66cbb651))

## 2.10.0
* added `transcribe_minimal()` ([ef8a7f1](https://github.com/jianfch/stable-ts/commit/ef8a7f1202fc7b2f315b3b690924425a5d465b72))
* added `force_order` to `result.WhisperResult` ([ef8a7f1](https://github.com/jianfch/stable-ts/commit/ef8a7f1202fc7b2f315b3b690924425a5d465b72))
* added `max_instant_words` to `transcribe()` ([ef8a7f1](https://github.com/jianfch/stable-ts/commit/ef8a7f1202fc7b2f315b3b690924425a5d465b72))
* added `progress_callback` to `transcribe()` ([ef8a7f1](https://github.com/jianfch/stable-ts/commit/ef8a7f1202fc7b2f315b3b690924425a5d465b72))
* changed default to `clip_start=True` for `WhisperResult.clamp_max()` ([ef8a7f1](https://github.com/jianfch/stable-ts/commit/ef8a7f1202fc7b2f315b3b690924425a5d465b72))
* added logic to check if the installed Whisper version is compatible ([e53f4be](https://github.com/jianfch/stable-ts/commit/e53f4be56314a45fe5032b9ac37bf28844e1172b))
* fixed `tag` for `result_to_ass()` to work as intended ([ea8cac8](https://github.com/jianfch/stable-ts/commit/ea8cac848535d3408760d5227c7f6945459c3192))

## 2.9.0
* added logic to ensure ascending timestamps in `result.WhisperResult` ([fd78cd7](https://github.com/jianfch/stable-ts/commit/fd78cd7432f098306706f49d7de75be888847222))
* updated default regroup algorithm  ([fd78cd7](https://github.com/jianfch/stable-ts/commit/fd78cd7432f098306706f49d7de75be888847222), [77dcfdf](https://github.com/jianfch/stable-ts/commit/77dcfdf0817d59d5aed054d6d420945aa11d69dc))
* updated long form transcription logic ([fd78cd7](https://github.com/jianfch/stable-ts/commit/fd78cd7432f098306706f49d7de75be888847222))
* fixed skipping words ([77dcfdf](https://github.com/jianfch/stable-ts/commit/77dcfdf0817d59d5aed054d6d420945aa11d69dc))
* avoid computing higher temperatures on `no_speech` segments ([fd78cd7](https://github.com/jianfch/stable-ts/commit/fd78cd7432f098306706f49d7de75be888847222))
* removed any segments that contains only punctuations ([fd78cd7](https://github.com/jianfch/stable-ts/commit/fd78cd7432f098306706f49d7de75be888847222))
* removed segments with 50%+ instantaneous words ([fd78cd7](https://github.com/jianfch/stable-ts/commit/fd78cd7432f098306706f49d7de75be888847222))
* updated `README.md` ([f5b4c22](https://github.com/jianfch/stable-ts/commit/f5b4c22d60e757a93937dddc2e53ee0098a2ff80))

## 2.8.1
* allow `regroup_algo` to be bool for `regroup()` ([4984163](https://github.com/jianfch/stable-ts/commit/49841632722e6dc5fa20a67a28c69dc377849d32))

## 2.8.0
* added `even_split` to `split_by_length()` ([7b867d6](https://github.com/jianfch/stable-ts/commit/7b867d6713c4374f017bb088711fcf6abb47e139))
* changed default behavior of `split_by_length()` ([7b867d6](https://github.com/jianfch/stable-ts/commit/7b867d6713c4374f017bb088711fcf6abb47e139))
* changed default to `verbose=False` for `clamp_max()` ([7b867d6](https://github.com/jianfch/stable-ts/commit/7b867d6713c4374f017bb088711fcf6abb47e139))

## 2.7.2
* ignore `min_word_dur` when missing words timestamps ([e93c280](https://github.com/jianfch/stable-ts/commit/e93c28047167b6b14b4efbd31719205971099e65))
* fixed `min_word_dur` not working for word timestamps ([e93c280](https://github.com/jianfch/stable-ts/commit/e93c28047167b6b14b4efbd31719205971099e65))

## 2.7.1
* added `verbose` to `clamp_max()` ([70f092f](https://github.com/jianfch/stable-ts/commit/70f092f9f765123985b0c91005f9bc8e23a7be50))
* fixed typo in `examples\non-whisper.ipynb` ([70f092f](https://github.com/jianfch/stable-ts/commit/70f092f9f765123985b0c91005f9bc8e23a7be50))

## 2.7.0
* added `clamp_max()` to `WhisperResult` and `WordTiming` ([bfe93ab](https://github.com/jianfch/stable-ts/commit/bfe93abfe8150dd3c5d0a5e9bfb05a12f19c46c5))
* added `cm` as method key for `clamp_max()` ([bfe93ab](https://github.com/jianfch/stable-ts/commit/bfe93abfe8150dd3c5d0a5e9bfb05a12f19c46c5))
* added `non_whisper.transcribe_any()` ([789bb54](https://github.com/jianfch/stable-ts/commit/789bb543115e486fad450dffd40e0dcf39e041d5))
* changed default to `suppress_ts_tokens=False` ([789bb54](https://github.com/jianfch/stable-ts/commit/789bb543115e486fad450dffd40e0dcf39e041d5))
* fixed hyperlinks in `README.md` not linking to the latest commit ([87636ef](https://github.com/jianfch/stable-ts/commit/87636efefc59ed724444580106415cf95f17b6ad))
* fixed incorrect line numbers for docstring hyperlinks ([52b8b7a](https://github.com/jianfch/stable-ts/commit/52b8b7a315e1fc6aeb1e4728e9810fe5e81ecd79))

## 2.6.4
* fixed `--regroup` default ([af5579e](https://github.com/jianfch/stable-ts/commit/af5579e273f7caf1b80ad5a1d76c51b8f320572d))

## 2.6.3
* added string form custom regrouping algorithm ([cc352cd](https://github.com/jianfch/stable-ts/commit/cc352cd9e1d98803aea3b4bac90f665a3ab21cc8))

## 2.6.2
* fixed [#153](https://github.com/jianfch/stable-ts/issues/153) ([9e3ba72](https://github.com/jianfch/stable-ts/commit/9e3ba724a2f4781127248c7d196173a79191474d))
* removed max limit on audio threshold) ([9e3ba72](https://github.com/jianfch/stable-ts/commit/9e3ba724a2f4781127248c7d196173a79191474d))
* updated `non-whisper.ipynb` ([da3721b](https://github.com/jianfch/stable-ts/commit/da3721b9dea0f9e7875917d679932588a889a569), [7866462](https://github.com/jianfch/stable-ts/commit/786646251cac8c79d7d563fabb0ab2db43cf45d5))

## 2.6.1
* changed `result.WhisperResult` to only require necessary data to initialize ([cdf3ea9](https://github.com/jianfch/stable-ts/commit/cdf3ea9ff7ef940db8bd616ebeb9730154d50a91))
* added `--karaoke` to CLI ([cdf3ea9](https://github.com/jianfch/stable-ts/commit/cdf3ea9ff7ef940db8bd616ebeb9730154d50a91))
* updated `README.md` ([0635e15](https://github.com/jianfch/stable-ts/commit/0635e15d7880442cb7bb62ef5ac70961717b4eae), [2f094f8](https://github.com/jianfch/stable-ts/commit/2f094f8fbf5474b57749a1d697e9d4882458e71f), [fb23c27](https://github.com/jianfch/stable-ts/commit/fb23c27c4ef729ead539b8bd356a8a2e05592124))

## 2.6.0
* added support for TSV output format ([d30d0d1](https://github.com/jianfch/stable-ts/commit/d30d0d1cfb5b17b4bf59c3fafcbbd21e37598ab9))
* changed to VTT and ASS default output to use more efficient formats ([d30d0d1](https://github.com/jianfch/stable-ts/commit/d30d0d1cfb5b17b4bf59c3fafcbbd21e37598ab9))
* fixed non-VAD suppression not working properly ([d30d0d1](https://github.com/jianfch/stable-ts/commit/d30d0d1cfb5b17b4bf59c3fafcbbd21e37598ab9))
* improved language detection ([d30d0d1](https://github.com/jianfch/stable-ts/commit/d30d0d1cfb5b17b4bf59c3fafcbbd21e37598ab9))

## 2.5.3
* fixed [#145](https://github.com/jianfch/stable-ts/issues/145) ([efbe6b6](https://github.com/jianfch/stable-ts/commit/efbe6b66008af8a07547d2c958fda608ec02a265))

## 2.5.2
* re-added [#143](https://github.com/jianfch/stable-ts/pulls/143) ([5ea52b2](https://github.com/jianfch/stable-ts/commit/5ea52b22b56af9134f243281055ecb5a0b9a842e))

## 2.5.1
* added logic for loading audio with yt-dlp ([8960922](https://github.com/jianfch/stable-ts/commit/896092288c6768b16220ed3cb3458db85e8c3b9d))
* added `only_ffmpeg` to `transcribe()` and CLI ([8960922](https://github.com/jianfch/stable-ts/commit/896092288c6768b16220ed3cb3458db85e8c3b9d))
* added `shell=True` to subprocess call ([a8df3b5](https://github.com/jianfch/stable-ts/commit/a8df3b59cabe06a41f853bf976cb8c475ffd9ade))

## 2.5.0
* added classes: `SegmentMatch` and `WhisperResultMatches` ([1eabb37](https://github.com/jianfch/stable-ts/commit/1eabb37847c46a4e0f73345d4c6b0ba7d8d6ce55))
* added fallback logic to word alignment ([1eabb37](https://github.com/jianfch/stable-ts/commit/1eabb37847c46a4e0f73345d4c6b0ba7d8d6ce55))
* added `find()` to `result.WhisperResult` ([1eabb37](https://github.com/jianfch/stable-ts/commit/1eabb37847c46a4e0f73345d4c6b0ba7d8d6ce55))
* added `suppress_ts_tokens` and `gap_padding` to `transcribe()` and CLI ([1eabb37](https://github.com/jianfch/stable-ts/commit/1eabb37847c46a4e0f73345d4c6b0ba7d8d6ce55))
* added `shell=True` to `is_ytdlp_available()` ([d2b7f3f](https://github.com/jianfch/stable-ts/commit/d2b7f3f7994967973bd501eeea4be520be5de641))
* fixed `NaN` values in the logits ([1eabb37](https://github.com/jianfch/stable-ts/commit/1eabb37847c46a4e0f73345d4c6b0ba7d8d6ce55))

## 2.4.1
* added `result_to_any()` ([eab8319](https://github.com/jianfch/stable-ts/commit/eab83194f715cdf4890281a2b0b8b1f34ab104f2))
* changed rtl to `reverse_text` ([eab8319](https://github.com/jianfch/stable-ts/commit/eab83194f715cdf4890281a2b0b8b1f34ab104f2))

## 2.4.0
* added `offset_time()` to `WhisperResult`, `Segment`, `WordTiming` ([1447a66](https://github.com/jianfch/stable-ts/commit/1447a66b8ffe2d01645d88826fbfb8f5729f7410))
* added support for audio as URLs ([1447a66](https://github.com/jianfch/stable-ts/commit/1447a66b8ffe2d01645d88826fbfb8f5729f7410))
* fixed `language` detection for English models ([1447a66](https://github.com/jianfch/stable-ts/commit/1447a66b8ffe2d01645d88826fbfb8f5729f7410))

## 2.3.1
* added `split_callback` ([44af5c4](https://github.com/jianfch/stable-ts/commit/44af5c4709a8d554bf2d94bf4dfd3f6e3dde0f0d))
* changed parameters of `split_callback` ([c003ce4](https://github.com/jianfch/stable-ts/commit/c003ce4315d489a51e136b67e917d55750a42b96))
* corrected the docstring for `rtl` ([169e014](https://github.com/jianfch/stable-ts/commit/169e0146370a08224f85b587774cdca464ff1534))
* fixed punctuation split/merge to work as intended ([a84a346](https://github.com/jianfch/stable-ts/commit/a84a3464a8056d9e12ed96aa28e713885ae2b21d))

## 2.3.0
* added regrouping list ([a0021bd](https://github.com/jianfch/stable-ts/commit/a0021bd60e853ece6bfe4ea470aa26e1265bddca))
* added `--max_chars` and `--max_words` to CLI ([f913d6f](https://github.com/jianfch/stable-ts/commit/f913d6f31b6ba2c4359b01581cad05a50d8ad9f6))
* added `rtl` [#116](https://github.com/jianfch/stable-ts/issues/116) ([f913d6f](https://github.com/jianfch/stable-ts/commit/f913d6f31b6ba2c4359b01581cad05a50d8ad9f6))
* corrected VAD pytorch requirement ([60f668d](https://github.com/jianfch/stable-ts/commit/60f668df509b0583687a9c1a7f33f18a02bab78f))
* fixed `visualize_suppression()` error when `max_width=-1` ([918e3ba](https://github.com/jianfch/stable-ts/commit/918e3baa08f8817735ad9bafca797d04f6921386))
* fixed out of range error ([918e3ba](https://github.com/jianfch/stable-ts/commit/918e3baa08f8817735ad9bafca797d04f6921386))

## 2.2.0
* added `merge_all_segments()` to `result.WhisperResult`  ([7c69535](https://github.com/jianfch/stable-ts/commit/7c6953526dce5d9058b23e8d0c223272bf808be7))
* added `split_by_length()` to `result.WhisperResult` ([7c69535](https://github.com/jianfch/stable-ts/commit/7c6953526dce5d9058b23e8d0c223272bf808be7))

## 2.1.3
* fixed transcription logic ([d44d287](https://github.com/jianfch/stable-ts/commit/d44d287cbf93d1ad703359023fcb3f4ebbe02d46))

## 2.1.2
* added Tips to `README.md` ([c21e198](https://github.com/jianfch/stable-ts/commit/c21e1982c19ce60695e22085a7251686693ffc19))
* added new token splitting method ([fa813fe](https://github.com/jianfch/stable-ts/commit/fa813fea52962cafbc035f0afd611afe0d308adc))
* fixed [#112](https://github.com/jianfch/stable-ts/issues/112)([3985791](https://github.com/jianfch/stable-ts/commit/3985791192d47dacb32758e3003b4a1df1c99cd3))
* fixed [#117](https://github.com/jianfch/stable-ts/issues/117) ([3985791](https://github.com/jianfch/stable-ts/commit/3985791192d47dacb32758e3003b4a1df1c99cd3))
* added instructions for installing demucs via error ([de3c812](https://github.com/jianfch/stable-ts/commit/de3c81237a671ac730833d98f5709358586d13c4))
* added `encoding='utf-8'` to `read_me()` in `setup.py` ([ff34b27](https://github.com/jianfch/stable-ts/commit/ff34b27da4b2eb3f9a80a2072785c5e46d477df0))
* updated `README.md` ([dfb147e](https://github.com/jianfch/stable-ts/commit/dfb147ebd70f0617a6ab612864a73afd05528faa))

## 2.1.1
* added `mel_first` ([8fa5670](https://github.com/jianfch/stable-ts/commit/8fa5670316eecf3b823a7bc453866c907d616a51))
* fixed: to not apply `min_dur` on words if segments contains no words ([8fa5670](https://github.com/jianfch/stable-ts/commit/8fa5670316eecf3b823a7bc453866c907d616a51))
* updated regroup demo video ([e9932fe](https://github.com/jianfch/stable-ts/commit/e9932fefb6c1e863cae1e3cf412ef5ea852cd521))

## 2.1.0
* added 1.x to 2.x guide `README.md` ([19ba449](https://github.com/jianfch/stable-ts/commit/19ba4499c997859647c7b93d36adfcb08e90ace4))
* added `min_dur` ([8c62ee1](https://github.com/jianfch/stable-ts/commit/8c62ee157c11ba0154020d9e51d70384b64deda8))
* fixed `regroup` ([8c62ee1](https://github.com/jianfch/stable-ts/commit/8c62ee157c11ba0154020d9e51d70384b64deda8))

## 2.0.4
* fixed timestamps to jump backwards ([26918d5](https://github.com/jianfch/stable-ts/commit/26918d599349ea7199c1be5bddce2509c13781a0))

## 2.0.3
* changed default `strip=True` for `result_to_srt_vtt()` ([ce4c7b3](https://github.com/jianfch/stable-ts/commit/ce4c7b31ca607f4c84c3323a489ecfc887bea33e))
* keep segments when if segment has no words from the start ([6ccfa17](https://github.com/jianfch/stable-ts/commit/6ccfa17fc29526b593955d32c976bf4803e75082))
* improved `stabilization.audio2loudness()` efficiency ([db99d6b](https://github.com/jianfch/stable-ts/commit/db99d6b47a908c25a68611f60ae1d66a5e974ac6))
* fixed `regroup=True` when `word_timestamp=sFalse` ([6ccfa17](https://github.com/jianfch/stable-ts/commit/6ccfa17fc29526b593955d32c976bf4803e75082))
* fixed `word_level=False` failing output when `word_timestamps=False` ([ce4c7b3](https://github.com/jianfch/stable-ts/commit/ce4c7b31ca607f4c84c3323a489ecfc887bea33e))
* fixed ASS output formatting ([ce4c7b3](https://github.com/jianfch/stable-ts/commit/ce4c7b31ca607f4c84c3323a489ecfc887bea33e))
* updated `README.md` ([f9f7c51](https://github.com/jianfch/stable-ts/commit/f9f7c514d78490ffca68956e9d69ed3a1567eee5))


## 2.0.2
* fixed `wav2mask()` when `suppress_silence=True` ([e884e38](https://github.com/jianfch/stable-ts/commit/e884e380db43964597d3c9d99f539fea277936a1))
* fixed typo ([58006ec](https://github.com/jianfch/stable-ts/commit/58006ec9a70ab139d70fe964b00ff924c00c7971))

## 2.0.1
* added examples videos/images ([75611f7](https://github.com/jianfch/stable-ts/commit/75611f7559565d8b1eb2f55efb41191a552d2631))
* updated `README.md` ([0f2f699](https://github.com/jianfch/stable-ts/commit/0f2f69920c2ca10915c3bf159aabfc3b031c5c7e))

## 2.0.0
* added segment-level and word-level support to SRT/VTT/ASS outputs ([2248087](https://github.com/jianfch/stable-ts/commit/2248087c9c2fd1ff5e0d10bd9834fd7bfed8162a))
* added `result.WhisperResult` ([2248087](https://github.com/jianfch/stable-ts/commit/2248087c9c2fd1ff5e0d10bd9834fd7bfed8162a))
* added Silero VAD support ([2248087](https://github.com/jianfch/stable-ts/commit/2248087c9c2fd1ff5e0d10bd9834fd7bfed8162a))
* added `visualize_suppression()` ([2248087](https://github.com/jianfch/stable-ts/commit/2248087c9c2fd1ff5e0d10bd9834fd7bfed8162a))
* added regrouping methods ([2248087](https://github.com/jianfch/stable-ts/commit/2248087c9c2fd1ff5e0d10bd9834fd7bfed8162a))
* changed python requirement from 3.7+ to 3.8+ ([2248087](https://github.com/jianfch/stable-ts/commit/2248087c9c2fd1ff5e0d10bd9834fd7bfed8162a))
* improved non-vad suppression ([2248087](https://github.com/jianfch/stable-ts/commit/2248087c9c2fd1ff5e0d10bd9834fd7bfed8162a))
* improve word-level timestamps reliability ([2248087](https://github.com/jianfch/stable-ts/commit/2248087c9c2fd1ff5e0d10bd9834fd7bfed8162a))
* updated `README.md` ([eb5e68c](https://github.com/jianfch/stable-ts/commit/eb5e68c439e1fef973d0bc29bd7f28e3c0508a32))

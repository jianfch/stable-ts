import os
import subprocess as sp
import warnings
from typing import List

__all__ = ['encode_video_comparison']


def encode_video_comparison(
        audiofile,
        subtitle_files: List[str],
        output_videopath: str = None,
        *,
        labels: List[str] = None,
        height: int = 90,
        width: int = 720,
        color: str = 'black',
        fontsize: int = 70,
        border_color: str = 'white',
        label_color: str = 'white',
        label_size: int = 14,
        fps: int = 25,
        video_codec: str = None,
        audio_codec: str = None,
        overwrite=False,
        only_cmd: bool = False,
        verbose=True
) -> (str, None):
    vc = '' if video_codec is None else f' -c:v {video_codec}'
    ac = '' if audio_codec is None else f' -c:a {audio_codec}'
    background = f'-f lavfi -i color=size={width}x{height}:rate={fps}:color={color}'
    audio = f'-i "{audiofile}"'
    cfilters0 = []
    assert labels is None or len(labels) == len(subtitle_files)
    for i, sub in enumerate(subtitle_files):
        label = sub if labels is None else labels[i]
        label = label.replace("'", '"')
        fil = f"[0]drawtext=text='{label}':fontcolor={label_color}:fontsize={label_size}:x=10:y=10[a{i}]," \
              f"[a{i}]subtitles='{sub}':force_style='Fontsize={fontsize}'[b{i}]"
        cfilters0.append(fil)
    cfilters0.append(f'color={border_color}:{width}x3[c]')
    cfilters1 = ('[c]'.join(
        f'[b{i}]' for i in range(len(cfilters0) - 1)) + f'vstack=inputs={(len(cfilters0) - 1) * 2 - 1}')
    final_fil = ','.join(cfilters0) + f';{cfilters1}'
    ow = '-y' if overwrite else '-n'
    if output_videopath is None:
        name = os.path.split(os.path.splitext(audiofile)[0])[1]
        output_videopath = f'{name}_sub_comparison.mp4'
    cmd = f'ffmpeg {ow} {background} {audio} -filter_complex "{final_fil}"{vc}{ac} -shortest "{output_videopath}"'
    if only_cmd:
        return cmd
    if verbose:
        print(cmd)
    rc = sp.run(cmd, capture_output=not verbose).returncode
    if rc == 0:
        if verbose:
            print(f'Encoded: {output_videopath}')
    else:
        warnings.warn(f'Failed to encode {output_videopath}')


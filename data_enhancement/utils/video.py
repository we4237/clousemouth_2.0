import json
import os
import subprocess
from typing import Tuple

import av
import numpy as np


def get_video_info(video_path,need_audio=True) -> Tuple[str, str, str, str]:
    assert os.path.exists(video_path) and os.path.isfile(video_path), "video_path is illegal"
    cmd = f'ffprobe -v quiet -print_format json -show_format -show_streams "{video_path}"'
    result = subprocess.run(cmd, capture_output=True, shell=True)
    output = result.stdout.strip()
    video_info = json.loads(output)

    if need_audio:
        assert len(video_info.get('streams', [])) >= 2, f'video or audio stream is missing in {video_path}'

    video_stream_info = video_info.get('streams')[0]
    src_colorspace, color_range, color_transfer, color_primaries = video_stream_info.get('color_space'), \
                                                                   video_stream_info.get('color_range'), \
                                                                   video_stream_info.get('color_transfer'), \
                                                                   video_stream_info.get('color_primaries')

    return src_colorspace, color_range, color_transfer, color_primaries


def get_first_frame(video_path, source_color_space="ITU709") -> np.ndarray:
    container = av.open(video_path)
    try:
        for _, frame in enumerate(container.decode(video=0)):
            first_frame = av.VideoFrame.reformat(frame, format="bgr24",
                                                 src_colorspace=source_color_space).to_ndarray()
            break
    except Exception as e:
        raise e
    finally:
        container.close()
    return first_frame

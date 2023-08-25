"""图片、音频、视频的编码操作
"""

from typing import List, Union, Generator, Optional
import numpy as np
import io

class BaseMultimediaEncoder:
    """编码器接口
    """
    def init_video(self, temp_dir: str = None) -> None:
        """初始化视频

        Args:
            temp_dir (str, optional): 临时目录. Defaults to None.
        """
        raise NotImplementedError

    def encode_audio(self, audio: Union[str, io.BytesIO]) -> Optional[Generator]:
        """编码输入的音频流

        Args:
            audio: 音频路径，或者io.BytesIO(bytes_audio)

        Yields:
            编码后的音频帧
        """
        raise NotImplementedError

    def encode_frames(self, frames: 'np.ndarray') -> Optional[Generator]:
        """编码输入的图片帧

        Args:
            frames: 输入帧数组

        Yields:
            编码后的图片帧
        """
        raise NotImplementedError

    def check_new_video_seg(self) -> List[str]:
        """检查是否有新的video seg生成。

        Returns:
            video segs的绝对路径['/xxx/yyy/zzz_001.mp4', '/xxx/yyy/zzz_002.mp4']
            如果没有新的视频seg生成，则返回[]
        """
        raise NotImplementedError


    def end_encoding(self) -> None:
        """初始化视频encoding
        """
        raise NotImplementedError



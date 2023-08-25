"""人脸检测
"""
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np


class BaseFaceDetector:
    """人脸检测器
    """

    def detect(self, frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """检测输入帧中的人脸

        Args:
            frames: 驱动帧, 输入为BGR格式的，channel-last的image数组/images的数字
                    shape应该为(num_frame, frame_height, frame_width, frame_channel)
                    如果shape是(frame_height, frame_width, frame_channel)，则会在外面再加一个axis

        Returns:
            (boxes, landmarks)
            Tuple[np.ndarray, np.ndarray]: 截取的人脸bounding box以及landmarks
        """
        raise NotImplementedError


class BaseFaceDetectionPostProcess():
    """人脸检测，后处理
    """
    pass

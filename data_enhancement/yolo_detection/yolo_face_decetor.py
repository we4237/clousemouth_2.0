import copy
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from talkingface.detection.yolov5.models.experimental import attempt_load
from talkingface.detection.yolov5.utils_yolo.general_v3 import letterbox, check_img_size, non_max_suppression_face, \
    scale_coords

# from data_enhancement.yolo_detection.datasets import LoadImages

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

@dataclass
class Yolov5FaceDetectorConfig:
    """Yolov5FaceWithLandmarkDetector的配置类
    Attributes:
        batch_size (int, optional): batch大小. Defaults to 4.
        device (str, optional): 使用GPU还是CPU. Defaults to 'cuda'.
        img_size (int, optional): preprocess中的resize的大小. Defaults to 640.
        conf_threshold (float, optional): conf判断阈值. Defaults to 0.6.
        iou_threshold (float, optional): iou判断阈值. Defaults to 0.5.

    """
    model_path: str = ''
    batch_size: int = 4
    device: str = 'cuda'
    image_size: int = 640
    conf_threshold: float = 0.6
    iou_threshold: float = 0.5


class Yolov5FaceDetector:

    def __init__(self, config: Yolov5FaceDetectorConfig) -> None:
        """使用yolov5进行人脸检测

        Args:

        """
        self.batch_size = config.batch_size
        self.device = config.device
        self.image_size = config.image_size
        self.conf_threshold = config.conf_threshold
        self.iou_threshold = config.iou_threshold
        self.model = self._load_model(config.model_path)

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """加载模型

        Args:
            model_path (str): 模型位置

        Returns:
            torch.nn.Module: torch模型
        """
        # load FP32 model
        model = attempt_load(model_path, map_location=self.device)
        return model

    def detect(self, source_image: np.ndarray):
        source_h, source_w = source_image.shape[:2]
        orgimg = letterbox(source_image, new_shape=(self.image_size,) * 2)[0]

        img0 = copy.deepcopy(orgimg)
        h0, w0 = orgimg.shape[:2]  # orig hw
        r = self.image_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        imgsz = check_img_size(self.image_size, self.model.stride.max())  # check img_size

        img = letterbox(img0, new_shape=imgsz)[0]
        # Convert from w,h,c to c,w,h
        img = img.transpose(2, 0, 1).copy()

        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img)[0]

        # Apply NMS
        pred = non_max_suppression_face(pred, self.conf_threshold, self.iou_threshold)
        # 没有找到人脸
        if len(pred) == 0:
            return None

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            im0 = source_image.copy()

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for j in range(det.size()[0]):
                    xyxy = det[j, :4].view(-1).tolist()
                    x1, y1, x2, y2 = xyxy
                    # 将框扩大1.2
                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    new_width = width * 1.2
                    new_height = height * 1.2
                    x1 = max(x_center - new_width / 2, 0)
                    y1 = max(y_center - new_height / 2,0)
                    x2 = min(x_center + new_width / 2, source_w)
                    y2 = min(y_center + new_height / 2,source_h)
                    return int(x1), int(y1), int(x2), int(y2)


if __name__ == "__main__":
    t1 = round(time.time())
    config = Yolov5FaceDetectorConfig(model_path='/code/talking_face_v3/core/talkingface/resources/models/face_detector/yolov5l-face_landmark.pt')
    decteor = Yolov5FaceDetector(config)
    t2 = round(time.time())
    frame = '/data/data_enhancement/dataset/video1/yolo_frame.jpg'
    frame = cv2.imread(frame)
    for _ in range(125):
        face_box = decteor.detect(frame)
    t3 = round(time.time())
    print(face_box)
    print(f'total_cost: {t3-t1}, init_cost: {t2-t1}, infer_cost:{t3-t2}')

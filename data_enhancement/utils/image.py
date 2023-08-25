import cv2
import torch


def frame_preprocess(frame, face_box, transformer, target_size=512):
    cropped_frame = frame[face_box[1]:face_box[3], face_box[0]:face_box[2], :]
    cropped_img = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

    target_height, target_width = target_size, target_size
    # 获取原始图像的尺寸
    height, width = cropped_img.shape[:2]
    # 计算缩放比例
    scale = min(target_width / width, target_height / height)
    # 计算调整后的尺寸
    resized_width = int(width * scale)
    resized_height = int(height * scale)
    # 缩放图像
    resized_img = cv2.resize(cropped_img, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    # 计算填充尺寸
    delta_w = target_width - resized_width
    delta_h = target_height - resized_height
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left
    # 使用cv2.copyMakeBorder()函数进行填充
    resized_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    result_img = transformer(resized_img)
    return result_img
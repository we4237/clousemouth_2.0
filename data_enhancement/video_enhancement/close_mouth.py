import glob
import os
import subprocess
import time
from dataclasses import dataclass

import av
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
from tqdm import tqdm

from data_enhancement.yolo_detection.yolo_face_decetor import Yolov5FaceDetector, Yolov5FaceDetectorConfig
from data_enhancement.utils.base import timer
from data_enhancement.utils.image import frame_preprocess
from data_enhancement.utils.video import get_video_info, get_first_frame
from data_enhancement.video_enhancement.video_encoder import PyavVideoEncoder
from talkingface.codecs.video import libx264Options
from model import BiSeNet
from model import Fairseq


@dataclass
class CloseMouthWorkerConfig:
    """
    为视频增加静音帧的配置类

    worker 使用的模型与V3 保持一致
    Args:
        wav2vector_model_path (str): wav2vector_model_path 模型地址
        yolo_model_path (str): yolo_model_path 模型地址
        face_segment_classes (str): 分类数，默认19
        face_segment_model_path (str): 模型地址
        resnet_state_dict_file (str): 权重文件地址
        fps (str): 视频帧率，默认只支持25
    """
    wav2vector_model_path: str = ''
    yolo_model_path: str = ''
    face_segment_classes: int = 19
    face_segment_model_path: str = ''
    resnet_state_dict_file: str = ''
    fps: int = 25


class CloseMouthWorker:
    """
    为视频增加静音帧的类

    该worker 需要需talkingface_V3 搭配使用，共用一套镜像及模型
    默认每分钟选择4个适合插入闭嘴的位置，插入5秒的静音帧
    Args:
        config (CloseMouthWorkerConfig): worker的配置类
    """

    def __init__(self, config):
        # 静音检测模型
        print('close_mouth worker init start')
        self._face_detector_model = None
        self._face_segment_model = None
        self._wav2vector_model = None
        self.config = config
        self._model_init(config)
        self.work_result_dir = ''
        print('close_mouth worker init finish')

    def _model_init(self, config):
        self._load_model_wav2vector(config.wav2vector_model_path)
        # self._load_model_face_detector(config.yolo_model_path)
        self._load_model_face_segment(config.face_segment_model_path,
                                      config.resnet_state_dict_file,
                                      config.face_segment_classes)

    @property
    def face_detector(self):
        if not self._face_detector_model:
            self._face_detector_model = self._load_model_face_detector(self.config.yolo_model_path)
        return self._face_detector_model

    @timer("load face_detector model")
    def _load_model_face_detector(self, model_path):
        yolo_config = Yolov5FaceDetectorConfig(model_path=model_path)
        face_detector_model = Yolov5FaceDetector(yolo_config)
        return face_detector_model

    @timer("load wav2vector model")
    def _load_model_wav2vector(self, cache_dir: str, device: str = "cuda"):
        self.wav2vector = Fairseq(cache_dir, device)

    @timer("load face_segment model")
    def _load_model_face_segment(self, model_path: str, resnet_state_dict_file: str, n_classes: int = 19):
        """加载模型
            Args:
                model_path (str): 模型路径
                resnet_state_dict_file (str): 权重文件
                n_classes (int, optional): 分类类别数. Defaults to 19.

            Returns:
                torch.nn.Module: 返回模型
            """
        model = BiSeNet(n_classes=n_classes, resnet_state_dict_file=resnet_state_dict_file).cuda()
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        model.eval()
        self.face_segment = model

    def _wave2vector(self, audio_path: str):
        predict = self.wav2vector.get_emission3(audio_path)
        return predict

    @staticmethod
    def _split_audio(video_path: str, audio_path: str, sample_rate: int = 16000):
        cmd = f'ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar {sample_rate} {audio_path} -y -loglevel  quiet'
        subprocess.run(cmd, shell=True)
        return

    def _frame_preprocess_bak(self, frame, face_box, transformer, target_size=512):
        cropped_frame = frame[face_box[1]:face_box[3], face_box[0]:face_box[2], :]
        frame = torch.from_numpy(cropped_frame).cuda()[:, :, [2, 1, 0]].permute(2, 0, 1).contiguous()

        # 图片scale
        origin_h, origin_w = frame.shape[1:]
        scale_ratio = (min(target_size / origin_h, target_size / origin_w))
        resized_h, resized_w = int(origin_h * scale_ratio), int(origin_w * scale_ratio)
        scaled_frame = F.interpolate(frame.unsqueeze(0).float(), (resized_h, resized_w), mode='bilinear')

        # 计算填充尺寸
        delta_h, delta_w = target_size - resized_h, target_size - resized_w
        top, left = delta_h // 2, delta_w // 2
        bottom, right = delta_h - top, delta_w - left

        # 边框填充
        bordered_frame = F.pad(scaled_frame, (left, right, top, bottom), value=0)[0]
        target_frame = transformer(bordered_frame / 255)

        return target_frame

    def _get_mouth_area(self, video_path, src_colorspace, face_box, start_idx):
        mouth_area_list = []
        teeth_mark, up_lip_mark, down_lip_mark = 11, 12, 13
        # transformer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        container = av.open(video_path)
        try:
            video = tqdm(enumerate(container.decode(video=0)))
            video.set_description(f'calculate mouth area')
            for idx, org_frame in video:
                if idx < start_idx:
                    continue
                frame = av.VideoFrame.reformat(org_frame, format="bgr24", src_colorspace=src_colorspace).to_ndarray()
                frame = frame_preprocess(frame, face_box, transformer).cuda()
                with torch.no_grad():
                    out = self.face_segment(frame.unsqueeze(0))[0]
                    parsing = out.argmax(1)
                    for idx, pair in enumerate(parsing):
                        unique = torch.unique(pair)
                        # 统计嘴巴区域像素数量
                        if teeth_mark not in unique:
                            vis_parsing_anno = pair.clone().to(torch.uint8)
                            face_coord = torch.nonzero(
                                torch.logical_or(vis_parsing_anno == up_lip_mark, vis_parsing_anno == down_lip_mark))
                            mouth_area = len(face_coord)
                            mouth_area_list.append(mouth_area)
                        else:
                            mouth_area_list.append(0)
        except Exception as e:
            raise e
        finally:
            container.close()
        mouth_areas = np.array(mouth_area_list)
        return mouth_areas

    def _get_silence_frame_index(self, predicts, mouth_areas, start_idx, fps=25, threshold=3,
                                 index_count_per_minute=4, baseline=None):
        ans = {}
        predicts = predicts[:, start_idx * 2:]
        length = predicts.shape[1] // 2

        last_zero_idx = -1

        for i in range(min(len(mouth_areas), length)):
            # 声音或者画面中有一个不为0，就不是静音帧，跳过
            if (predicts[:, i * 2] != 0 or predicts[:, i * 2 + 1] != 0) or mouth_areas[i] == 0:
                last_zero_idx = i
                continue
            else:
                # 如果是连续的静音帧，就继续往后找
                if i + 1 != min(len(mouth_areas), length) and mouth_areas[i + 1] != 0:
                    continue

                # 如果当前静音片段长度小于阈值，就跳过, 为保证不命中短暂停顿的首尾，需要去掉首尾两个静音帧
                silent_part_start = last_zero_idx + 1 + 1
                silent_part_end = i - 1
                silent_part_length = silent_part_end - silent_part_start + 1
                if silent_part_length < threshold:
                    continue

                silent_part = list(mouth_areas[silent_part_start:silent_part_end + 1])


                # 获取baseline，选择第一个静音片段的中位数作为baseline
                if not baseline:
                    middle_index = silent_part_length // 2
                    target_index = silent_part_start + silent_part.index(sorted(silent_part)[middle_index])
                    baseline = mouth_areas[target_index]
                else:
                    target_index = silent_part_start + silent_part.index(sorted(silent_part, key=lambda x:abs(x-baseline))[0])

                target_index_time_location = target_index // (fps * 60) #目前只支持25fps
                ans[target_index] = target_index_time_location

        frame_numbers = list(ans.keys())
        frame_times = list(ans.values())
        print(f'origin_silence_index_list :{frame_numbers}, count: {len(frame_numbers)}')

        filtered_frames = []

        if len(frame_numbers) == 0:
            return filtered_frames

        for minute in range(0, max(frame_times) + 1):
            # 获取当前分钟的帧数列表
            frames_in_minute = [frame for m, frame in zip(frame_times, frame_numbers) if m == minute]

            if len(frames_in_minute) <= index_count_per_minute:
                # 如果帧数不超过4个，则全部记录下来
                filtered_frames.extend(frames_in_minute)
                # print(f'第{minute}分钟记录了{len(frames_in_minute)}帧')
            else:
                # 选择最接近baseline大小的四个帧数
                diff = [abs(mouth_areas[frame] - baseline) for frame in frames_in_minute]
                sorted_frames = sorted(zip(frames_in_minute, diff), key=lambda x: x[1])
                closest_frames = [frame for frame, _ in sorted_frames[:4]]
                filtered_frames.extend(sorted(closest_frames))
                # print(f'第{minute}分钟记录了4帧')
        filtered_frames = [idx + start_idx for idx in filtered_frames]
        print(f'filtered_frames: {filtered_frames}')
        return filtered_frames

    def _generate_silence_video_part(self, part_video_path, frame, pyav_dst_colorspace, x264_color_dict,
                                     silence_duration, fps=25):
        video_height, video_width = frame.shape[:2]
        part_video_encoder = PyavVideoEncoder(part_video_path, video_height, video_width, x264_color_dict)
        for _ in range(silence_duration * fps):
            part_video_encoder.encode_video_frame(frame, pyav_dst_colorspace)
        part_video_encoder.encode_finish()

        print(f'generate silence video part {part_video_path} finish')

        return True

    def _generate(self,
                  video_path,
                  video_height,
                  video_width,
                  pyav_src_colorspace,
                  pyav_dst_colorspace,
                  x264_color_dict,
                  silence_frame_index_list,
                  start_idx,
                  worker_dir,
                  silence_duration=5,
                  fps=25
                  ):
        result_video_path = os.path.join(worker_dir, 'extend_video.mp4')
        result_silence_frame_path_map = {}
        result_silence_video_path_map = {}

        if len(silence_frame_index_list) == 0:
            return video_path, result_silence_video_path_map, result_silence_frame_path_map

        in_video_container = av.open(video_path)
        in_audio_container = av.open(video_path)

        result_video_encoder = PyavVideoEncoder(result_video_path, video_height, video_width, x264_color_dict)

        current_silence_index = 0
        try:
            # 生成视频
            input_video_stream = in_video_container.streams.get(video=0)[0]
            videos = tqdm(enumerate(in_video_container.decode(input_video_stream)))
            videos.set_description('encode extend video stream')
            for idx, frame in videos:
                if idx < start_idx:
                    continue
                frame = av.VideoFrame.reformat(frame, format="bgr24", src_colorspace=pyav_src_colorspace).to_ndarray()
                result_video_encoder.encode_video_frame(frame, pyav_dst_colorspace)
                if silence_frame_index_list and current_silence_index < len(silence_frame_index_list) and idx == \
                        silence_frame_index_list[current_silence_index]:

                    # 插入连续闭嘴画面
                    for _ in range(silence_duration * fps):
                        result_video_encoder.encode_video_frame(frame, pyav_dst_colorspace)

                    # 保存闭嘴视频片段
                    part_video_path = os.path.join(worker_dir, f'silence_part_{idx}.mp4')
                    self._generate_silence_video_part(part_video_path, frame, pyav_dst_colorspace, x264_color_dict, silence_duration)
                    result_silence_video_path_map[silence_frame_index_list[current_silence_index]] = part_video_path

                    # 保存闭嘴帧图片
                    silence_frame_path = os.path.join(worker_dir, f'silence_frame_{idx}.png')
                    cv2.imwrite(silence_frame_path, frame)
                    result_silence_frame_path_map[silence_frame_index_list[current_silence_index]] = silence_frame_path

                    current_silence_index += 1

            # 读取原始音频
            audio_frame_list = []
            input_audio_stream = in_audio_container.streams.get(audio=0)[0]
            audios = tqdm(in_audio_container.decode(input_audio_stream))
            audios.set_description('decode origin audio frame')
            for audio_frame in audios:
                audio_frame.pts = None
                resampled_audio_frame = result_video_encoder.audio_resampler.resample(audio_frame)
                audio_frame_list.append(resampled_audio_frame.to_ndarray())
            audio_frames = np.hstack(audio_frame_list)

            # 按照16k 采样率，每帧40ms，插入静音音频帧
            start_audio_idx = int(start_idx * 40 * 16000 / 1000)
            silence_audio_part = np.zeros((1, 16000 * 5), dtype=np.int16)
            for idx in silence_frame_index_list:
                end_audio_idx = int(idx * 40 * 16000 / 1000)
                audio_part = audio_frames[:1, start_audio_idx: end_audio_idx]
                start_audio_idx = end_audio_idx
                result_video_encoder.encode_audio_frame(audio_part)
                result_video_encoder.encode_audio_frame(silence_audio_part)
            result_video_encoder.encode_audio_frame(audio_frames[:1, start_audio_idx:])
            result_video_encoder.encode_finish()

        except Exception as e:
            raise e
        finally:
            in_video_container.close()
            in_audio_container.close()

        return result_video_path, result_silence_video_path_map, result_silence_frame_path_map

    @staticmethod
    def get_pyav_color_space(src_colorspace, target_color_space):
        if target_color_space is None:
            target_color_space = src_colorspace
        pyav_src_colorspace = libx264Options.colorspace.get(src_colorspace)
        pyav_dst_colorspace = libx264Options.colorspace.get(target_color_space)
        if pyav_src_colorspace is None or pyav_dst_colorspace is None:
            raise Exception(f'colorspace only support :{list(libx264Options.colorspace.keys())} but get: {src_colorspace}, {target_color_space}')

        return pyav_src_colorspace, pyav_dst_colorspace

    def get_face_box(self, video_path: str, source_color_space: str, deadline_index: int = 25 * 5):
        face_box,idx = None, 0
        container = av.open(video_path)
        try:
            for idx, frame in enumerate(container.decode(video=0)):
                if idx > deadline_index:
                    break
                frame_npy = av.VideoFrame.reformat(frame, format="bgr24",
                                                   src_colorspace=source_color_space).to_ndarray()
                face_box = self.face_detector.detect(frame_npy)
                if face_box is not None and len(face_box) > 0:
                    break
        except Exception as e:
            raise e
        finally:
            container.close()
        print(f'video_face_box: {face_box}, index: {idx}')
        return face_box, idx

    def extend_silence_video(self,
                             video_path: str,
                             worker_dir: str,
                             ):
        """
        生成静音视频方法

        Args:
            video_path (str): 原始视频地址
            worker_dir (str): 结果保存地址
        Returns:
            dict: 处理后的视频数据字典，包含以下键值对:
            - 'silence_frames' (dict): 包含静音帧位置和对应视频路径的字典。
              静音帧位置是键，对应的视频路径是值。
            - 'silence_videos' (dict): 包含静音帧位置和对应截图路径的字典。
              静音帧位置是键，对应的截图路径是值。
            - 'video_with_silence' (str): 处理后的带有静音标记的视频路径。

        Examples:
            video_data = {
            ...     'silence_videos': {
            ...         88: '/data/result/silence_part_0.mp4',
            ...         631: '/data/result/silence_part_1.mp4',
            ...         # 其他静音帧位置和对应视频路径
            ...     },
            ...     'silence_frames': {
            ...         88: '/data/result/silence_frame_0.png',
            ...         631: '/data/result/silence_frame_1.png',
            ...         # 其他静音帧位置和对应截图路径
            ...     },
            ...     'video_with_silence': '/data/data_enhancement/dataset/video1/result/extend_video.mp4'
            ... }
        """
        try:
            # 参数校验
            src_colorspace, color_range, color_transfer, color_primaries = get_video_info(video_path)
            pyav_src_colorspace, pyav_dst_colorspace = self.get_pyav_color_space(src_colorspace, None)

            self.work_result_dir = worker_dir
            t1 = round(time.time())
            t_start = round(time.time())

            # 获取脸部方框
            video_first_frame = get_first_frame(video_path, pyav_src_colorspace)
            height, width = video_first_frame.shape[:2]
            face_box, start_idx = self.get_face_box(video_path, pyav_src_colorspace)
            assert face_box is not None and len(face_box) > 0, 'face not exist in video'
            t2 = round(time.time())
            # 分离音频
            audio_path = os.path.join(self.work_result_dir, "audio.wav")
            self._split_audio(video_path, audio_path)
            t3 = round(time.time())

            # 获取音频特征
            wave_vectors = self.wav2vector.get_emission3(audio_path)
            t4 = round(time.time())

            # 获取所有闭嘴区域

            mouth_areas = self._get_mouth_area(video_path, pyav_src_colorspace, face_box, start_idx)
            t5 = round(time.time())

            # 获取闭嘴帧index
            silence_frame_index_list = self._get_silence_frame_index(wave_vectors, mouth_areas, start_idx)

            t6 = round(time.time())

            x264_color_dict = PyavVideoEncoder.get_x264_color_dict(src_colorspace, color_range, color_transfer, color_primaries)
            result_video, silence_videos, silence_frames = self._generate(video_path,
                                                                          height,
                                                                          width,
                                                                          pyav_src_colorspace,
                                                                          pyav_dst_colorspace,
                                                                          x264_color_dict,
                                                                          silence_frame_index_list,
                                                                          start_idx,
                                                                          worker_dir
                                                                          )

            t7 = round(time.time())
            t_end = round(time.time())

            print(f'cost: {t_end - t_start},generate:{t7 - t6}, get_silence_index: {t6 - t5}, mouth_area:{t5 - t4},audio_vector:{t4 - t3}, split_audio:{t3 - t2}, face_box:{t2 - t1}')

            result = {
                "silence_frames": silence_frames,
                "silence_videos": silence_videos,
                "video_with_silence": result_video
            }
            return result

        except Exception as e:
            raise e


if __name__ == "__main__":
    config = CloseMouthWorkerConfig(
        wav2vector_model_path='/mnt/users/chenmuyin/closedmouth_2.0/data_enhancement/checkpoints/temp',
        yolo_model_path='/mnt/users/chenmuyin/closedmouth_2.0/data_enhancement/checkpoints/yolov5l-face.pt',
        face_segment_model_path="/mnt/users/chenmuyin/closedmouth_2.0/data_enhancement/checkpoints/79999_iter.pth",
        resnet_state_dict_file=''
    )

    worker = CloseMouthWorker(config)

    video_folder = "/mnt/users/chenmuyin/closedmouth_2.0/testdata/0824_cases"
    
    names = glob.glob(os.path.join(video_folder, '*/'))

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    for name in names:
        # video_path = "/data/52e01dba17064ffe918d65eecbaebc39.mp4"
        video_path = os.path.join(name,'video.mp4')

        result = worker.extend_silence_video(video_path, "/result")

        print(result)

"""视频编码器
author: Yin Jiakang
email: yinjiakang@xiaobing.ai
"""
import csv
import io
import math
import os
from dataclasses import dataclass
from enum import Enum
from tempfile import mktemp
from typing import Generator, List, Optional, Tuple, Union, Dict

import av
import librosa
import numpy as np
import soundfile as sf

# from talkingface.log import error_logger

from .base import BaseMultimediaEncoder


@dataclass
class libx264Options:
    """
    x264编码器支持参数，参考：x264 --fullhelp 数据
    Attributes:
        colorprim: x264编码器 --colorprim 参数。
        range: x264编码器 --range 参数。
        transfer: x264编码器 --transfer 参数。
        colormatrix: x264编码器 --colormatrix 参数。
        colorspace:  x264编码器 --colormatrix 参数所对应的pyav中的dst_colorspace参数。
    """

    colorprim = {
        "undef",
        "bt709",
        "bt470m",
        "bt470bg",
        "smpte170m",
        "smpte240m",
        "film",
        "bt2020",
        "smpte428",
        "smpte431",
        "smpte432",
    }
    range = {"auto", "tv", "pc"}
    transfer = {
        "undef",
        "bt709",
        "bt470m",
        "bt470bg",
        "smpte170m",
        "smpte240m",
        "linear",
        "log100",
        "log316",
        "iec61966-2-4",
        "bt1361e",
        "iec61966-2-1",
        "bt2020-10",
        "bt2020-12",
        "smpte2084",
        "smpte428",
        "arib-std-b67",
    }
    colormatrix = {
        "undef",
        "bt709",
        "fcc",
        "bt470bg",
        "smpte170m",
        "smpte240m",
        "GBR",
        "YCgCo",
        "bt2020nc",
        "bt2020c",
        "smpte2085",
        "chroma-derived-nc",
        "chroma-derived-c",
        "ICtCp",
    }
    colorspace = {
        "bt709": "ITU709",
        "fcc": "FCC",
        "bt470bg": "ITU601",
        "smpte170m": "SMPTE170M",
        "smpte240m": "SMPTE240M",
    }

class VideoFormat(str, Enum):
    mp4 = "mp4"
    mkv = "mkv"
    webm = "webm"
    mov = "mov"


@dataclass
class PyavVideoEncoderConfig:
    """PyavVideoEncoder的配置类
    Attributes:
        fps (str, optional): 视频的帧率. Defaults to "25".
        crf (str, optional): 恒定速率因子,控制视频大小和画质. Defaults to "20".
        video_shape (Tuple[int], optional): 视频的width * height. Defaults to (1920, 1080).
        segment_times (str, optional): _description_. Defaults to "1,2,4,8,12,16,22,28,36,44".
        video_codec (str, optional): 视频的编码译码器. Defaults to "libx264"..
        audio_codec (str, optional): 音频的编码译码器. Defaults to "aac".
    """

    fps: str = "25"
    crf: str = "20"
    video_shape: Tuple[int] = (1920, 1080)
    segment_times: str = "1,2,4,8,12,16,22,28,36,44"
    video_codec: str = "libx264"
    audio_codec: str = "aac"


@dataclass
class CustomSegPyavVideoEncoderConfig:
    """CustomSegPyavVideoEncoder的配置类
    Attributes:
        fps (str, optional): 视频的帧率. Defaults to "25".
        crf (str, optional): 恒定速率因子,控制视频大小和画质,仅支持mp4和mkv(libx264). Defaults to "20".
        bit_rate(int, optional): 视频码率,仅支持webm(libvpx). Default to 1000000000
        video_shape (Tuple[int], optional): 视频的width * height. Defaults to (1920, 1080).
        seg_duration (Tuple[int], optional): 每个seg的时长. Defaults to ().
        video_format (VideoFormat, optional): 视频的类型. Defaults to VideoFormat.mkv.
        color_space (str, optional)：视频颜色空间选项，仅libx264生效，默认为None，与原流程保持一致。
        color_range (str, optional)：视频颜色空间范围选项，仅libx264生效，默认为None，与原流程保持一致。
        color_transfer (str, optional)：视频颜色空间相关选项，仅libx264生效，默认为None，与原流程保持一致。
        color_primaries (str, optional)：视频颜色空间相关选项，仅libx264生效，默认为None，与原流程保持一致。
    """

    fps: str = "25"
    crf: str = "20"
    bit_rate: int = 1000000000
    video_shape: Tuple[int] = (1920, 1080)
    seg_duration: Tuple[int] = ()
    video_format: VideoFormat = VideoFormat.mkv
    color_space: str = None
    color_range: str = None
    color_transfer: str = None
    color_primaries: str = None


class PyavVideoEncoder(BaseMultimediaEncoder):
    def __init__(self, config: PyavVideoEncoderConfig) -> None:
        """初始化pyav_video_encoder

        Args:
            config (PyavVideoEncoderConfig): 用于初始化初始化pyav_video_encoder的配置

        """

        super().__init__()
        self.video_format = VideoFormat.mp4
        self.video_codec = config.video_codec
        self.audio_codec = config.audio_codec
        self.fps = config.fps
        self.crf = config.crf
        self.video_shape = config.video_shape
        self.segment_times = config.segment_times

        self.out_video_path = ""
        self.out_video_path_csv = self.out_video_path + ".csv"
        self.container = None
        self.video_stream = None
        self.audio_stream = None
        self.output_video_files = []
        self.final_output_video_files = None

    def init_video(self, temp_dir: str = None) -> None:
        """初始化一个新的视频

        Args:
            temp_dir (str, optional): 指定临时文件生成的目录. Defaults to None.
        """
        self.out_video_path = mktemp(suffix=f".{self.video_format}", dir=temp_dir)
        # TODO: (sulei@xiaobing.ai) csv文件的使用是否是必须的？是否有更好的解决方案需要调研后给出更合理的结论
        self.out_video_path_csv = self.out_video_path + ".csv"
        self.output_video_files = []
        self.final_output_video_files = None

        container_opts = {
            "g": "25",
            "keyint_min": "25",
            "segment_times": self.segment_times,
            "segment_list": f"{self.out_video_path_csv}",
            "segment_list_type": "csv",
            "forced-idr": "1",
            "segment_format_options": "movflags=frag_keyframe+empty_moov+omit_tfhd_offset+faststart+separate_moof+disable_chpl+default_base_moof+dash",
            "max_interleave_delta": "0",
        }

        self.container = av.open(
            f"{self.out_video_path}_%03d.{self.video_format}",
            format="segment",
            mode="w",
            options=container_opts,
        )

        self.container.flags |= (
            av.container.Flags.SHORTEST
            | av.container.Flags.NOBUFFER
            | av.container.Flags.FLUSH_PACKETS
        )

        # video stream
        self.video_stream = self.container.add_stream(
            self.video_codec, rate=str(self.fps)
        )
        self.video_stream.width = self.video_shape[0]
        self.video_stream.height = self.video_shape[1]
        video_stream_opts = {
            "g": "25",
            "keyint_min": "25",
            "force_key_frames": self.segment_times,
            "segment_times": self.segment_times,
            "preset": "ultrafast",
            "crf": str(self.crf),
            "profile:v": "main",
            "pix_fmt": "yuv420p",
        }

        if self.video_codec in ["h264_nvenc", "nvenc_h264", "nvenc_hevc", "hevc_nvenc"]:
            video_stream_opts.pop("preset")
        self.video_stream.options = video_stream_opts

        # audio stream
        self.audio_stream = self.container.add_stream(self.audio_codec, rate=24000)
        self.audio_stream.options = {
            "force_key_frames": self.segment_times,
            "segment_times": self.segment_times,
        }

    def encode_audio(self, audio: Union[str, io.BytesIO]) -> Optional[Generator]:
        """对传入的音频进行编码

        Args:
            audio (Union[str, io.BytesIO]): 视频路径或者io.BytesIO(bytes_audio)

        Returns:
            None
        """
        input_audio_container = av.open(audio)
        input_audio_stream = input_audio_container.streams.get(audio=0)[0]

        for audio_frame in input_audio_container.decode(input_audio_stream):
            audio_frame.pts = None
            for packet in self.audio_stream.encode(audio_frame):
                self.container.mux(packet)

    def encode_frames(self, frames: np.ndarray) -> Optional[Generator]:
        """对传入的图片帧数组进行编码

        Args:
            frames (np.ndarray): 图片帧数组(N, height, width, channel)

        Returns:
            None
        """

        for frame_arr in frames:
            frame = av.VideoFrame.from_ndarray(frame_arr, format="bgr24")

            for packet in self.video_stream.encode(frame):
                self.container.mux(packet)

    def end_encoding(self) -> None:
        """flush缓存区，并关闭container,并删除csv文件"""
        for packet in self.video_stream.encode():
            self.container.mux(packet)

        for packet in self.audio_stream.encode():
            self.container.mux(packet)

        self.container.close()
        self.final_output_video_files = self._collect_all_segments_generated_so_far()
        # TODO: (sulei@xiaobing.ai) 如果csv文件是必须的，则是否有更好的方案安全地删除
        # The upper-level code cannot guarantee that __del__ will be called, so we have to do clean here
        self._clean_old_csv_file()

    def _clean_old_csv_file(self) -> None:
        """清理csv文件"""
        try:
            os.remove(self.out_video_path_csv)
        except Exception as e:
            print(e)

    def _collect_all_segments_generated_so_far(self):
        video_files = []
        segment_list_file = f"{self.out_video_path_csv}"
        dir_path = os.path.dirname(segment_list_file)
        with open(segment_list_file, encoding="utf-8") as csvfile:
            video_segs = csv.reader(csvfile)
            video_files = [os.path.join(dir_path, seg[0].strip()) for seg in video_segs]
        return video_files

    def check_new_video_seg(self) -> List[str]:
        """检查是否有新的video seg生成。可以在每次encode_frames后检查，也可以仅在end_encoding后检查，一次性返回所有segs

        Returns:
            List[str]: 视频seg的绝对路径的list，如果没有新seg生成，则返回[]
        """
        segment_list_file = f"{self.out_video_path_csv}"
        if os.path.exists(segment_list_file):
            video_files = self._collect_all_segments_generated_so_far()
        else:
            if self.final_output_video_files is None:
                # 如果没有这个文件，且init_video之后还没有生成新的video-segmentation
                video_files = []
            else:
                video_files = self.final_output_video_files

        if len(video_files) == 0:
            return []

        len_2 = len(video_files)
        len_1 = len(self.output_video_files)

        new_video_files = video_files[len_1:len_2]

        self.output_video_files = video_files

        return new_video_files


class CustomSegPyavVideoEncoder(BaseMultimediaEncoder):
    def __init__(self, config: CustomSegPyavVideoEncoderConfig) -> None:

        super().__init__()
        self.video_format = config.video_format
        self.out_video_path = None
        self.fps = config.fps
        self.crf = config.crf
        self.bit_rate = config.bit_rate
        self.video_shape = config.video_shape
        self.seg_idx = 0
        self.acum_ids = 0
        self.audio_start = 0
        self.pcm_audio = None
        self.seg_duration = config.seg_duration
        self.to_output_videos = []
        self.container = None
        self.video_stream = None
        self.audio_stream = None
        self.colorspace = config.color_space
        self.color_range = config.color_range
        self.color_trc = config.color_transfer
        self.color_primaries = config.color_primaries

    def encode_audio(self, audio: Union[str, io.BytesIO]) -> Optional[Generator]:
        """不真实进行音频的encoding，仅读取音频

        Args:
            audio: 音频路径，或者io.BytesIO(bytes_audio)

        Return:
            None
        """

        self.pcm_audio, _ = librosa.core.load(audio, sr=24000)

    def encode_frames(self, frames: "np.ndarray") -> Optional[Generator]:
        """对传入的图片帧数组进行编码

        Args:
            frames (np.ndarray): 图片帧数组(N, height, width, channel)

        Returns:
            None
        """

        audio_end = self.audio_start + len(frames) * int(0.04 * 24000)

        batch_audio = self.pcm_audio[self.audio_start : audio_end]
        self.audio_start = audio_end
        pcm_audio_part = io.BytesIO()
        sf.write(pcm_audio_part, batch_audio, 24000, format="wav", subtype="PCM_16")

        for frame in frames:
            if (
                self.video_format == VideoFormat.webm
                or self.video_format == VideoFormat.mov
            ):
                frame = av.VideoFrame.from_ndarray(frame, format="bgra")
            else:
                frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
                dst_colorspace =  self.get_x264_dst_colorspace()
                if dst_colorspace is not None and self.video_stream.pix_fmt is not None:
                    frame = frame.reformat(dst_colorspace=dst_colorspace, format=self.video_stream.pix_fmt)

            for packet in self.video_stream.encode(frame):
                self.container.mux(packet)

        # 这边会有一个问题，如果每个seg的第一个batch不inference结束的话，实际上音频并不会真的进行encoding，而是一直处在block状态
        pcm_audio_part.seek(0)
        input_ = av.open(
            pcm_audio_part,
            options={"format": "s16le", "acodec": "pcm_s16le", "ac": "1", "ar": "24k"},
        )
        for frame in input_.decode():
            frame.pts = None
            for packet in self.audio_stream.encode(frame):
                self.container.mux(packet)

        self.acum_ids += len(frames)
        if (
            len(self.seg_duration) > 0
            and self.acum_ids
            >= int(self.fps)
            * self.seg_duration[min(len(self.seg_duration) - 1, self.seg_idx)]
        ):
            self._reset_video_encoder()

    def check_new_video_seg(self) -> List[str]:
        """检查是否有的新的视频seg生成

        Returns:
            List[str]: 视频seg的绝对路径的list，如果没有新seg生成，则返回[]
        """
        new_videos = self.to_output_videos
        self.to_output_videos = []
        return new_videos

    def _end_current_seg_encoding(self):
        """结束当前seg的encoding"""
        for packet in self.video_stream.encode():
            self.container.mux(packet)

        for packet in self.audio_stream.encode(None):
            self.container.mux(packet)

        self.container.close()

        if os.path.exists(self.current_seg_video_path):
            self.to_output_videos.append(self.current_seg_video_path)

        self.acum_ids = 0

    def end_encoding(self):
        """结束当前视频的encoding"""
        self._end_current_seg_encoding()
        self.seg_idx = 0

    def _reset_video_encoder(self):
        """当前seg encoding结束，seg_idx加一，初始化一个新的seg"""
        self._end_current_seg_encoding()
        self.seg_idx += 1
        self._init_new_seg()

    def init_video(self, temp_dir: str = None):
        """初始化一个新的视频

        初始化新状态

        初始化一个新视频的name，以便生成seg
        f'{video_name}_{seg_idx}.{video_format}'
        'xxxx_001.mkv','xxx_002.mkv'

        初始化一个新的seg

        Args:
            temp_dir (str, optional): 指定临时文件生成的目录. Defaults to None.
        """
        self.seg_idx = 0
        self.acum_ids = 0
        self.to_output_videos = []

        self.out_video_path = mktemp(suffix=f".{self.video_format}", dir=temp_dir)

        self._init_new_seg()

    def _init_new_seg(self):
        """初始化seg的container和stream"""
        new_opt = {"movflags": "frag_keyframe+empty_moov+default_base_moof+faststart"}

        self.current_seg_video_path = (
            f"{self.out_video_path}_{str(self.seg_idx).zfill(3)}.{self.video_format}"
        )
        self.container = av.open(self.current_seg_video_path, mode="w", options=new_opt)

        if self.video_format == VideoFormat.webm:
            webm_bit_rate = (
                math.ceil(
                    self.video_shape[0] * self.video_shape[1] * 1.0 / (1920 * 1080)
                )
                * 8000000
            )
            self.video_stream = self.container.add_stream("libvpx", rate=self.fps)
            self.video_stream.pix_fmt = "yuva420p"
            self.video_stream.qscale = 1
            self.video_stream.bit_rate = webm_bit_rate
            self.video_stream.options = {
                "threads": "12",
                "auto-alt-ref": "0",
                "bufsize": f"{int(webm_bit_rate) * 4}",
                "quality": "good",
                "cpu-used": "5",
                "speed": "2",
                "crf": self.crf,
                "row-mt": "1",
            }
            self.audio_stream = self.container.add_stream("libopus", rate=24000)

        elif self.video_format == VideoFormat.mov:
            self.video_stream = self.container.add_stream("prores_ks", rate=self.fps)
            self.video_stream.bit_rate = 100
            self.video_stream.pix_fmt = "yuva444p10le"
            self.video_stream.qscale = 32

            self.video_stream.options = {"bits_per_mb": "8192"}

            self.audio_stream = self.container.add_stream("pcm_s16le", rate=24000)

        elif self.video_format == VideoFormat.mkv:
            self.video_stream = self.container.add_stream("libx264", rate=self.fps)  #
            self.video_stream.options = {
                "preset": "ultrafast",
                "crf": self.crf,
                "profile:v": "main",
                "pix_fmt": "yuv420p",
                **self.get_x264_color_dict(),
            }
            self.audio_stream = self.container.add_stream("pcm_s16le", rate=24000)

        else:  # VideoFormat.mp4
            self.video_stream = self.container.add_stream("libx264", rate=self.fps)  #
            self.video_stream.options = {
                "preset": "ultrafast",
                "crf": self.crf,
                "profile:v": "main",
                "pix_fmt": "yuv420p",
                **self.get_x264_color_dict(),
            }
            self.audio_stream = self.container.add_stream("aac", rate=24000)
            self.audio_stream.bit_rate = 384000

        self.video_stream.width = self.video_shape[0]
        self.video_stream.height = self.video_shape[1]

    def get_x264_color_dict(self)->Dict:
        """
        生成一个x264使用的与colorspace相关的配置
        """
        color = dict()
        if self.colorspace in libx264Options.colormatrix:
            color["colorspace"] = self.colorspace
        if self.color_range in libx264Options.range:
            color["color_range"] = self.color_range
        if self.color_trc in libx264Options.transfer:
            color["color_trc"] = self.color_trc
        if self.color_primaries in libx264Options.colorprim:
            color["color_primaries"] = self.color_primaries
        return color

    def get_x264_dst_colorspace(self) -> str:
        """
        获取pyav使用的dst_colorspace参数
        """
        return (
            libx264Options.colorspace[self.colorspace]
            if self.colorspace in libx264Options.colorspace
            else None
        )

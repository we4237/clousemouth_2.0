import os

import av

from talkingface.codecs.video import libx264Options


class PyavVideoEncoder:
    def __init__(self, video_path, height, width, x264_color_dict, fps=25) -> None:
        """初始化pyav_video_encoder

        Args:
            video_path (): 输出video 地址

        """

        self.container = av.open(video_path, mode='w')
        self.video_format = self._get_format(video_path)
        self._video_init(height, width, x264_color_dict, fps)
        self._audio_init()
        self.audio_resampler = av.audio.resampler.AudioResampler('s16', 'mono', 16000)

    def _get_format(self, video_path):
        return os.path.basename(video_path).split('.')[-1]

    def _video_init(self, height, width, x264_color_dict, fps=25):
        if self.video_format == "mp4":
            self.video_stream = self.container.add_stream("libx264", rate=fps)
            self.video_stream.options = {
                "preset": "ultrafast",
                "crf": "18",
                "profile:v": "main",
                "pix_fmt": "yuv420p",
                **x264_color_dict
            }
        elif self.video_format == "webm":
            self.video_stream = self.container.add_stream("libvpx", rate=fps)
            self.video_stream.pix_fmt = "yuva420p"

            self.video_stream.options = {
                "threads": "12",
                "auto-alt-ref": "0",
                "quality": "good",
                "cpu-used": "5",
                "speed": "2",
                "row-mt": "1",
                # **self.get_vpx_color_dict(),
            }
        self.video_stream.width, self.video_stream.height = width, height

    def _audio_init(self):
        if self.video_format == "mp4":
            self.audio_stream = self.container.add_stream("aac", rate=16000)
        elif self.video_format == "webm":
            self.audio_stream = self.container.add_stream("libopus", rate=16000)

    def encode_video_frame(self, src_frame, dst_colorspace="ITU709"):
        if self.video_format == "mp4":
            frame = av.VideoFrame.from_ndarray(src_frame, format="bgr24")
            frame = frame.reformat(dst_colorspace=dst_colorspace, format="yuv420p")
        elif self.video_format == "webm":
            frame = av.VideoFrame.from_ndarray(src_frame, format="bgra")
        for packet in self.video_stream.encode(frame):
            self.container.mux(packet)

    def encode_audio_frame(self, src_frame, sample_rate=16000):
        frame = av.AudioFrame.from_ndarray(src_frame, format='s16p', layout='mono')
        frame.sample_rate = sample_rate
        try:
            frame.pts = None
            for packet in self.audio_stream.encode(frame):
                self.container.mux(packet)
        except Exception as e:
            print(e)

    def encode_finish(self):
        for packet in self.video_stream.encode():
            self.container.mux(packet)
        for packet in self.audio_stream.encode():
            self.container.mux(packet)
        self.container.close()

    @staticmethod
    def get_x264_color_dict(dst_colorspace, color_range, color_trc, color_primaries):
        """
        生成一个x264使用的与colorspace相关的配置
        """
        color = dict()
        if dst_colorspace in libx264Options.colormatrix:
            color["colorspace"] = dst_colorspace
        if color_range in libx264Options.range:
            color["color_range"] = color_range
        if color_trc in libx264Options.transfer:
            color["color_trc"] = color_trc
        if color_primaries in libx264Options.colorprim:
            color["color_primaries"] = color_primaries
        return color

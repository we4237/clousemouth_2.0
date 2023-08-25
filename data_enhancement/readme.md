插入静音说明文档

1 简介

功能：此函数为自动寻找语句断点的闭嘴位置插入静音
解决问题：实现录制数据不用每句话停顿3S

数据存放样例：
    [project]/dataset/video1/video.mp4
    针对每一个视频，中间生成的内容存放在video.mp4对应的文件夹下

2 环境

    无额外环境,talkingface可运行

3 算法原理
    
    a 寻找视频中面部位置(face_detector.detect)
    b 得到视频中面部中牙齿部分面积作为参考(_get_mouth_area)
    c 找到闭嘴且静音位置(_get_silence_frame_index)

4. 使用样例
    config = CloseMouthWorkerConfig(
        wav2vector_model_path='/code/talking_face_v3/core/talkingface/resources/models/wav2lip/default/wav2vec2/wav2vec2_phoneme',
        yolo_model_path='/code/talking_face_v3/core/talkingface/resources/models/face_detector/yolov5l-face_landmark.pt',
        face_segment_model_path="/code/talking_face_v3/core/talkingface/resources/models/face_segmenter/79999_iter.pth",
        resnet_state_dict_file='/code/talking_face_v3/core/talkingface/resources/models/face_segmenter/resnet18-5c106cde.pth'
    )

    worker = CloseMouthWorker(config)
    video_path = "/data/data_enhancement/dataset/video1/video.mp4"
    result = worker.process(video_path, "/data/data_enhancement/dataset/video1/result")

    保存两部分内容：
    1）video_with_silence.mp4 插帧后的视频
    x）可选可视化中间结果：silence_frame_{index}.png (静音的帧图片和对应的index)
                       silence_part_{index}.mp4 (静音片段和对应的index)
   
5. 颜色空间
   目前支持 "bt709", "fcc", "bt470bg", "smpte170m", "smpte240m"， 默认值为 "bt709"
import gc
import os
import math
from fractions import Fraction

import av
import numpy as np

__all__ = ['read_vedio', 'save_vedio']
_CALLED_TIMES = 0
_GC_COLLECTION_INTERVAL = 10

# def read_vedio(filename):
#     """Reads the contents of file to a Vedio instance.
        
#     Args:
#         filename: str, vedio absolute path.
#     Returns:
#         a Vedio instance.
#     """
#     return Vedio(filename)


def format_convert(filename_in, filename_out):
    """transform vedio file format.
    
    Args
        filename_in: input vedio path.
        filename_out: output vedio path.
    """
    input_ = av.open(filename_in)
    output = av.open(filename_out, "w")

    in_stream = input_.streams.video[0]
    out_stream = output.add_stream(template=in_stream)

    for packet in input_.demux(in_stream):
        if packet.dts is None:
            continue
        packet.stream = out_stream
        output.mux(packet)

    input_.close()
    output.close()


class Vedio():
    def __init__(self, filename=None, params=None, data=None, **kwargs):
        """Reads the contents of file to a Vedio instance.
        
        Args:
            filename: str, vedio absolute path.
        """
        self._filename = filename
        self.vedio_params = params
        self._data = data
        self._stream = []
        if filename is not None:
            self._file = av.open(filename, 'rb')
            vedio = self._file.streams.video[0]
            self.vedio_size = self._file.size
            self.vedio_duration = self._file.duration/1000000
            self.vedio_bitrate = self._file.bit_rate
            self.vedio_frames = vedio.frames
            self.vedio_shape = (vedio.codec_context.width, vedio.codec_context.height)
            self.vedio_params = {'vedio_size':self.vedio_size, 
                                 'vedio_duration':self.vedio_duration, 
                                 'vedio_bitrate':self.vedio_bitrate,
                                 'vedio_frames':self.vedio_frames,
                                 'vedio_shape':self.vedio_shape,
                                }
        if data is not None:
            self.vedio_fps = self.vedio_params['fps']
            self.vedio_duration = self.vedio_params['vedio_duration']
#             self.vedio_bitrate = self._file.bit_rate
            self.vedio_frames = self.vedio_params['vedio_frames']
            self.vedio_shape = self.vedio_params['vedio_shape']
                    
    def close(self):
        """Close the underlying file."""
        self._file.close()
        
    def add_stream(self, stream):
        self._stream.append(stream)
        
        
def save_vedio(filename, video_array, video_fps, video_codec="libx264", options=None,
               audio_array=None, audio_fps=None, audio_codec=None, audio_options=None):
    """
    Writes a 4d array in [H, W, C, N] format in a video file
    Args:
        filename: path where the video will be saved
        video_array: array containing the individual frames, as a uint8 array in [H, W, C, N] format,
            or array of list or Image of list.
        video_fps: video frames per second
        video_codec: the name of the video codec, i.e. "libx264", "h264", etc.
        options: dictionary containing options to be passed into the PyAV video stream
        audio_array: array[C, N] containing the audio, where C is the number of channels
            and N is the number of samples
        audio_fps: audio sample rate, typically 44100 or 48000
        audio_codec: the name of the audio codec, i.e. "mp3", "aac", etc.
        audio_options: dictionary containing options to be passed into the PyAV audio stream
    """
    with av.open(filename, mode="w") as container:
        stream = container.add_stream(video_codec, rate=int(video_fps))
        stream.pix_fmt = "yuv420p" if video_codec != "libx264rgb" else "rgb24"
        stream.options = options or {}
        if isinstance(vedio_array, np.ndarray):
            assert vedio_array.ndim==4, '`video_array` type error.'
            stream.width = video_array.shape[1]
            stream.height = video_array.shape[0]
            for i in range(vedio_array.shape[-1]):
                frame = av.VideoFrame.from_ndarray(vedio_array[:,:,:,i], format="rgb24")
                frame.pict_type = "NONE"
                for packet in stream.encode(frame):
                    container.mux(packet)
        else:
            if isinstance(vedio_array, list):
                if isinstance(vedio_array[0], np.ndarray):
                    assert vedio_array[0].ndim==3, '`video_array` type error.'
                    stream.width = video_array[0].shape[1]
                    stream.height = video_array[0].shape[0]
                    for i in vedio_array:
                        frame = av.VideoFrame.from_ndarray(i, format="rgb24")
                        frame.pict_type = "NONE"
                        for packet in stream.encode(frame):
                            container.mux(packet)
                else:
                    try:
                        stream.width = video_array[0].width
                        stream.height = video_array[0].height
                        for i in vedio_array:
                            frame = av.VideoFrame.from_image(i)
                            frame.pict_type = "NONE"
                            for packet in stream.encode(frame):
                                container.mux(packet)
                    except:
                        raise ValueError('`video_array` type error.')
            else:
                raise ValueError('`video_array` type error.')
        for packet in stream.encode():
            container.mux(packet)

        if audio_array is not None:
            audio_format_dtypes = {
                "dbl": "<f8",
                "dblp": "<f8",
                "flt": "<f4",
                "fltp": "<f4",
                "s16": "<i2",
                "s16p": "<i2",
                "s32": "<i4",
                "s32p": "<i4",
                "u8": "u1",
                "u8p": "u1",
            }
            a_stream = container.add_stream(audio_codec, rate=int(audio_fps))
            a_stream.options = audio_options or {}

            num_channels = audio_array.shape[0]
            audio_layout = "stereo" if num_channels > 1 else "mono"
            audio_sample_fmt = container.streams.audio[0].format.name

            dtype = np.dtype(audio_format_dtypes[audio_sample_fmt])

            frame = av.AudioFrame.from_ndarray(audio_array.astype(dtype), format=audio_sample_fmt, layout=audio_layout)
            frame.sample_rate = int(audio_fps)

            for packet in a_stream.encode(frame):
                container.mux(packet)

            for packet in a_stream.encode():
                container.mux(packet)

        
def _read_stream(container, start_offset, end_offset, stream, stream_name):
    global _CALLED_TIMES, _GC_COLLECTION_INTERVAL
    _CALLED_TIMES += 1
    if _CALLED_TIMES % _GC_COLLECTION_INTERVAL == _GC_COLLECTION_INTERVAL - 1:
        gc.collect()

    start_offset = int(math.floor(start_offset * (1 / stream.time_base)))
    if end_offset != float("inf"):
        end_offset = int(math.ceil(end_offset * (1 / stream.time_base)))

    frames = {}
    should_buffer = True
    max_buffer_size = 5
    if stream.type == "video":
        extradata = stream.codec_context.extradata
        if extradata and b"DivX" in extradata:
            pos = extradata.find(b"DivX")
            d = extradata[pos:]
            o = re.search(rb"DivX(\d+)Build(\d+)(\w)", d)
            if o is None:
                o = re.search(rb"DivX(\d+)b(\d+)(\w)", d)
            if o is not None:
                should_buffer = o.group(3) == b"p"
    seek_offset = max(start_offset - 1, 0)
    if should_buffer:
        seek_offset = max(seek_offset - max_buffer_size, 0)
    try:
        container.seek(seek_offset, any_frame=False, backward=True, stream=stream)
    except av.AVError:
        return []
    buffer_count = 0
    try:
        for _idx, frame in enumerate(container.decode(**stream_name)):
            frames[frame.pts] = frame
            if frame.pts >= end_offset:
                if should_buffer and buffer_count < max_buffer_size:
                    buffer_count += 1
                    continue
                break
    except av.AVError:
        pass
    result = [frames[i] for i in sorted(frames) if start_offset <= frames[i].pts <= end_offset]
    if len(frames) > 0 and start_offset > 0 and start_offset not in frames:
        preceding_frames = [i for i in frames if i < start_offset]
        if len(preceding_frames) > 0:
            first_frame_pts = max(preceding_frames)
            result.insert(0, frames[first_frame_pts])
    return result


def read_vedio(filename, start_pts=0, end_pts=None, vedio_format="HWCN", vedio_type=np.uint8):
    """Reads a video from a file, returning both the video frames and the audio frames
    
    Args:
        filename: path to the video file
        start_pts: float, The start presentation time of the video
        end_pts: float, The end presentation time.
        output_format: The format of the output video shape. Can be "HWCN" (default), "Image" return pillow instance.
    Returns:
        vedio: array[H, W, C, N]: the `N` video frames.
        audio: array[C, L]): the audio frames, where `C` is the number of channels and `L` is the number of points.
        info: metadata for the video and audio.
    """
    vedio_format = vedio_format.upper()
    if vedio_format!='IMAGE':
        if len([i for i in set(vedio_format) if i in 'HWCN'])!=len(vedio_format):
            raise ValueError(f"`vedio_format` should be 'HWCN' or 'TCHW' e.g., got {vedio_format}.")

    if not os.path.exists(filename):
        raise RuntimeError(f"File not found: {filename}")

    if end_pts is None:
        end_pts = float("inf")

    if end_pts < start_pts:
        raise ValueError("end_pts should be larger than start_pts")

    info = {'filename':filename}
    video_frames = []
    audio_frames = []
    audio_timebase = Fraction(0, 1)

    try:
        with av.open(filename, metadata_errors="ignore") as container:
            if container.streams.audio:
                audio_timebase = container.streams.audio[0].time_base
            if container.streams.video:
                video_frames = _read_stream(container, start_pts, end_pts, container.streams.video[0], {"video": 0})
                video_fps = container.streams.video[0].average_rate
                if video_fps is not None:
                    info["video_fps"]  = float(video_fps)
                info["vedio_duration"] = container.duration/1000000
                info["vedio_bitrate"]  = container.bit_rate
                info["vedio_frames"]   = container.streams.video[0].frames
                info["vedio_shape"]    = (container.streams.video[0].codec_context.height, 
                                          container.streams.video[0].codec_context.width)
            if container.streams.audio:
                audio_frames = _read_stream(container, start_pts, end_pts, container.streams.audio[0], {"audio": 0})
                info["audio_fps"]       = container.streams.audio[0].rate
                info["audio_channel"]   = container.streams.audio[0].codec_context.channels
                info["audio_frames"]     = container.streams.audio[0].duration
    except av.AVError:
        pass

    if vedio_format!='IMAGE':
        vframes_list = [frame.to_rgb().to_ndarray() for frame in video_frames]
        if vframes_list:
            vframes = np.stack(vframes_list).astype(vedio_type)
        else:
            vframes = np.empty((0, 1, 1, 3), dtype=np.uint8)
        transpose = {'N':0, 'H':1, 'W':2, 'C':3}
        if vedio_format!='NHWC':
            vframes = vframes.transpose(tuple(transpose[i] for i in vedio_format))
    else:
        vframes_list = [frame.to_image() for frame in video_frames]
    
    aframes_list = [frame.to_ndarray() for frame in audio_frames]
    if aframes_list:
        aframes = np.concatenate(aframes_list, 1)
        start_pts = int(math.floor(start_pts * (1 / audio_timebase)))
        if end_pts != float("inf"):
            end_pts = int(math.ceil(end_pts * (1 / audio_timebase)))
        start, end = audio_frames[0].pts, audio_frames[-1].pts
        total_aframes = aframes.shape[1]
        step_per_aframe = (end - start + 1) / total_aframes
        s_idx = 0
        e_idx = total_aframes
        if start < start_pts:
            s_idx = int((start_pts - start) / step_per_aframe)
        if end > end_pts:
            e_idx = int((end_pts - end) / step_per_aframe)
        aframes = aframes[:, s_idx:e_idx]
    else:
        aframes = np.empty((1, 0), dtype=np.float32)
    return {'vedio':vframes, 'audio':aframes, 'info':info}



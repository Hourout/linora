import gc
import io
import warnings

import av
import numpy as np

__all__ = ['read_vedio', 'read_vedio_timestamps', 'save_vedio', 'VedioStream']



# def format_convert(filename_in, filename_out):
#     """transform vedio file format.
    
#     Args
#         filename_in: input vedio path.
#         filename_out: output vedio path.
#     """
#     input_ = av.open(filename_in)
#     output = av.open(filename_out, "w")

#     in_stream = input_.streams.video[0]
#     out_stream = output.add_stream(template=in_stream)
#     for packet in input_.demux(in_stream):
#         if packet.dts is None:
#             continue
#         packet.stream = out_stream
#         output.mux(packet)
    
#     in_audio = input_.streams.audio[0]
#     out_audio = output.add_stream(template=in_audio)
#     for packet in input_.demux(in_audio):
#         if packet.dts is None:
#             continue
#         packet.stream = out_audio
#         output.mux(packet)

#     input_.close()
#     output.close()

# def format_convert(filename_in, filename_out):
#     """transform vedio file format.
    
#     Args
#         filename_in: input vedio path.
#         filename_out: output vedio path.
#     """
#     input_ = av.open(filename_in)
#     output = av.open(filename_out, "w")

#     in_stream = input_.streams.video[0]
#     out_stream = output.add_stream(template=in_stream)

#     for packet in input_.demux(in_stream):
#         if packet.dts is None:
#             continue
#         packet.stream = out_stream
#         output.mux(packet)

#     input_.close()
#     output.close()
        
        
def save_vedio(filename, video_array, video_fps, video_codec="libx264", options=None,
               audio_array=None, audio_fps=None, audio_codec=None, audio_options=None):
    """Writes a 4d array in [H, W, C, N] format in a video file
    
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
    container = av.open(filename, mode="w")
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
    container.close()

        
def read_vedio(src, start_sec=0, end_sec=np.inf, vedio_format="HWCN", vedio_type=np.uint8, audio_format='CL'):
    """Reads a video from a file, returning both the video frames and the audio frames.
    
    Args:
        src: path to the video file or bytes.
        start_sec: float, The start presentation time of the video.
        end_sec: float, The end presentation time of the video.
        vedio_format: The format of the output video shape. Can be "HWCN" (default), "Image" return pillow instance.
        vedio_type: The dtype of the output video.
        audio_format: The format of the output audio shape. 'CL' or 'LC' e.g. 
    Returns:
        vedio: array[H, W, C, N]: the `N` video frames.
        audio: array[C, L]): the audio frames, where `C` is the number of channels and `L` is the number of points.
        metadata: metadata for the video and audio.
    """
    vedio_format = vedio_format.upper()
    if vedio_format!='IMAGE':
        if len([i for i in set(vedio_format) if i in 'HWCN'])!=len(vedio_format):
            raise ValueError(f"`vedio_format` should be 'HWCN' or 'NCHW' e.g., got {vedio_format}.")

    if end_sec < start_sec:
        raise ValueError("end_sec should be larger than start_sec")

    if isinstance(src, bytes):
        src = io.BytesIO(src)
        info = {'filename':''}
    else:
        info = {'filename':src}

    try:
        with av.open(src, metadata_errors="ignore") as container:
            if container.streams.video:
                container.streams.video[0].thread_type = "AUTO" 
                video_fps = container.streams.video[0].average_rate
                if video_fps is not None:
                    info["video_fps"]  = float(video_fps)
                info["vedio_duration"] = container.streams.video[0].duration*container.streams.video[0].time_base
                info["vedio_bitrate"]  = container.bit_rate
                info["vedio_frames"]   = container.streams.video[0].frames
                info["vedio_shape"]    = (container.streams.video[0].codec_context.height, 
                                          container.streams.video[0].codec_context.width)
                
                vframes = [frame for frame in container.decode(video=0) if start_sec<float(frame.pts*frame.time_base)<end_sec]
                if vedio_format!='IMAGE':
                    if vframes:
                        vframes = [frame.to_rgb().to_ndarray() for frame in vframes]
                        vframes = np.stack(vframes).astype(vedio_type)
                    else:
                        vframes = np.empty((0, 1, 1, 3), dtype=np.uint8)
                    if vedio_format!='NHWC':
                        transpose = {'N':0, 'H':1, 'W':2, 'C':3}
                        vframes = vframes.transpose(tuple(transpose[i] for i in vedio_format))
                else:
                    vframes = [frame.to_image() for frame in vframes]
        with av.open(src, metadata_errors="ignore") as container:
            if container.streams.audio:
                info["audio_fps"]       = container.streams.audio[0].rate
                info["audio_channel"]   = container.streams.audio[0].codec_context.channels
                info["audio_frames"]    = container.streams.audio[0].duration
                info["audio_duration"]  = container.streams.audio[0].duration*container.streams.audio[0].time_base
                
                aframes = [frame.to_ndarray() for frame in container.decode(audio=0) if start_sec<float(frame.pts*frame.time_base)<end_sec]
                if aframes:
                    aframes = np.concatenate(aframes, axis=1, dtype=np.float32)
                    if aframes.shape[0]!=info["audio_channel"]:
                        aframes = aframes.reshape(-1, info["audio_channel"]).T
                else:
                    aframes = np.empty((1, 0), dtype=np.float32)
                if audio_format!='CL':
                    transpose = {'C':0, 'L':1}
                    aframes = aframes.transpose(tuple(transpose[i] for i in audio_format))
    except av.AVError:
        pass
    gc.collect()
    return {'vedio':vframes, 'audio':aframes, 'metadata':info}


def read_vedio_timestamps(filename, pts_unit="sec"):
    """List the video frames timestamps.
    Note that the function decodes the whole video frame-by-frame.
    
    Args:
        filename: path to the video file
        pts_unit: unit in which timestamp values will be returned
            either 'pts' or 'sec'. Defaults to 'pts'.
    Returns:
        pts (List[int] if pts_unit = 'pts', List[Fraction] if pts_unit = 'sec'):
            presentation timestamps for each one of the frames in the video.
        video_fps (float, optional): the frame rate for the video
    """
    video_fps = None
    pts = []

    try:
        container = av.open(filename, metadata_errors="ignore")
        if container.streams.video:
            video_stream = container.streams.video[0]
            video_time_base = video_stream.time_base
            try:
                extradata = container.streams[0].codec_context.extradata
                if extradata is None:
                    pts = [x.pts for x in container.decode(video=0) if x.pts is not None]
                elif b"Lavc" in extradata:
                    pts = [x.pts for x in container.demux(video=0) if x.pts is not None]
                else:
                    pts = [x.pts for x in container.decode(video=0) if x.pts is not None]
            except av.AVError:
                warnings.warn(f"Failed decoding frames for file {filename}")
            video_fps = float(video_stream.average_rate)
        container.close()
    except av.AVError as e:
        msg = f"Failed to open container for {filename}; Caught error: {e}"
        warnings.warn(msg, RuntimeWarning)

    pts.sort()
    if pts_unit == "sec":
        pts = [x * video_time_base for x in pts]

    return {'timestamps':pts, 'video_fps':video_fps}


class VedioStream():
    def __init__(self, src, batch=1, data_format="HWCN", start_sec=0, end_sec=np.inf, dtype=np.uint8):
        """Converts a Vedio instance to a Numpy array.
    
        - if data_format is 'CL' or 'LC' e.g. return audio array.
          batch=4096, dtype=np.float32
        
        Args:
            src: input vedio path or bytes.
            batch: each iter batch number.
            data_format: array data format, eg.'HWCN', 'CHWN'. 
                'image' return pillow instance.
                'CL' or 'LC' e.g. return audio array.
            start_sec: float, The start presentation time of the video.
            end_sec: float, The end presentation time of the video.
            dtype: Dtype to use for the returned array.
                if 'CL' or 'LC' e.g. dtype is np.float32
        Returns:
            A Numpy array iterator.
        """
        self._batch = int(batch)
        self._data_format = data_format.upper()
        if 'L' in self._data_format:
            if len([i for i in set(self._data_format) if i in 'CL'])!=len(self._data_format):
                raise ValueError(f"`data_format` should be 'CL' or 'LC' e.g., got {data_format}.")
            self._dtype = np.float32
            if int(batch)==1:
                self._batch = 4096
        elif self._data_format!='IMAGE':
            if len([i for i in set(self._data_format) if i in 'HWCN'])!=len(self._data_format):
                raise ValueError(f"`data_format` should be 'HWCN' or 'NCHW' e.g., got {data_format}.")
            self._dtype = dtype
            
        if end_sec < start_sec:
            raise ValueError("end_sec should be larger than start_sec")
        self._end_sec = end_sec
        self._start_sec = start_sec
        
        if isinstance(src, bytes):
            src = io.BytesIO(src)
            self.metadata = {'filename':''}
        else:
            self.metadata = {'filename':src}
            
        self._container = av.open(src, metadata_errors="ignore")
        try:
            if self._container.streams.video:
                video_fps = self._container.streams.video[0].average_rate
                if video_fps is not None:
                    self.metadata["vedio_fps"]  = float(video_fps)
                self.metadata["vedio_duration"] = float(self._container.streams.video[0].duration*self._container.streams.video[0].time_base)
                self.metadata["vedio_bitrate"]  = self._container.bit_rate
                self.metadata["vedio_frames"]   = self._container.streams.video[0].frames
                self.metadata["vedio_shape"]    = (self._container.streams.video[0].codec_context.height, 
                                                   self._container.streams.video[0].codec_context.width)
            if self._container.streams.audio:
                self.metadata["audio_fps"]       = self._container.streams.audio[0].rate
                self.metadata["audio_channel"]   = self._container.streams.audio[0].codec_context.channels
                self.metadata["audio_frames"]    = self._container.streams.audio[0].duration
                self.metadata["audio_duration"]  = float(self.metadata["audio_frames"]*self._container.streams.audio[0].time_base)
        except av.AVError:
            pass
        if 'L' in self._data_format:
            try:
                self._aframes = np.concatenate([frame.to_ndarray() for frame in self._container.decode(audio=0)],
                                               axis=1, dtype=np.float32)
                if self._aframes.shape[0]!=self.metadata["audio_channel"]:
                    self._aframes = self._aframes.reshape(-1, self.metadata["audio_channel"]).T
                mix = self._container.streams.audio[0].start_time/1000000
                interval = float(self.metadata["audio_duration"])-mix
                self._batch_start_time = max(self._start_sec, mix)
                self._batch_end_time = min(self._end_sec, float(self.metadata["audio_duration"]))
                self._batch_time = self._batch/self._aframes.shape[1]*interval
                start = (self._batch_start_time-mix)/interval
                end = (self._batch_end_time-mix)/interval
                self._aframes = self._aframes[:, int(self._aframes.shape[1]*start):int(self._aframes.shape[1]*end)]
            except av.AVError:
                self._aframes = np.empty((1, 0), dtype=np.float32)
        else:
            self._c = self._container.decode(video=0)

    def __next__(self):
        try:
            batch_array = []
            batch_pts = []
            count = 0
            if 'L' in self._data_format:
                if not self._aframes.shape[1]:
                    self._container.close()
                    raise StopIteration
                batch_array = self._aframes[:,:self._batch]
                self._aframes = self._aframes[:,self._batch:]
                batch_pts.append(self._batch_start_time)
                if batch_array.shape[1]<self._batch:
                    batch_pts.append(self._batch_end_time)
                else:
                    self._batch_start_time += self._batch_time
                    batch_pts.append(self._batch_start_time)
                if self._data_format!='CL':
                    transpose = {'C':0, 'L':1}
                    batch_array = batch_array.transpose(tuple(transpose[i] for i in self._data_format))
            elif self._data_format=='IMAGE':
                try: 
                    while 1:
                        frame = next(self._c)
                        time_stamp = float(frame.pts*frame.time_base)
                        if self._start_sec<time_stamp<self._end_sec:                            
                            batch_array.append(frame.to_image())
                            batch_pts.append(time_stamp)
                            count += 1
                        if count==self._batch:
                            break
                except:
                    pass
                if not batch_array:
                    self._container.close()
                    raise StopIteration
            else:
                while 1:
                    try:
                        frame = next(self._c)
                        time_stamp = float(frame.pts*frame.time_base)
                        if self._start_sec<time_stamp<self._end_sec:
                            batch_array.append(frame.to_rgb().to_ndarray())
                            batch_pts.append(time_stamp)
                            count += 1
                        if count==self._batch:
                            break
                    except:
                        break
                if not batch_array:
                    self._container.close()
                    raise StopIteration
                batch_array = np.stack(batch_array).astype(self._dtype)
                if self._data_format!='NHWC':
                    transpose = {'N':0, 'H':1, 'W':2, 'C':3}
                    batch_array = batch_array.transpose(tuple(transpose[i] for i in self._data_format))
        except av.error.EOFError:
            self._container.close()
            raise StopIteration
        return {"data": batch_array, "pts": batch_pts}

    def __iter__(self):
        return self
    



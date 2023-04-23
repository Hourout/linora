import av

__all__ = ['read_vedio', 'save_video']


def read_vedio(filename):
    """Reads the contents of file to a Vedio instance.
        
    Args:
        filename: str, vedio absolute path.
    Returns:
        a Vedio instance.
    """
    return Vedio(filename)


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
        
        
def save_video(filename, video_array, video_fps, video_codec="libx264", options=None,
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

        
        

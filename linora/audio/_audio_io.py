import gc
import io
import wave

import av
import numpy as np

__all__ = ['read_audio', 'save_audio', 'AudioStream']


def save_audio(filename, audio_array, audio_fps=44100, audio_codec=None, audio_options=None):
    """
    Writes a 2d array in [C, L] format in a audio file
    Args:
        filename: path where the video will be saved.
        audio_array: array[C, N] containing the audio, where C is the number of channels
            and N is the number of samples
        audio_fps: audio sample rate, typically 44100 or 48000
        audio_codec: the name of the audio codec, i.e. "mp3", "aac", etc.
        audio_options: dictionary containing options to be passed into the PyAV audio stream
    """
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
    audio_codec_type = {'wav':'pcm_s16le'}
    if audio_codec is None:
        audio_codec = audio_codec_type.get(filename.split('.')[-1].lower(), filename.split('.')[-1].lower())
    if audio_codec in ['pcm_s16le']:
        f = wave.open(filename, "wb")
        f.setnchannels(len(audio_array.shape))
        f.setsampwidth(2)
        f.setframerate(int(audio_fps))
        f.setnframes(max(audio_array.shape))
        f.writeframes(audio_array.T.astype(np.int16).flat[:].tobytes())
        f.close()
        return 
        
        
    audio_layout = "stereo" if audio_array.shape[0] > 1 else "mono"
    container = av.open(filename, mode="w")
    stream = container.add_stream(audio_codec, rate=int(audio_fps))
    stream.options = audio_options or {}
    audio_sample_fmt = stream.codec_context.format.name
    dtype = np.dtype(audio_format_dtypes[audio_sample_fmt])
#     print(stream.codec_context.format.name, dtype)

#             frame = av.AudioFrame.from_ndarray(np.expand_dims(audio_array.T.flat[:].astype(dtype), 0),
#                                                format=audio_sample_fmt, layout=audio_layout)
    frame = av.AudioFrame.from_ndarray(audio_array.astype(dtype), 
                                       format=audio_sample_fmt, layout=audio_layout)
    frame.pts = None
    frame.sample_rate = int(audio_fps)

    for packet in stream.encode(frame):
        container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()

# def save_audio(filename, audio, file_format=None):
#     """Saves an audio stored as a Numpy array to a path or file object.
        
#     Args
#         filename: Path or file object.
#         audio: a numpy array.
#         file_format: Optional file format override. If omitted, the
#             format to use is determined from the filename extension.
#     """
#     if file_format is None:
#         file_format = filename.split('.')[-1]
#     if file_format == "ogg":
#         file_format = 'libopus'#"libvorbis"
#     if file_format in ['WAV', 'wav']:
#         return save_wav(filename, audio.audio_data, params=audio.audio_params)
#     container = av.open(filename, 'w')
#     stream = container.add_stream(file_format)
#     frame = av.AudioFrame.from_ndarray(audio.audio_data, format='fltp')
#     frame.pts = None
#     frame.sample_rate = audio.audio_framerate
#     for packet in stream.encode(frame):
#         container.mux(packet)
#     container.close()

# def save_wav(filename, audio, params=None):
#     """Saves an audio stored as a Numpy array to a path or file object.
    
#     Now only support audio file format with '.wav'.
    
#     Args
#         filename: Path or file object.
#         audio: a numpy array.
#         params: {'audio_channel':None, 'audio_width': 2, 'audio_framerate': 44100, 'audio_frame':None}
#     """
#     param = {'audio_channel':None, 'audio_width': 2, 'audio_framerate': 44100, 'audio_frame':None}
#     if params is not None:
#         for i in params:
#             param[i] = params[i]
#     if param['audio_channel'] is None:
#         param['audio_channel'] = len(audio.shape)
#     if param['audio_frame'] is None:
#         param['audio_frame'] = max(audio.shape)
#     param['audio_frame'] = min(param['audio_frame'], max(audio.shape))
#     if filename[-4:] in ['.WAV', '.wav']:
#         f = wave.open(filename, "wb")
#         f.setnchannels(param['audio_channel'])
#         f.setsampwidth(param['audio_width'])
#         f.setframerate(param['audio_framerate'])
#         f.setnframes(param['audio_frame'])
#         f.writeframes(audio.T.astype(np.int16)[:param['audio_frame']].flat[:].tobytes())
#         f.close()
#     else:
#         raise ValueError(f'Not support audio file format with {filename}')
        

class AudioWAV():
    def __init__(self, filename):
        """Reads the contents of file to a Audio instance.
        
        Args:
            filename: str, audio absolute path.
        """
        try:
            self._file = wave.open(filename, 'rb')
        except wave.Error:
            raise ValueError('File is not a WAV file.')
            
        self.audio_channel = self._file.getnchannels()
        self.audio_width = self._file.getsampwidth()
        self.audio_framerate = self._file.getframerate()
        self.audio_frame = self._file.getnframes()
        self.audio_duration = self.audio_frame / self.audio_framerate
        self.audio_params = {'audio_channel':self.audio_channel, 'audio_width':self.audio_width,
                             'audio_framerate':self.audio_framerate, 'audio_frame':self.audio_frame,
                             'audio_duration':self.audio_duration,
                             'audio_type':'wav', 'audio_name':filename[:-4]}
        
        if self.audio_width not in (1, 2, 3, 4):
            self.close()
            raise ValueError('The file uses an unsupported bit width.')

    def close(self):
        """Close the underlying file."""
        self._file.close()
        
    @property
    def audio_data(self):
        """Reads and returns all frames of audio"""
        data = self._get_data(self.audio_frame)
        self._file.rewind()
        return data
    
    def stream(self, nframes=1024):
        """Generates blocks of PCM data found in the file."""
        while True:
            data = self._get_data(nframes)
            if not data:
                break
            yield data
    
    def _get_data(self, n):
        str_data  = self._file.readframes(n)
        return np.frombuffer(str_data, dtype=np.int16).reshape(-1,self.audio_channel).T

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __iter__(self):
        return self.stream()
    
    
class Audio():
    def __init__(self, filename='', data=None, audio_params=None):
        """Reads the contents of file to a Vedio instance.
        
        Args:
            filename: str, vedio absolute path.
        """
        self._data = data
        self.audio_params = audio_params
        if filename:
            self._file = av.open(filename, 'r')
            audio = self._file.streams.audio[0]

            self.audio_size      = self._file.size
            self.audio_channel   = audio.codec_context.channels
    #         self.audio_width = self._file.getsampwidth()
            self.audio_framerate = audio.codec_context.sample_rate
            self.audio_frame     = audio.duration
            self.audio_duration  = self._file.duration/1000000
#             self.close()
        self.audio_params = {'audio_channel':self.audio_channel, 
                            'audio_duration':self.audio_duration,
                            'audio_framerate':self.audio_framerate, 
                            'audio_frame':self.audio_frame,
                            'audio_size':self.audio_size, 'audio_path':filename}
        
    def close(self):
        """Close the underlying file."""
        self._file.close()

    @property
    def audio_data(self):
        """Reads and returns all frames of audio"""
        if self._data is None:
            with av.open(self.audio_params['audio_path'], 'r') as container:
                self._data = np.concatenate([frame.to_ndarray() for frame in container.decode(audio=0)], axis=1)
        return self._data.reshape(-1,self.audio_channel).T
    
    def stream(self):
        """Generates blocks of PCM data found in the file."""
        if self._data is None:
            with av.open(self.audio_params['audio_path'], 'r') as container:
                self._data = [frame.to_ndarray() for frame in container.decode(audio=0)]
        for frame in self._data:
            yield frame.reshape(-1,self.audio_channel).T
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __iter__(self):
        return self.stream()


    
def read_audio(src, start_sec=0, end_sec=np.inf, audio_format="CL"):
    """Reads a audio from a file, returning both the audio frames and audio metadata.
    
    Args:
        src: path to the audio file or bytes.
        start_sec: float, The start presentation time of the audio.
        end_sec: float, The end presentation time of the audio.
        audio_format: The format of the output audio shape. 'CL' or 'LC' e.g. 
    Returns:
        audio: array[C, L]): the audio frames, where `C` is the number of channels and `L` is the number of points.
        metadata: metadata for the audio.
    """
    audio_format = audio_format.upper()
    if len([i for i in set(audio_format) if i in 'CL'])!=len(audio_format):
        raise ValueError(f"`data_format` should be 'CL' or 'LC' e.g., got {audio_format}.")

    if end_sec < start_sec:
        raise ValueError("end_sec should be larger than start_sec")

    if isinstance(src, bytes):
        src = io.BytesIO(src)
        info = {'filename':''}
    else:
        info = {'filename':src}

    try:
        container = av.open(src, metadata_errors="ignore")
        if container.streams.audio:
            info["audio_fps"]       = container.streams.audio[0].rate
            info["audio_channel"]   = container.streams.audio[0].codec_context.channels
            aframes = [frame.to_ndarray() for frame in container.decode(audio=0) if start_sec<float(frame.pts*frame.time_base)<end_sec]
            if aframes:
                aframes = np.concatenate(aframes, axis=1, dtype=np.float32)
                if aframes.shape[0]!=info["audio_channel"]:
                    aframes = aframes.reshape(-1, info["audio_channel"]).T
            else:
                aframes = np.empty((1, 0), dtype=np.float32)
            info["audio_frames"]    = container.streams.audio[0].duration
            if info["audio_frames"] is None:
                info["audio_frames"] = aframes.shape[1]
            info["audio_duration"]  = info["audio_frames"]*float(container.streams.audio[0].time_base)
        transpose = {'C':0, 'L':1}
        if audio_format!='CL':
            aframes = aframes.transpose(tuple(transpose[i] for i in audio_format))
        container.close()
    except av.AVError:
        pass
    gc.collect()
    return {'audio':aframes, 'metadata':info}


class AudioStream():
    def __init__(self, src, batch=4096, data_format="CL", start_sec=0, end_sec=np.inf):
        """Converts a Audio instance to a Numpy array.
        
        Args:
            src: input audio path or bytes.
            batch: each iter batch number.
            data_format: array data format, 'CL' or 'LC' e.g. 
            start_sec: float, The start presentation time of the audio.
            end_sec: float, The end presentation time of the audio.
        Returns:
            A Numpy array iterator.
        """
        self._batch = int(batch)
        self._data_format = data_format.upper()
        if len([i for i in set(self._data_format) if i in 'CL'])!=len(self._data_format):
            raise ValueError(f"`data_format` should be 'CL' or 'LC' e.g., got {data_format}.")
        self._dtype = np.float32
            
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
            if self._container.streams.audio:
                self.metadata["audio_fps"]       = self._container.streams.audio[0].rate
                self.metadata["audio_channel"]   = self._container.streams.audio[0].codec_context.channels
                self.metadata["audio_frames"]    = self._container.streams.audio[0].duration
                self.metadata["audio_duration"]  = self.metadata["audio_frames"]*self._container.streams.audio[0].time_base
                
                self._aframes = np.concatenate([frame.to_ndarray() for frame in self._container.decode(audio=0)], 
                                               axis=1, dtype=np.float32)
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

    def __next__(self):
        try:
            batch_pts = []
            count = 0
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
            transpose = {'C':0, 'L':1}
            if self._data_format!='CL':
                batch_array = batch_array.transpose(tuple(transpose[i] for i in self._data_format))
        except av.error.EOFError:
            self._container.close()
            raise StopIteration
        return {"data": batch_array, "pts": batch_pts}

    def __iter__(self):
        return self
import wave

import numpy as np

__all__ = ['read_audio', 'save_audio']


def read_audio(filename):
    """Reads the contents of file to a Audio instance.
        
        Args:
            filename: str, audio absolute path.
        Returns:
            a Audio instance.
        """
    if filename[-4:] in ['.WAV', '.wav']:
        return AudioWAV(filename)        
    else:
        raise ValueError(f'Not support audio file format with {filename}')

        
def save_audio(filename, audio, params=None):
    """Saves an audio stored as a Numpy array to a path or file object.
    
    Args
        filename: Path or file object.
        audio: a numpy array.
        params: {'audio_channel':None, 'audio_width': 2, 'audio_framerate': 44100, 'audio_frame':None}
    """
    param = {'audio_channel':None, 'audio_width': 2, 'audio_framerate': 44100, 'audio_frame':None}
    if params is not None:
        for i in params:
            param[i] = params[i]
    if param['audio_channel'] is None:
        param['audio_channel'] = len(audio.shape)
    if param['audio_frame'] is None:
        param['audio_frame'] = max(audio.shape)
    param['audio_frame'] = min(param['audio_frame'], max(audio.shape))
    if filename[-4:] in ['.WAV', '.wav']:
        f = wave.open(filename, "wb")
        f.setnchannels(param['audio_channel'])
        f.setsampwidth(param['audio_width'])
        f.setframerate(param['audio_framerate'])
        f.setnframes(param['audio_frame'])
        f.writeframes(audio.T.astype(np.int16)[:param['audio_frame']].flat[:].tobytes())
        f.close()
    else:
        raise ValueError(f'Not support audio file format with {filename}')
        

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
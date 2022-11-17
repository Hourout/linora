import wave

import numpy as np

__all__ = ['read_audio']


def read_audio(filename):
    """Reads the contents of file to a Audio instance.
        
        Args:
            filename: str, audio absolute path.
        Returns:
            a Audio instance.
        """
    if filename.endswith('.wav'):
        return AudioWAV(filename)
#     elif filename.endswith('.mp3'):
        
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
                             'audio_duration':self.audio_duration}
        
        if self.audio_width not in (1, 2, 3, 4):
            self.close()
            raise ValueError('The file uses an unsupported bit width.')
        if self.audio_channel not in (1, 2):
            self.close()
            raise ValueError('The file uses an unsupported channel.')

    def close(self):
        """Close the underlying file."""
        self._file.close()
        
    def read_data(self):
        """Reads and returns all frames of audio"""
        data = self._get_data(self.audio_frame)
        self._file.rewind()
        return data
    
    def read_stream(self, nframes=1024):
        """Generates blocks of PCM data found in the file."""
        while True:
            data = self._get_data(nframes)
            if not data:
                break
            yield data
    
    def _get_data(self, n):
        str_data  = self._file.readframes(n)
        wave_data = np.frombuffer(str_data, dtype=np.short)
        if self.audio_channel==1:
            wave_data.shape = -1,1
        else:
            wave_data.shape = -1,2
        wave_data = wave_data.T
        return wave_data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __iter__(self):
        return self.read_stream()

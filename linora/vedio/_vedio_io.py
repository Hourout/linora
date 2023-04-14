import av

__all__ = ['read_vedio']


def read_vedio(filename):
    """Reads the contents of file to a Vedio instance.
        
    Args:
        filename: str, vedio absolute path.
    Returns:
        a Vedio instance.
    """
    return Vedio(filename)

class Vedio():
    def __init__(self, filename):
        """Reads the contents of file to a Vedio instance.
        
        Args:
            filename: str, vedio absolute path.
        """
        self._filename = filename
        self._file = av.open(filename, 'rb')
        vedio = self._file.streams.video[0]
        self.vedio_size = self._file.size
        self.vedio_duration = self._file.duration/1000000
        self.vedio_bitrate = self._file.bit_rate
        self.vedio_shape = (vedio.codec_context.width, vedio.codec_context.height)
        self.vedio_params = {'vedio_size':self.vedio_size, 
                             'vedio_duration':self.vedio_duration, 
                             'vedio_bitrate':self.vedio_bitrate,
                             'vedio_shape':self.vedio_shape,
                             
                            }
        
    def close(self):
        """Close the underlying file."""
        self._file.close()


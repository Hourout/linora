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


def save_vedio(filename, vedio):
    """Saves an vedio stored as a Numpy array to a path or file object.
    
    Args
        filename: Path or file object.
        vedio: A Vedio instance.
    """
    if vedio._data is not None:
        container = av.open(filename, mode="w")
        stream = container.add_stream("mpeg4", rate=vedio.vedio_params['vedio_fps'])
        stream.width = vedio.vedio_params['vedio_shape'][0]
        stream.height = vedio.vedio_params['vedio_shape'][1]
        stream.pix_fmt = "yuv420p"

        for i in range(vedio._data.shape[-1]):
            frame = av.VideoFrame.from_ndarray(vedio._data[:,:,:,i], format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

        # Flush stream
        for packet in stream.encode():
            container.mux(packet)

        # Close the file
        container.close()
    else:
        input_ = av.open(vedio._filename)
        output = av.open(filename, "w")
        
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


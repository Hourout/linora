from linora.image._image_util import list_images

__all__ = ['list_vedios', 'vedio_to_array', 'vedio_to_stream']


def list_vedios(directory, file_format=('mpeg', 'mpg', 'dat', 'mp4', 'avi', 'mov', 'asf', 'wmv', 'mkv', 'flv', 'rmvb')):
    """Lists all vedios in a directory, including all subdirectories.
    
    Args:
        directory: string, absolute path to the directory.
        file_format: tuple of strings or single string, extensions of the vedios.
    Returns:
        a list of vedio paths.
    """
    return list_images(directory, file_format=file_format)


def vedio_to_stream(vedio, batch=1, data_format='HWCN', dtype='float32'):
    """Converts a Vedio instance to a Numpy array.
    
    Args:
        image: Vedio instance.
        batch: each iter batch number.
        data_format: array data format, eg.'HWCN', 'CHWN'. 'image' return pillow instance.
        dtype: Dtype to use for the returned array.
    Returns:
        A Numpy array iterator.
    """
    container = av.open(vedio._filename)
    container.streams.video[0].thread_type = "AUTO"
    stream = container.streams.video[0]
    batch_array = []
    
    if data_format=='image':
        for r, frame in enumerate(container.decode(stream), start=1):
            batch_array.append(frame.to_image())
            if not r%batch:
                yield batch_array
                batch_array = []
        if batch_array:
            yield batch_array
    else:
        transpose = {'H':0, 'W':1, 'C':2, 'N':3}
        for r, frame in enumerate(container.decode(stream), start=1):
            batch_array.append(np.expand_dims(frame.to_ndarray(format="rgb24"), axis=-1))
            if not r%batch:
                array = np.concatenate(batch_array, axis=-1, dtype=dtype).transpose(tuple(transpose[i] for i in data_format))
                batch_array = []
                yield array
        if batch_array:
            yield np.concatenate(batch_array, axis=-1, dtype=dtype).transpose(tuple(transpose[i] for i in data_format))
    container.close()

    
def vedio_to_array(vedio, data_format='HWCN', dtype='float32'):
    """Converts a Vedio instance to a Numpy array.
    
    Args:
        image: Vedio instance.
        data_format: array data format, eg.'HWCN', 'CHWN'.
        dtype: Dtype to use for the returned array.
    Returns:
        A Numpy array.
    """
    container = av.open(vedio._filename)
    container.streams.video[0].thread_type = "AUTO"
    stream = container.streams.video[0]
    transpose = {'H':0, 'W':1, 'C':2, 'N':3}
    data = np.concatenate([np.expand_dims(frame.to_ndarray(format="rgb24"), axis=-1) for frame in container.decode(stream)],
                          axis=-1, dtype=dtype)
    if data_format!='HWCN':
        data = data.transpose(tuple(transpose[i] for i in data_format))
    container.close()
    return data
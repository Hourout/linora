from linora.image._image_util import list_images

__all__ = ['list_vedios', 'vedio_to_array', 'vedio_to_stream']


def list_audios(directory, file_format=('mpeg', 'mpg', 'dat', 'mp4', 'avi', 'mov', 'asf', 'wmv', 'mkv', 'flv', 'rmvb')):
    """Lists all vedios in a directory, including all subdirectories.
    
    Args:
        directory: string, absolute path to the directory.
        file_format: tuple of strings or single string, extensions of the vedios.
    Returns:
        a list of vedio paths.
    """
    return list_images(directory, file_format=file_format)


def vedio_to_stream(vedio, batch, data_format='HWCN'):
    container = av.open(vedio._filename)
    container.streams.video[0].thread_type = "AUTO"
    stream = container.streams.video[0]
    transpose = {'H':0, 'W':1, 'C':2, 'N':3}
    batch_array = []
    for r, frame in enumerate(container.decode(stream), start=1):
        batch_array.append(np.expand_dims(frame.to_ndarray(format="rgb24"), axis=-1))
        if not r%batch:
            array = np.concatenate(batch_array, axis=-1).transpose(tuple(transpose[i] for i in data_format))
            batch_array = []
            yield array
    if batch_array:
        yield np.concatenate(batch_array, axis=-1).transpose(tuple(transpose[i] for i in data_format))
    container.close()

    
def vedio_to_array(vedio, data_format='HWCN'):
    container = av.open(vedio._filename)
    container.streams.video[0].thread_type = "AUTO"
    stream = container.streams.video[0]
    transpose = {'H':0, 'W':1, 'C':2, 'N':3}
    data = np.concatenate([np.expand_dims(frame.to_ndarray(format="rgb24"), axis=-1) for frame in container.decode(stream)], axis=-1)
    if data_format!='HWCN':
        data = data.transpose(tuple(transpose[i] for i in data_format))
    container.close()
    return data
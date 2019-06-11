import tensorflow as tf

__all__ = ['read_image', 'save_image']

def read_image(filename, channel=0, image_format='mix', **kwarg):
    """Convenience function for read image type one of `bmp`, `gif`, `jpeg`, `jpg`, and `png`.
    
    Detects whether an image is a BMP, GIF, JPEG, JPG, or PNG, and performs the
    appropriate operation to convert the input bytes `string` into a `Tensor`
    of type `dtype`.
    
    Note: `gif` returns a 4-D array `[num_frames, height, width, 3]`, as
    opposed to `bmp`, `jpeg`, `jpg` and `png`, which return 3-D
    arrays `[height, width, num_channels]`. Make sure to take this into account
    when constructing your graph if you are intermixing GIF files with BMP, JPEG, JPG,
    and/or PNG files.
    Args:
        filename: 0-D `string`. image absolute path.
        channels: An optional `int`. Defaults to `0`. Number of color channels for
                  the decoded image. 1 for `grayscale` and 3 for `rgb`.
        image_format: 0-D `string`. image format type one of `bmp`, `gif`, `jpeg`,
                      `jpg`, `png` and `mix`. `mix` mean contains many types image format.
    Returns:
        `Tensor` with type uint8 and shape `[height, width, num_channels]` for
        BMP, JPEG, and PNG images and shape `[num_frames, height, width, 3]` for
        GIF images.
    Raises:
        ValueError: On incorrect number of channels.
    """
    assert channel in [0, 1, 3], 'channel should be one of [0, 1, 3].'
    image = tf.io.read_file(filename)
    if image_format=='png':
        image = tf.io.decode_png(image, channel)
    elif image_format=='bmp':
        image = tf.io.decode_bmp(image, channel)
    elif image_format=='gif':
        image = tf.io.decode_gif(image)
    elif image_format in ["jpg", "jpeg"]:
        image = tf.io.decode_jpeg(image, channel)
    elif image_format=='mix':
        image = tf.io.decode_image(image)
    else:
        raise ValueError('image_format should be one of "mix", "jpg", "jpeg", "png", "gif", "bmp".')
    return image if kwarg else image.numpy()

def save_image(image, filename):
    """Writes image to the file at input filename. 
    
    Args:
        image:    A Tensor of type string. scalar. The content to be written to the output file.
        filename: A string. scalar. The name of the file to which we write the contents.
    Raises:
        ValueError: If `filename` is not in `[`jpg`, `jpeg`, `png`]`.
    """
    if filename.split('.')[-1] in ['jpg', 'jpeg']:
        tf.io.write_file(filename, tf.io.encode_jpeg(image))
    elif filename.split('.')[-1]=='png':
        tf.io.write_file(filename, tf.image.encode_png(image))
    else:
        raise ValueError('filename should be one of [`jpg`, `jpeg`, `png`].')

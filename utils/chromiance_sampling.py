import numpy as np

def downsample_chromiance_component(converted_image):
    """
    Downsample chrominance components (4:2:0 subsampling).
    
    Args:
        converted_image: YCrCb image (H, W, 3)
    
    Returns:
        y_channel, cr_channel_downsampled, cb_channel_downsampled
    """
    factor = 2
    h, w, c = converted_image.shape

    y_channel = converted_image[:, :, 0]
    cr_channel = converted_image[:, :, 1]
    cb_channel = converted_image[:, :, 2]

    cr_channel_downsampled = cr_channel.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))
    cb_channel_downsampled = cb_channel.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))

    return y_channel, cr_channel_downsampled, cb_channel_downsampled


def upsample_chrominance_component(received_image):
    """
    Upsample chrominance components from 4:2:0 to 4:4:4.
    
    Args:
        received_image: List of [y_channel, cr_channel_downsampled, cb_channel_downsampled]
    
    Returns:
        YCrCb image (H, W, 3)
    """
    factor = 2

    y_channel = received_image[0]
    cr_channel_downsampled = received_image[1]
    cb_channel_downsampled = received_image[2]

    h, w = y_channel.shape

    cr_channel_upsampled = np.repeat(np.repeat(cr_channel_downsampled, factor, axis=0), factor, axis=1)
    cb_channel_upsampled = np.repeat(np.repeat(cb_channel_downsampled, factor, axis=0), factor, axis=1)

    cr_channel_upsampled = cr_channel_upsampled[:h, :w]
    cb_channel_upsampled = cb_channel_upsampled[:h, :w]

    upsampled_image = np.stack((y_channel, cr_channel_upsampled, cb_channel_upsampled), axis=2)

    return upsampled_image



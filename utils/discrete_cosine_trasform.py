import numpy as np
import scipy.fftpack as fftpack

def level_shift(image):
    y_channel = image[:, :, 0] - 128
    cr_channel = image[:, :, 1] - 128
    cb_channel = image[:, :, 2] - 128
    shifted_image = np.stack((y_channel, cr_channel, cb_channel), axis=2)
    return shifted_image

def inverse_level_shift(shifted_image):
    y_channel = shifted_image[:, :, 0] + 128
    cr_channel = shifted_image[:, :, 1] + 128
    cb_channel = shifted_image[:, :, 2] + 128
    image = np.stack((y_channel, cr_channel, cb_channel), axis=2)
    return image


def apply_block_dct(channel, block_size=8):
    h, w = channel.shape
    dct_channel = np.zeros((h, w))
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = channel[i:i+block_size, j:j+block_size]
            # Apply 2D DCT to this 8x8 block
            dct_block = fftpack.dct(fftpack.dct(block.T, norm='ortho').T, norm='ortho')
            dct_channel[i:i+block_size, j:j+block_size] = dct_block
    return dct_channel

def apply_dct(input_image):
    y_channel = input_image[:, :, 0]
    cr_channel = input_image[:, :, 1]
    cb_channel = input_image[:, :, 2]

    dct_y = apply_block_dct(y_channel)
    dct_cr = apply_block_dct(cr_channel)
    dct_cb = apply_block_dct(cb_channel)

    return dct_y, dct_cr, dct_cb

def apply_block_inverse_dct(dct_channel, block_size=8):
    h, w = dct_channel.shape
    idct_channel = np.zeros((h, w))
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = dct_channel[i:i+block_size, j:j+block_size]
            # Apply 2D inverse DCT to this 8x8 block
            idct_block = fftpack.idct(fftpack.idct(block.T, norm='ortho').T, norm='ortho')
            idct_channel[i:i+block_size, j:j+block_size] = idct_block
    return idct_channel

def apply_inverse_dct(dct_image):
    dct_y = dct_image[0]
    dct_cr = dct_image[1]
    dct_cb = dct_image[2]

    idct_y = apply_block_inverse_dct(dct_y)
    idct_cr = apply_block_inverse_dct(dct_cr)
    idct_cb = apply_block_inverse_dct(dct_cb)

    # Stack the channels back into a 3D image
    idct_image = np.stack((idct_y, idct_cr, idct_cb), axis=2)

    return idct_image
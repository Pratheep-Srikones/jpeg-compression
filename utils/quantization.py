import numpy as np

# Base quantization tables (standard JPEG tables)
BASE_LUMINANCE_TABLE = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99
]

BASE_CHROMINANCE_TABLE = [
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99
]


def get_quantization_table(component='luminance', quality=85):
    """
    Get quantization table scaled by quality factor.
    
    Args:
        component: 'luminance' or 'chrominance'
        quality: Quality factor from 1-100 (higher = better quality, less compression)
                 50 = baseline, 85 = high quality, 95 = very high quality
    
    Returns:
        8x8 quantization table
    """
    if component == 'luminance':
        base_table = np.array(BASE_LUMINANCE_TABLE).reshape((8, 8))
    elif component == 'chrominance':
        base_table = np.array(BASE_CHROMINANCE_TABLE).reshape((8, 8))
    else:
        raise ValueError("Component must be either 'luminance' or 'chrominance'.")
    
    # Scale table based on quality
    # Quality > 50: reduce quantization (better quality)
    # Quality < 50: increase quantization (more compression)
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality
    
    # Apply scaling and ensure minimum value of 1
    scaled_table = np.floor((base_table * scale + 50) / 100)
    scaled_table = np.maximum(scaled_table, 1)  # Minimum value of 1
    
    return scaled_table.astype(np.int32)


def quantize_block(dct_block, component='luminance', quality=85):
    """
    Quantize a DCT block.
    
    Args:
        dct_block: 8x8 DCT coefficient block
        component: 'luminance' or 'chrominance'
        quality: Quality factor (1-100, higher = better quality)
    
    Returns:
        Quantized block
    """
    quant_table = get_quantization_table(component, quality)
    quantized_block = np.round(dct_block / quant_table).astype(np.int32)
    return quantized_block

def quantize_image(dct_image):
    dct_y = dct_image[0]
    dct_cr = dct_image[1]
    dct_cb = dct_image[2]

    h, w = dct_y.shape
    quantized_y = np.zeros((h, w), dtype=np.int32)
    quantized_cr = np.zeros((h, w), dtype=np.int32)
    quantized_cb = np.zeros((h, w), dtype=np.int32)

    for i in range(0, h, 8):
        for j in range(0, w, 8):
            quantized_y[i:i+8, j:j+8] = quantize_block(dct_y[i:i+8, j:j+8], component='luminance')
            quantized_cr[i:i+8, j:j+8] = quantize_block(dct_cr[i:i+8, j:j+8], component='chrominance')
            quantized_cb[i:i+8, j:j+8] = quantize_block(dct_cb[i:i+8, j:j+8], component='chrominance')

    return quantized_y, quantized_cr, quantized_cb


def dequantize_block(quantized_block, component='luminance', quality=85):
    """
    Dequantize a quantized block.
    
    Args:
        quantized_block: 8x8 quantized coefficient block
        component: 'luminance' or 'chrominance'
        quality: Quality factor (1-100, must match quantization quality)
    
    Returns:
        Dequantized block
    """
    quant_table = get_quantization_table(component, quality)
    dequantized_block = (quantized_block * quant_table).astype(np.float32)
    return dequantized_block

def dequantize_image(quantized_image):
    quantized_y = quantized_image[0]
    quantized_cr = quantized_image[1]
    quantized_cb = quantized_image[2]

    h, w = quantized_y.shape
    dequantized_y = np.zeros((h, w), dtype=np.float32)
    dequantized_cr = np.zeros((h, w), dtype=np.float32)
    dequantized_cb = np.zeros((h, w), dtype=np.float32)

    for i in range(0, h, 8):
        for j in range(0, w, 8):
            dequantized_y[i:i+8, j:j+8] = dequantize_block(quantized_y[i:i+8, j:j+8], component='luminance')
            dequantized_cr[i:i+8, j:j+8] = dequantize_block(quantized_cr[i:i+8, j:j+8], component='chrominance')
            dequantized_cb[i:i+8, j:j+8] = dequantize_block(quantized_cb[i:i+8, j:j+8], component='chrominance')

    return dequantized_y, dequantized_cr, dequantized_cb






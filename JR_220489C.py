"""
JPEG Image Compression and Decompression Implementation
Author: JR_220489C
All-in-one implementation with complete JPEG compression pipeline
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
import heapq


# ============================================================================
# COLOR SPACE CONVERSION FUNCTIONS
# ============================================================================

def rgb_to_ycrcb(image):
    """
    Convert RGB image to YCrCb color space with float precision.
    
    Args:
        image: RGB image as numpy array (H, W, 3) with values 0-255
    
    Returns:
        YCrCb image as float array (H, W, 3)
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a 3D array with 3 channels (RGB).")
    
    # Convert to float to maintain precision
    image_float = image.astype(np.float32)
    
    # Define the transformation matrix from RGB to YCrCb
    transformation_matrix = np.array([[ 0.299,     0.587,     0.114    ],
                                       [-0.168736, -0.331264,  0.5      ],
                                       [ 0.5,      -0.418688, -0.081312 ]])
    shift = np.array([0, 128, 128])

    # Reshape the image to a 2D array of pixels
    h, w, c = image_float.shape
    flat_image = image_float.reshape(-1, 3)
    # Apply the transformation
    ycrcb_flat = np.dot(flat_image, transformation_matrix.T) + shift

    # Reshape back to the original image shape
    ycrcb_image = ycrcb_flat.reshape(h, w, c)
    # Keep as float32 for processing, round to maintain precision
    return np.round(ycrcb_image).astype(np.float32)


def ycrcb_to_rgb(image):
    """
    Convert YCrCb image to RGB color space.
    
    Args:
        image: YCrCb image as numpy array (H, W, 3)
    
    Returns:
        RGB image as uint8 array (H, W, 3) with values 0-255
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a 3D array with 3 channels (YCrCb).")
    
    # Convert to float if not already
    image_float = image.astype(np.float32)
    
    # Define the inverse transformation matrix from YCrCb to RGB
    transformation_matrix = np.array([[1.0,  0.0,       1.402   ],
                                       [1.0, -0.344136, -0.714136],
                                       [1.0,  1.772,     0.0     ]])
    shift = np.array([0, -128, -128])
    
    # Reshape the image to a 2D array of pixels
    h, w, c = image_float.shape
    flat_image = image_float.reshape(-1, 3)
    # Apply the inverse transformation
    rgb_flat = np.dot(flat_image + shift, transformation_matrix.T)
    # Reshape back to the original image shape
    rgb_image = rgb_flat.reshape(h, w, c)
    # Clip values to valid range [0, 255] and convert to uint8
    rgb_image = np.clip(np.round(rgb_image), 0, 255)
    return rgb_image.astype(np.uint8)


# ============================================================================
# CHROMINANCE SAMPLING FUNCTIONS
# ============================================================================

def downsample_chromiance_component(converted_image):
    """
    Downsample chrominance components by factor of 2 (4:2:0 subsampling).
    
    Args:
        converted_image: YCrCb image (H, W, 3)
    
    Returns:
        Tuple of (Y channel, downsampled Cr channel, downsampled Cb channel)
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
    Upsample chrominance components by factor of 2.
    
    Args:
        received_image: List of [Y channel, Cr channel, Cb channel]
    
    Returns:
        Upsampled YCrCb image (H, W, 3)
    """
    factor = 2

    y_channel = received_image[0]
    cb_channel_downsampled = received_image[1]
    cr_channel_downsampled = received_image[2]

    h, w = y_channel.shape

    cb_channel_upsampled = np.repeat(np.repeat(cb_channel_downsampled, factor, axis=0), factor, axis=1)
    cr_channel_upsampled = np.repeat(np.repeat(cr_channel_downsampled, factor, axis=0), factor, axis=1)

    cb_channel_upsampled = cb_channel_upsampled[:h, :w]
    cr_channel_upsampled = cr_channel_upsampled[:h, :w]

    upsampled_image = np.stack((y_channel, cb_channel_upsampled, cr_channel_upsampled), axis=2)

    return upsampled_image


# ============================================================================
# DCT (DISCRETE COSINE TRANSFORM) FUNCTIONS
# ============================================================================

def apply_block_dct(channel, block_size=8):
    """
    Apply 2D DCT to 8x8 blocks of a channel.
    
    Args:
        channel: 2D numpy array
        block_size: Size of blocks (default: 8)
    
    Returns:
        DCT transformed channel
    """
    h, w = channel.shape
    dct_channel = np.zeros((h, w))
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = channel[i:i+block_size, j:j+block_size]
            # Apply 2D DCT to this 8x8 block
            dct_block = fftpack.dct(fftpack.dct(block.T, norm='ortho').T, norm='ortho')
            dct_channel[i:i+block_size, j:j+block_size] = dct_block
    return dct_channel


def apply_block_inverse_dct(dct_channel, block_size=8):
    """
    Apply inverse 2D DCT to 8x8 blocks of a channel.
    
    Args:
        dct_channel: DCT transformed 2D numpy array
        block_size: Size of blocks (default: 8)
    
    Returns:
        Reconstructed channel
    """
    h, w = dct_channel.shape
    idct_channel = np.zeros((h, w))
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = dct_channel[i:i+block_size, j:j+block_size]
            # Apply 2D inverse DCT to this 8x8 block
            idct_block = fftpack.idct(fftpack.idct(block.T, norm='ortho').T, norm='ortho')
            idct_channel[i:i+block_size, j:j+block_size] = idct_block
    return idct_channel


# ============================================================================
# QUANTIZATION FUNCTIONS
# ============================================================================

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
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality
    
    # Apply scaling and ensure minimum value of 1
    scaled_table = np.floor((base_table * scale + 50) / 100)
    scaled_table = np.maximum(scaled_table, 1)
    
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


def quantize_channel_separately(dct_channel, component='luminance', quality=85):
    """
    Quantize a DCT channel using the appropriate quantization table.
    
    Args:
        dct_channel: 2D numpy array of DCT coefficients
        component: 'luminance' or 'chrominance'
        quality: Quality factor (1-100, higher = better quality)
    
    Returns:
        Quantized channel as 2D numpy array
    """
    h, w = dct_channel.shape
    quantized = np.zeros((h, w), dtype=np.int32)
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = dct_channel[i:i+8, j:j+8]
            if block.shape == (8, 8):
                quantized[i:i+8, j:j+8] = quantize_block(block, component=component, quality=quality)
    return quantized


def dequantize_channel_separately(quantized_channel, component='luminance', quality=85):
    """
    Dequantize a quantized channel using the appropriate quantization table.
    
    Args:
        quantized_channel: 2D numpy array of quantized coefficients
        component: 'luminance' or 'chrominance'
        quality: Quality factor (1-100, must match quantization quality)
    
    Returns:
        Dequantized channel as 2D numpy array
    """
    h, w = quantized_channel.shape
    dequantized = np.zeros((h, w), dtype=np.float32)
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = quantized_channel[i:i+8, j:j+8]
            if block.shape == (8, 8):
                dequantized[i:i+8, j:j+8] = dequantize_block(block, component=component, quality=quality)
    return dequantized


# ============================================================================
# RUN-LENGTH ENCODING FUNCTIONS
# ============================================================================

def zig_zag_scan(block):
    """
    Perform zig-zag scan on an 8x8 block.
    
    Args:
        block: 8x8 numpy array
    
    Returns:
        1D array of 64 elements in zig-zag order
    """
    h, w = block.shape
    if h != 8 or w != 8:
        raise ValueError("Input block must be of size 8x8.")
    
    result = []
    for s in range(h + w - 1):
        if s % 2 == 0:
            for y in range(s + 1):
                x = s - y
                if x < w and y < h:
                    result.append(block[y, x])
        else:
            for x in range(s + 1):
                y = s - x
                if x < w and y < h:
                    result.append(block[y, x])
    return np.array(result)


def inverse_zig_zag_scan(array):
    """
    Perform inverse zig-zag scan to reconstruct 8x8 block.
    
    Args:
        array: 1D array of 64 elements
    
    Returns:
        8x8 numpy array
    """
    if len(array) != 64:
        raise ValueError("Input array must have 64 elements.")
    
    block = np.zeros((8, 8), dtype=array.dtype)
    index = 0
    for s in range(8 + 8 - 1):
        if s % 2 == 0:
            for y in range(s + 1):
                x = s - y
                if x < 8 and y < 8:
                    block[y, x] = array[index]
                    index += 1
        else:
            for x in range(s + 1):
                y = s - x
                if x < 8 and y < 8:
                    block[y, x] = array[index]
                    index += 1
    return block


def run_length_encode(array):
    """
    Apply run-length encoding to an array.
    
    Args:
        array: 1D numpy array
    
    Returns:
        List of (value, count) tuples
    """
    if len(array) == 0:
        return []
    
    encoded = []
    count = 1
    previous = array[0]
    
    for current in array[1:]:
        if current == previous:
            count += 1
        else:
            encoded.append((previous, count))
            previous = current
            count = 1
    encoded.append((previous, count))
    
    # Add EOB marker if last element is 0
    if len(encoded) > 0 and encoded[-1][0] == 0:
        encoded.append((0, 0))
    
    return encoded


def run_length_decode(encoded):
    """
    Decode run-length encoded data.
    
    Args:
        encoded: List of (value, count) tuples
    
    Returns:
        Decoded numpy array
    """
    decoded = []
    for value, count in encoded:
        # Stop at EOB marker
        if value == 0 and count == 0:
            break
        decoded.extend([value] * count)
    return np.array(decoded)


def process_channel_rle(channel):
    """
    Apply zig-zag scan and run-length encoding to an entire channel.
    
    Args:
        channel: 2D numpy array representing a channel
    
    Returns:
        List of RLE-encoded blocks
    """
    h, w = channel.shape
    encoded_blocks = []
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = channel[i:i+8, j:j+8]
            zigzag = zig_zag_scan(block)
            rle = run_length_encode(zigzag)
            encoded_blocks.append(rle)
    return encoded_blocks


def flatten_rle(rle_blocks):
    """
    Flatten RLE blocks into a single list of symbols for Huffman coding.
    
    Args:
        rle_blocks: List of RLE-encoded blocks
    
    Returns:
        Flattened list of (value, count) tuples
    """
    flattened = []
    for block in rle_blocks:
        for value, count in block:
            flattened.append((value, count))
    return flattened


def reconstruct_channel(flat_data, num_blocks, shape):
    """
    Reconstruct a channel from flattened RLE data.
    
    Args:
        flat_data: Flattened list of (value, count) tuples
        num_blocks: Number of 8x8 blocks in the channel
        shape: Target shape (h, w) of the reconstructed channel
    
    Returns:
        Reconstructed channel as 2D numpy array
    """
    h, w = shape
    channel = np.zeros((h, w), dtype=np.int32)
    
    data_idx = 0
    blocks_per_row = w // 8
    
    for block_num in range(num_blocks):
        # Extract this block's RLE data
        block_rle = []
        symbols_in_block = 0
        while data_idx < len(flat_data) and symbols_in_block < 64:
            block_rle.append(flat_data[data_idx])
            value, count = flat_data[data_idx]
            if value == 0 and count == 0:  # EOB marker
                data_idx += 1
                break
            symbols_in_block += count
            data_idx += 1
        
        # Decode this block
        zigzag = run_length_decode(block_rle)
        
        # Pad if necessary
        if len(zigzag) < 64:
            zigzag = np.pad(zigzag, (0, 64 - len(zigzag)), 'constant')
        
        # Inverse zig-zag scan
        block = inverse_zig_zag_scan(zigzag[:64])
        
        # Place block in channel
        block_row = (block_num // blocks_per_row) * 8
        block_col = (block_num % blocks_per_row) * 8
        channel[block_row:block_row+8, block_col:block_col+8] = block
    
    return channel


# ============================================================================
# HUFFMAN CODING FUNCTIONS
# ============================================================================

class HuffmanNode:
    """Node class for Huffman tree."""
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def build_frequency_dict(data):
    """
    Build frequency dictionary from data.
    
    Args:
        data: List of symbols
    
    Returns:
        Dictionary mapping symbols to frequencies
    """
    frequency = {}
    for symbol in data:
        if symbol in frequency:
            frequency[symbol] += 1
        else:
            frequency[symbol] = 1
    return frequency


def build_huffman_tree(frequency_data):
    """
    Build Huffman tree from frequency data.
    
    Args:
        frequency_data: Dictionary of symbol frequencies
    
    Returns:
        Root node of Huffman tree
    """
    heap = [HuffmanNode(symbol, freq) for symbol, freq in frequency_data.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]


def build_huffman_codes(node, prefix="", codebook=None):
    """
    Build Huffman codes from tree.
    
    Args:
        node: Root node of Huffman tree
        prefix: Current code prefix
        codebook: Dictionary to store codes
    
    Returns:
        Dictionary mapping symbols to Huffman codes
    """
    if codebook is None:
        codebook = {}
    if node is not None:
        if node.symbol is not None:
            codebook[node.symbol] = prefix
        build_huffman_codes(node.left, prefix + "0", codebook)
        build_huffman_codes(node.right, prefix + "1", codebook)
    return codebook


def huffman_encode(data):
    """
    Encode data using Huffman coding.
    
    Args:
        data: List of symbols to encode
    
    Returns:
        Encoded bitstring
    """
    frequency_data = build_frequency_dict(data)
    huffman_tree = build_huffman_tree(frequency_data)
    huffman_codes = build_huffman_codes(huffman_tree)

    encoded_data = ''.join(huffman_codes[symbol] for symbol in data)
    return encoded_data


def huffman_decode(encoded_data, huffman_codes):
    """
    Decode Huffman encoded data.
    
    Args:
        encoded_data: Encoded bitstring
        huffman_codes: Dictionary of Huffman codes
    
    Returns:
        Decoded list of symbols
    """
    reverse_codes = {v: k for k, v in huffman_codes.items()}
    current_code = ""
    decoded = []
    for bit in encoded_data:
        current_code += bit
        if current_code in reverse_codes:
            decoded.append(reverse_codes[current_code])
            current_code = ""
    if current_code != "":
        raise ValueError("Invalid encoded data: leftover bits do not map to any symbol")
    return decoded


# ============================================================================
# MAIN COMPRESSION AND DECOMPRESSION FUNCTIONS
# ============================================================================

def compress_image(image_path, quality=85, chroma_subsampling=True):
    """
    Compress an image using JPEG compression steps.
    
    Args:
        image_path: Path to the input image
        quality: Quality factor (1-100, higher = better quality, less compression)
                 Default: 85 (high quality)
        chroma_subsampling: Whether to apply 4:2:0 chrominance subsampling
        
    Returns:
        compressed_data: Dictionary containing all compressed data
        original_image: Original RGB image as numpy array
    """
    print("=" * 60)
    print("JPEG COMPRESSION PROCESS")
    print(f"Quality setting: {quality}/100")
    print(f"Chroma subsampling: {'4:2:0' if chroma_subsampling else '4:4:4 (no subsampling)'}")
    print("=" * 60)
    
    # Step 1: Load the image
    print("\n[Step 1] Loading image...")
    img = Image.open(image_path)
    original_image = np.array(img)
    print(f"   Original image shape: {original_image.shape}")
    
    # Ensure dimensions are multiples of 16
    h, w = original_image.shape[:2]
    new_h = (h // 16) * 16
    new_w = (w // 16) * 16
    if new_h != h or new_w != w:
        print(f"   Resizing to make dimensions multiples of 16: ({new_h}, {new_w})")
        img = img.resize((new_w, new_h))
        original_image = np.array(img)
    
    # Step 2: Color space conversion (RGB to YCrCb)
    print("\n[Step 2] Converting RGB to YCrCb color space...")
    ycrcb_image = rgb_to_ycrcb(original_image)
    print(f"   YCrCb image shape: {ycrcb_image.shape}")
    
    # Step 3: Chrominance downsampling
    if chroma_subsampling:
        print("\n[Step 3] Downsampling chrominance components (4:2:0)...")
        y_channel, cr_channel, cb_channel = downsample_chromiance_component(ycrcb_image)
    else:
        print("\n[Step 3] Skipping chrominance downsampling (4:4:4 - maximum quality)...")
        y_channel = ycrcb_image[:, :, 0]
        cr_channel = ycrcb_image[:, :, 1]
        cb_channel = ycrcb_image[:, :, 2]
    
    print(f"   Y channel shape: {y_channel.shape}")
    print(f"   Cr channel shape: {cr_channel.shape}")
    print(f"   Cb channel shape: {cb_channel.shape}")
    
    # Step 4: Level shift
    print("\n[Step 4] Applying level shift (centering around 0)...")
    y_shifted = y_channel.astype(np.float32) - 128
    cr_shifted = cr_channel.astype(np.float32) - 128
    cb_shifted = cb_channel.astype(np.float32) - 128
    print(f"   Shifted Y range: [{y_shifted.min():.1f}, {y_shifted.max():.1f}]")
    
    # Step 5: DCT
    print("\n[Step 5] Applying DCT to 8x8 blocks...")
    dct_y = apply_block_dct(y_shifted)
    dct_cr = apply_block_dct(cr_shifted)
    dct_cb = apply_block_dct(cb_shifted)
    print(f"   DCT Y shape: {dct_y.shape}")
    print(f"   DCT Cr shape: {dct_cr.shape}")
    print(f"   DCT Cb shape: {dct_cb.shape}")
    
    # Step 6: Quantization
    print("\n[Step 6] Quantizing DCT coefficients...")
    quantized_y = quantize_channel_separately(dct_y, component='luminance', quality=quality)
    quantized_cr = quantize_channel_separately(dct_cr, component='chrominance', quality=quality)
    quantized_cb = quantize_channel_separately(dct_cb, component='chrominance', quality=quality)
    print(f"   Non-zero coefficients in Y: {np.count_nonzero(quantized_y)}/{quantized_y.size}")
    
    # Step 7: Zig-zag scan and RLE
    print("\n[Step 7] Applying zig-zag scan and run-length encoding...")
    rle_y = process_channel_rle(quantized_y)
    rle_cr = process_channel_rle(quantized_cr)
    rle_cb = process_channel_rle(quantized_cb)
    print(f"   Number of 8x8 blocks encoded in Y: {len(rle_y)}")
    
    # Step 8: Huffman coding
    print("\n[Step 8] Applying Huffman coding...")
    flat_y = flatten_rle(rle_y)
    flat_cr = flatten_rle(rle_cr)
    flat_cb = flatten_rle(rle_cb)
    
    freq_y = build_frequency_dict(flat_y)
    tree_y = build_huffman_tree(freq_y)
    codes_y = build_huffman_codes(tree_y)
    
    freq_cr = build_frequency_dict(flat_cr)
    tree_cr = build_huffman_tree(freq_cr)
    codes_cr = build_huffman_codes(tree_cr)
    
    freq_cb = build_frequency_dict(flat_cb)
    tree_cb = build_huffman_tree(freq_cb)
    codes_cb = build_huffman_codes(tree_cb)
    
    huffman_y = huffman_encode(flat_y)
    huffman_cr = huffman_encode(flat_cr)
    huffman_cb = huffman_encode(flat_cb)
    
    print(f"   Huffman encoded Y: {len(huffman_y)} bits")
    print(f"   Huffman encoded Cr: {len(huffman_cr)} bits")
    print(f"   Huffman encoded Cb: {len(huffman_cb)} bits")
    
    # Calculate compression ratio
    original_bits = original_image.size * 8
    compressed_bits = len(huffman_y) + len(huffman_cr) + len(huffman_cb)
    compression_ratio = original_bits / compressed_bits
    
    print(f"\n{'='*60}")
    print(f"COMPRESSION SUMMARY")
    print(f"{'='*60}")
    print(f"Original size: {original_bits} bits ({original_bits / 8 / 1024:.2f} KB)")
    print(f"Compressed size: {compressed_bits} bits ({compressed_bits / 8 / 1024:.2f} KB)")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Space saved: {(1 - 1/compression_ratio) * 100:.1f}%")
    
    compressed_data = {
        'huffman_y': huffman_y,
        'huffman_cr': huffman_cr,
        'huffman_cb': huffman_cb,
        'codes_y': codes_y,
        'codes_cr': codes_cr,
        'codes_cb': codes_cb,
        'rle_y_blocks': len(rle_y),
        'rle_cr_blocks': len(rle_cr),
        'rle_cb_blocks': len(rle_cb),
        'y_shape': quantized_y.shape,
        'cr_shape': quantized_cr.shape,
        'cb_shape': quantized_cb.shape,
        'original_shape': original_image.shape,
        'quality': quality,
        'chroma_subsampling': chroma_subsampling
    }
    
    return compressed_data, original_image


def decompress_image(compressed_data):
    """
    Decompress the image using JPEG decompression steps.
    
    Args:
        compressed_data: Dictionary containing all compressed data
        
    Returns:
        decompressed_image: Reconstructed RGB image as numpy array
    """
    print("\n" + "=" * 60)
    print("JPEG DECOMPRESSION PROCESS")
    print("=" * 60)
    
    # Step 1: Huffman decoding
    print("\n[Step 1] Huffman decoding...")
    flat_y = huffman_decode(compressed_data['huffman_y'], compressed_data['codes_y'])
    flat_cr = huffman_decode(compressed_data['huffman_cr'], compressed_data['codes_cr'])
    flat_cb = huffman_decode(compressed_data['huffman_cb'], compressed_data['codes_cb'])
    print(f"   Decoded Y symbols: {len(flat_y)}")
    
    # Step 2: Reconstruct channels
    print("\n[Step 2] Run-length decoding and inverse zig-zag scan...")
    quantized_y = reconstruct_channel(flat_y, compressed_data['rle_y_blocks'], compressed_data['y_shape'])
    quantized_cr = reconstruct_channel(flat_cr, compressed_data['rle_cr_blocks'], compressed_data['cr_shape'])
    quantized_cb = reconstruct_channel(flat_cb, compressed_data['rle_cb_blocks'], compressed_data['cb_shape'])
    
    # Step 3: Dequantization
    print("\n[Step 3] Dequantizing DCT coefficients...")
    quality = compressed_data.get('quality', 85)
    dequantized_y = dequantize_channel_separately(quantized_y, component='luminance', quality=quality)
    dequantized_cr = dequantize_channel_separately(quantized_cr, component='chrominance', quality=quality)
    dequantized_cb = dequantize_channel_separately(quantized_cb, component='chrominance', quality=quality)
    
    # Step 4: Inverse DCT
    print("\n[Step 4] Applying inverse DCT...")
    idct_y = apply_block_inverse_dct(dequantized_y)
    idct_cr = apply_block_inverse_dct(dequantized_cr)
    idct_cb = apply_block_inverse_dct(dequantized_cb)
    
    # Step 5: Inverse level shift
    print("\n[Step 5] Applying inverse level shift...")
    y_unshifted = idct_y + 128
    cr_unshifted = idct_cr + 128
    cb_unshifted = idct_cb + 128
    
    # Step 6: Upsample chrominance
    chroma_subsampling = compressed_data.get('chroma_subsampling', True)
    if chroma_subsampling:
        print("\n[Step 6] Upsampling chrominance components...")
        upsampled_image = upsample_chrominance_component([y_unshifted, cr_unshifted, cb_unshifted])
    else:
        print("\n[Step 6] No upsampling needed (4:4:4 format)...")
        upsampled_image = np.stack([y_unshifted, cr_unshifted, cb_unshifted], axis=2)
    
    # Step 7: Convert back to RGB
    print("\n[Step 7] Converting YCrCb back to RGB...")
    rgb_image = ycrcb_to_rgb(upsampled_image)
    
    print(f"\n{'='*60}")
    print(f"DECOMPRESSION COMPLETE")
    print(f"{'='*60}")
    
    return rgb_image


def calculate_quality_metrics(original, decompressed):
    """Calculate MSE and PSNR between original and decompressed images."""
    min_h = min(original.shape[0], decompressed.shape[0])
    min_w = min(original.shape[1], decompressed.shape[1])
    original = original[:min_h, :min_w]
    decompressed = decompressed[:min_h, :min_w]
    
    mse = np.mean((original.astype(float) - decompressed.astype(float)) ** 2)
    
    if mse == 0:
        psnr = float('inf')
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return mse, psnr


def visualize_results(original_image, decompressed_image):
    """Display original and decompressed images side by side."""
    print("\nDisplaying comparison...")
    
    mse, psnr = calculate_quality_metrics(original_image, decompressed_image)
    
    print(f"\nQUALITY METRICS:")
    print(f"   Mean Squared Error (MSE): {mse:.2f}")
    print(f"   Peak Signal-to-Noise Ratio (PSNR): {psnr:.2f} dB")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(decompressed_image)
    axes[1].set_title(f'Decompressed Image\nPSNR: {psnr:.2f} dB', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to demonstrate JPEG compression and decompression."""
    
    # Configuration
    image_path = "test.jpg"
    quality = 95  # 1-100 (higher = better quality, less compression)
    chroma_subsampling = True  # True = 4:2:0 (standard), False = 4:4:4 (best quality)
    
    try:
        # Compress the image
        compressed_data, original_image = compress_image(image_path, quality=quality, chroma_subsampling=chroma_subsampling)
        
        # Decompress the image
        decompressed_image = decompress_image(compressed_data)
        
        # Visualize results
        visualize_results(original_image, decompressed_image)
        
    except FileNotFoundError:
        print(f"\n❌ Error: Image file '{image_path}' not found!")
        print("Please provide a valid image path.")
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

"""
Helper functions for JPEG compression and decompression pipeline.
These functions handle channel-wise processing operations.
"""
import numpy as np
from utils.quantization import quantize_block, dequantize_block
from utils.run_length_encoding import zig_zag_scan, run_length_encode, run_length_decode, inverse_zig_zag_scan


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
            # Only quantize if block is 8x8
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
            # Only dequantize if block is 8x8
            if block.shape == (8, 8):
                dequantized[i:i+8, j:j+8] = dequantize_block(block, component=component, quality=quality)
    return dequantized


def process_channel_rle(channel):
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
    flattened = []
    for block in rle_blocks:
        for value, count in block:
            flattened.append((value, count))
    return flattened


def reconstruct_channel(flat_data, num_blocks, shape):
    h, w = shape
    channel = np.zeros((h, w), dtype=np.int32)
    
    # Split flat data back into blocks
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

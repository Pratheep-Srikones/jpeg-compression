import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils.color_space_conversion import rgb_to_ycrcb, ycrcb_to_rgb
from utils.chromiance_sampling import downsample_chromiance_component, upsample_chrominance_component
from utils.discrete_cosine_trasform import apply_block_dct, apply_block_inverse_dct
from utils.huffman_coding import huffman_encode, build_frequency_dict, build_huffman_tree, build_huffman_codes, huffman_decode
from utils.compression_helpers import (
    quantize_channel_separately,
    dequantize_channel_separately,
    process_channel_rle,
    flatten_rle,
    reconstruct_channel
)


def compress_image(image_path, quality=85, chroma_subsampling=True):
    """
    Compress an image using JPEG compression steps.
    
    Args:
        image_path: Path to the input image
        quality: Quality factor (1-100, higher = better quality, less compression)
                 Default: 85 (high quality)
                 Typical values: 50=medium, 75=good, 85=high, 95=excellent
        chroma_subsampling: Whether to apply 4:2:0 chrominance subsampling
                            Set to False for maximum quality (4:4:4 sampling)
        
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
    
    # Ensure dimensions are multiples of 16 (for 8x8 blocks and 2x downsampling)
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
    
    # Step 3: Chrominance downsampling (4:2:0 subsampling)
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
    
    # Step 4: Level shift (subtract 128) - Apply to each channel separately
    print("\n[Step 4] Applying level shift (centering around 0)...")
    y_shifted = y_channel.astype(np.float32) - 128
    cr_shifted = cr_channel.astype(np.float32) - 128
    cb_shifted = cb_channel.astype(np.float32) - 128
    print(f"   Shifted Y range: [{y_shifted.min():.1f}, {y_shifted.max():.1f}]")
    
    # Step 5: DCT (Discrete Cosine Transform) - Apply to each channel separately
    print("\n[Step 5] Applying DCT to 8x8 blocks...")
    dct_y = apply_block_dct(y_shifted)
    dct_cr = apply_block_dct(cr_shifted)
    dct_cb = apply_block_dct(cb_shifted)
    print(f"   DCT Y shape: {dct_y.shape}")
    print(f"   DCT Cr shape: {dct_cr.shape}")
    print(f"   DCT Cb shape: {dct_cb.shape}")
    print(f"   DCT applied successfully")
    
    # Step 6: Quantization - Quantize each channel separately
    print("\n[Step 6] Quantizing DCT coefficients...")
    quantized_y = quantize_channel_separately(dct_y, component='luminance', quality=quality)
    quantized_cr = quantize_channel_separately(dct_cr, component='chrominance', quality=quality)
    quantized_cb = quantize_channel_separately(dct_cb, component='chrominance', quality=quality)
    
    print(f"   Quantized Y shape: {quantized_y.shape}")
    print(f"   Quantized Cr shape: {quantized_cr.shape}")
    print(f"   Quantized Cb shape: {quantized_cb.shape}")
    print(f"   Non-zero coefficients in Y: {np.count_nonzero(quantized_y)}/{quantized_y.size}")
    
    # Step 7: Zig-zag scan and Run-Length Encoding
    print("\n[Step 7] Applying zig-zag scan and run-length encoding...")
    rle_y = process_channel_rle(quantized_y)
    rle_cr = process_channel_rle(quantized_cr)
    rle_cb = process_channel_rle(quantized_cb)
    print(f"   Number of 8x8 blocks encoded in Y: {len(rle_y)}")
    
    # Step 8: Huffman coding
    print("\n[Step 8] Applying Huffman coding...")
    
    # Flatten RLE data
    flat_y = flatten_rle(rle_y)
    flat_cr = flatten_rle(rle_cr)
    flat_cb = flatten_rle(rle_cb)
    
    # Build Huffman codes
    freq_y = build_frequency_dict(flat_y)
    tree_y = build_huffman_tree(freq_y)
    codes_y = build_huffman_codes(tree_y)
    
    freq_cr = build_frequency_dict(flat_cr)
    tree_cr = build_huffman_tree(freq_cr)
    codes_cr = build_huffman_codes(tree_cr)
    
    freq_cb = build_frequency_dict(flat_cb)
    tree_cb = build_huffman_tree(freq_cb)
    codes_cb = build_huffman_codes(tree_cb)
    
    # Encode
    huffman_y = huffman_encode(flat_y)
    huffman_cr = huffman_encode(flat_cr)
    huffman_cb = huffman_encode(flat_cb)
    
    print(f"   Huffman encoded Y: {len(huffman_y)} bits")
    print(f"   Huffman encoded Cr: {len(huffman_cr)} bits")
    print(f"   Huffman encoded Cb: {len(huffman_cb)} bits")
    
    # Calculate compression ratio
    original_bits = original_image.size * 8  # 8 bits per byte
    compressed_bits = len(huffman_y) + len(huffman_cr) + len(huffman_cb)
    compression_ratio = original_bits / compressed_bits
    
    print(f"\n{'='*60}")
    print(f"COMPRESSION SUMMARY")
    print(f"{'='*60}")
    print(f"Original size: {original_bits} bits ({original_bits / 8 / 1024:.2f} KB)")
    print(f"Compressed size: {compressed_bits} bits ({compressed_bits / 8 / 1024:.2f} KB)")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Space saved: {(1 - 1/compression_ratio) * 100:.1f}%")
    
    # Store all compressed data
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
        'quality': quality,  # Store quality for decompression
        'chroma_subsampling': chroma_subsampling  # Store subsampling setting
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
    print(f"   Decoded Cr symbols: {len(flat_cr)}")
    print(f"   Decoded Cb symbols: {len(flat_cb)}")
    
    # Step 2: Reconstruct RLE blocks and Run-Length Decoding
    print("\n[Step 2] Run-length decoding and inverse zig-zag scan...")
    quantized_y = reconstruct_channel(flat_y, compressed_data['rle_y_blocks'], compressed_data['y_shape'])
    quantized_cr = reconstruct_channel(flat_cr, compressed_data['rle_cr_blocks'], compressed_data['cr_shape'])
    quantized_cb = reconstruct_channel(flat_cb, compressed_data['rle_cb_blocks'], compressed_data['cb_shape'])
    print(f"   Reconstructed Y shape: {quantized_y.shape}")
    print(f"   Reconstructed Cr shape: {quantized_cr.shape}")
    print(f"   Reconstructed Cb shape: {quantized_cb.shape}")
    
    # Step 3: Dequantization - Dequantize each channel separately
    print("\n[Step 3] Dequantizing DCT coefficients...")
    quality = compressed_data.get('quality', 85)  # Get quality from compressed data
    dequantized_y = dequantize_channel_separately(quantized_y, component='luminance', quality=quality)
    dequantized_cr = dequantize_channel_separately(quantized_cr, component='chrominance', quality=quality)
    dequantized_cb = dequantize_channel_separately(quantized_cb, component='chrominance', quality=quality)
    
    print(f"   Dequantized Y shape: {dequantized_y.shape}")
    print(f"   Dequantized Cr shape: {dequantized_cr.shape}")
    print(f"   Dequantized Cb shape: {dequantized_cb.shape}")
    
    # Step 4: Inverse DCT - Apply to each channel separately
    print("\n[Step 4] Applying inverse DCT...")
    idct_y = apply_block_inverse_dct(dequantized_y)
    idct_cr = apply_block_inverse_dct(dequantized_cr)
    idct_cb = apply_block_inverse_dct(dequantized_cb)
    print(f"   IDCT Y shape: {idct_y.shape}")
    print(f"   IDCT Cr shape: {idct_cr.shape}")
    print(f"   IDCT Cb shape: {idct_cb.shape}")
    
    # Step 5: Inverse level shift - Apply to each channel separately
    print("\n[Step 5] Applying inverse level shift...")
    y_unshifted = idct_y + 128
    cr_unshifted = idct_cr + 128
    cb_unshifted = idct_cb + 128
    print(f"   Unshifted Y range: [{y_unshifted.min():.1f}, {y_unshifted.max():.1f}]")
    
    # Step 6: Upsample chrominance components
    chroma_subsampling = compressed_data.get('chroma_subsampling', True)
    if chroma_subsampling:
        print("\n[Step 6] Upsampling chrominance components...")
        # Pass in YCrCb order
        upsampled_image = upsample_chrominance_component([y_unshifted, cr_unshifted, cb_unshifted])
    else:
        print("\n[Step 6] No upsampling needed (4:4:4 format)...")
        # Stack in YCrCb order
        upsampled_image = np.stack([y_unshifted, cr_unshifted, cb_unshifted], axis=2)
    print(f"   Final YCrCb image shape: {upsampled_image.shape}")
    
    # Step 7: Convert back to RGB
    print("\n[Step 7] Converting YCrCb back to RGB...")
    rgb_image = ycrcb_to_rgb(upsampled_image)
    print(f"   Final RGB image shape: {rgb_image.shape}")
    
    print(f"\n{'='*60}")
    print(f"DECOMPRESSION COMPLETE")
    print(f"{'='*60}")
    
    return rgb_image


def calculate_quality_metrics(original, decompressed):
    """Calculate quality metrics between original and decompressed images"""
    # Ensure same shape
    min_h = min(original.shape[0], decompressed.shape[0])
    min_w = min(original.shape[1], decompressed.shape[1])
    original = original[:min_h, :min_w]
    decompressed = decompressed[:min_h, :min_w]
    
    # Calculate MSE (Mean Squared Error)
    mse = np.mean((original.astype(float) - decompressed.astype(float)) ** 2)
    
    # Calculate PSNR (Peak Signal-to-Noise Ratio)
    if mse == 0:
        psnr = float('inf')
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return mse, psnr


def visualize_results(original_image, decompressed_image):
    """Display original and decompressed images side by side"""
    print("\nDisplaying comparison...")
    
    # Calculate quality metrics
    mse, psnr = calculate_quality_metrics(original_image, decompressed_image)
    
    print(f"\nQUALITY METRICS:")
    print(f"   Mean Squared Error (MSE): {mse:.2f}")
    print(f"   Peak Signal-to-Noise Ratio (PSNR): {psnr:.2f} dB")
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Decompressed image
    axes[1].imshow(decompressed_image)
    axes[1].set_title(f'Decompressed Image\nPSNR: {psnr:.2f} dB', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function to demonstrate JPEG compression and decompression"""
    
    # Path to your test image
    image_path = "test.jpg"  # Change this to your image path
    
    # Quality setting: 1-100 (higher = better quality, less compression)
    # 50 = medium quality, 75 = good quality, 85 = high quality, 95 = excellent quality, 99 = near-lossless
    quality = 95  # Using very high quality to minimize artifacts
    
    # Chroma subsampling: False = 4:4:4 (best quality), True = 4:2:0 (standard JPEG)
    chroma_subsampling = True  # Use standard 4:2:0 (good balance of quality and compression)
    
    try:
        # Compress the image
        compressed_data, original_image = compress_image(image_path, quality=quality, chroma_subsampling=chroma_subsampling)
        
        # Decompress the image
        decompressed_image = decompress_image(compressed_data)
        
        # Visualize results
        visualize_results(original_image, decompressed_image)
        
    except FileNotFoundError:
        print(f"\n❌ Error: Image file '{image_path}' not found!")
        print("Please provide a valid image path in the main() function.")
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

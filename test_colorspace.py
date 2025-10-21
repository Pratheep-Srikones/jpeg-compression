import numpy as np
from utils.color_space_conversion import rgb_to_ycrcb, ycrcb_to_rgb
from PIL import Image

# Load image
img = Image.open("test.jpg")
original = np.array(img)[:128, :128]  # Small patch for testing

print("Original RGB range:", original.min(), "-", original.max())

# Convert to YCrCb and back
ycrcb = rgb_to_ycrcb(original)
print("YCrCb range:", ycrcb.min(), "-", ycrcb.max())

reconstructed = ycrcb_to_rgb(ycrcb)
print("Reconstructed RGB range:", reconstructed.min(), "-", reconstructed.max())

# Calculate error
mse = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)
psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')

print(f"\nColor space conversion test:")
print(f"MSE: {mse:.2f}")
print(f"PSNR: {psnr:.2f} dB")

if psnr < 40:
    print("⚠️  WARNING: Color space conversion has significant loss!")
else:
    print("✓ Color space conversion is good")

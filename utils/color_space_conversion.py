import numpy as np

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
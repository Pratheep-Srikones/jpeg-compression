import numpy as np

def zig_zag_scan(block):
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
    decoded = []
    for value, count in encoded:
        # Stop at EOB marker
        if value == 0 and count == 0:
            break
        decoded.extend([value] * count)
    return np.array(decoded)
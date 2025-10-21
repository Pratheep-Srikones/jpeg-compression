import numpy as np
import heapq
class Node:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_frequency_dict(data):
    frequency = {}
    for symbol in data:
        if symbol in frequency:
            frequency[symbol] += 1
        else:
            frequency[symbol] = 1
    return frequency

def build_huffman_tree(frequency_data):

    heap = [Node(symbol, freq) for symbol, freq in frequency_data.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]

def build_huffman_codes(node, prefix="", codebook=None):
    if codebook is None:
        codebook = {}
    if node is not None:
        if node.symbol is not None:
            codebook[node.symbol] = prefix
        build_huffman_codes(node.left, prefix + "0", codebook)
        build_huffman_codes(node.right, prefix + "1", codebook)
    return codebook

def huffman_encode(data):
    frequency_data = build_frequency_dict(data)
    huffman_tree = build_huffman_tree(frequency_data)
    huffman_codes = build_huffman_codes(huffman_tree)

    encoded_data = ''.join(huffman_codes[symbol] for symbol in data)
    return encoded_data

def huffman_decode(encoded_data, huffman_codes):
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
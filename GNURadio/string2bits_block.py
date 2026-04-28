"""
Embedded Python Blocks:

Each time this file is saved, GRC will instantiate the first class it finds
to get ports and parameters of your block. The arguments to __init__  will
be the parameters. All of them are required to have default values!
"""

import numpy as np
from gnuradio import gr

def encode(str): 
    return ''.join(format(ord(char), '08b') for char in str)

class blk(gr.sync_block):  
    """Take a string message and output it as a stream of bytes."""

    def __init__(self, message="Hello, World!"):
        gr.sync_block.__init__(self, name='String to Bits', in_sig=[], out_sig=[np.int8])
        self.bits = np.array([int(bit) for bit in encode(message)], dtype=np.int8)
        self.index = 0

    def work(self, input_items, output_items):
        out = output_items[0]
        n = min(len(out), len(self.bits) - self.index)

        if n > 0:
            out[:n] = self.bits[self.index:self.index+n]
            self.index += n

        return n
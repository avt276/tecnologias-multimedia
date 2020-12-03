#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

''' Real-time Audio Intercommunicator (lossless compression of the chunks). '''

import zlib
import numpy as np
import struct
import math
try:
    import argcomplete  # <tab> completion for argparse.
except ImportError:
    print("Unable to import argcomplete")
import minimal
import buffer
import compress

class BR_Control(compress.Compression):
    def __init__(self):
        if __debug__:
            print("Running BR_Control.__init__")
        super().__init__()
        if __debug__:
            print("InterCom (BR_Control) is running")

    def deadzone_quantizer(self, x, quantization_step):
        k = (x / quantization_step).astype(np.int)
        return k

    def deadzone_dequantizer(self, k, quantization_step):
        y = quantization_step * k
        return y  

    def pack(self, chunk_number, chunk):
        quantized_step = 1
        quantized_chunk = self.deadzone_quantizer(chunk, quantized_step)
        quantized_chunk = super().pack(chunk_number, chunk)
        return quantized_chunk

    def unpack(self, packed_chunk, dtype=minimal.Minimal.SAMPLE_TYPE):
        (chunk_number, chunk) = super().unpack(packed_chunk, dtype)
        quantized_step = 1
        chunk = self.deadzone_dequantizer(chunk, quantized_step)
        return chunk_number, chunk

class BR_Control__verbose(BR_Control, compress.Compression__verbose):
    def __init__(self):
        if __debug__:
            print("Running BR_Control__verbose.__init__")
        super().__init__()
        self.variance = np.zeros(self.NUMBER_OF_CHANNELS) # Variance of the chunks_per_cycle chunks.
        self.entropy = np.zeros(self.NUMBER_OF_CHANNELS) # Entropy of the chunks_per_cycle chunks.
        self.bps = np.zeros(self.NUMBER_OF_CHANNELS) # Bits Per Symbol of the chunks_per_cycle compressed chunks.
        self.chunks_in_the_cycle = []

        self.average_variance = np.zeros(self.NUMBER_OF_CHANNELS)
        self.average_entropy = np.zeros(self.NUMBER_OF_CHANNELS)
        self.average_bps = np.zeros(self.NUMBER_OF_CHANNELS)

    def stats(self):
        string = super().stats()
        string += " {}".format(['{:9.0f}'.format(i) for i in self.variance])
        string += " {}".format(['{:4.1f}'.format(i) for i in self.entropy])
        string += " {}".format(['{:4.1f}'.format(i/self.frames_per_cycle) for i in self.bps])
        return string

    def first_line(self):
        string = super().first_line()
        string += "{:27s}".format('') # variance
        string += "{:17s}".format('') # entropy
        string += "{:17s}".format('') # bps
        return string

    def second_line(self):
        string = super().second_line()
        string += "{:>27s}".format("variance") # variance
        string += "{:>17s}".format("entropy") # entropy
        string += "{:>17s}".format("BPS") # bps
        return string

    def separator(self):
        string = super().separator()
        string += f"{'='*(27+17*2)}"
        return string

    def averages(self):
        string = super().averages()
        string += " {}".format(['{:9.0f}'.format(i) for i in self.average_variance])
        string += " {}".format(['{:4.1f}'.format(i) for i in self.average_entropy])
        string += " {}".format(['{:4.1f}'.format(i/self.frames_per_cycle) for i in self.average_bps])
        return string
        
    def entropy_in_bits_per_symbol(self, sequence_of_symbols):
        value, counts = np.unique(sequence_of_symbols, return_counts = True)
        probs = counts / len(sequence_of_symbols)
        #n_classes = np.count_nonzero(probs)

        #if n_classes <= 1:
        #    return 0

        entropy = 0.
        for i in probs:
            entropy -= i * math.log(i, 2)

        return entropy

    def cycle_feedback(self):
        try:
            concatenated_chunks = np.vstack(self.chunks_in_the_cycle)
        except ValueError:
            concatenated_chunks = np.vstack([self.zero_chunk, self.zero_chunk])
        
        self.variance = np.var(concatenated_chunks, axis=0)
        self.average_variance = self.moving_average(self.average_variance, self.variance, self.cycle)

        self.entropy[0] = self.entropy_in_bits_per_symbol(concatenated_chunks[:, 0])
        self.entropy[1] = self.entropy_in_bits_per_symbol(concatenated_chunks[:, 1])
        self.average_entropy = self.moving_average(self.average_entropy, self.entropy, self.cycle)

        self.average_bps = self.moving_average(self.average_bps, self.bps, self.cycle)

        super().cycle_feedback()
        self.chunks_in_the_cycle = []
        self.bps = np.zeros(self.NUMBER_OF_CHANNELS)
        
    def _record_send_and_play(self, indata, outdata, frames, time, status):
        super()._record_send_and_play(indata, outdata, frames, time, status)
        self.chunks_in_the_cycle.append(indata)
        # Remember: indata contains the recorded chunk and outdata,
        # the played chunk.

    def unpack(self, packed_chunk, dtype=minimal.Minimal.SAMPLE_TYPE):
        (chunk_number, len_compressed_channel_0) = struct.unpack("!HH", packed_chunk[:4])
        len_compressed_channel_1 = len(packed_chunk[len_compressed_channel_0+4:])

        self.bps[0] += len_compressed_channel_0*8
        self.bps[1] += len_compressed_channel_1*8
        chunk_number, chunk = super().unpack(packed_chunk, dtype)
        return chunk_number, chunk

if __name__ == "__main__":
    minimal.parser.description = __doc__
    try:
        argcomplete.autocomplete(minimal.parser)
    except Exception:
        if __debug__:
            print("argcomplete not working :-/")
        else:
            pass
    minimal.args = minimal.parser.parse_known_args()[0]
    if minimal.args.show_stats or minimal.args.show_samples:
        intercom = BR_Control__verbose()
    else:
        intercom = BR_Control__verbose()
    try:
        intercom.run()
    except KeyboardInterrupt:
        minimal.parser.exit("\nInterrupted by user")

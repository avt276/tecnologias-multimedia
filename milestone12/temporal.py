#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

''' Real-time Audio Intercommunicator (lossless compression of the chunks). '''

import zlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import pylab
import pywt
import pywt.data
import struct
import math
try:
    import argcomplete  # <tab> completion for argparse.
except ImportError:
    print("Unable to import argcomplete")
import minimal
import buffer
import compress
import threading
import time
import br_control
import sounddevice as sd
import spatial

class Temporal_decorrelation(spatial.Spatial_decorrelation):
    def __init__(self):
        if __debug__:
            print("Running Temporal_decorrelation.__init__")
        super().__init__()
        if __debug__:
            print("InterCom (Temporal_decorrelation) is running")
        self.levels = 3
        self.wavelet_name = "db5"
        self.wavelet = pywt.Wavelet(self.wavelet_name)
        self.quantization_step = 128
        self.quantized_coeffs
        self.slices

    '''
    # Forward transform:
    #
    #  [w[0]] = [1  1] [x[0]]
    #  [w[1]]   [1 -1] [x[1]]

    def MST_analyze(self, x):
        w = np.empty_like(x, dtype=np.int32)
        w[:, 0] = x[:, 0].astype(np.int32) + x[:, 1] # L(ow frequency subband)
        w[:, 1] = x[:, 0].astype(np.int32) - x[:, 1] # H(igh frequency subband)
        return w

    # Inverse transform:
    #
    #  [x[0]] = 1/2 [1  1] [w[0]]
    #  [x[1]]       [1 -1] [w[1]]

    def MST_synthesize(self, w):
        x = np.empty_like(w, dtype=np.int16)
        x[:, 0] = (w[:, 0] + w[:, 1])/2 # L(ow frequency subband)
        x[:, 1] = (w[:, 0] - w[:, 1])/2 # H(igh frequency subband)
        return x
    '''

    def deadzone_quantizer(self, x, quantization_step):
        k = (x / quantization_step).astype(np.int)
        return k

    def deadzone_dequantizer(self, k, quantization_step):
        y = quantization_step * k
        return y
    '''
    def transform_and_quantize(self, chunk):
        decomposition = pywt.wavedec(chunk, wavelet=self.wavelet, level=self.levels, mode="per")
        coefficients, slices = pywt.coeffs_to_array(decomposition)
        quantized_coeffs = self.deadzone_dequantizer(self.deadzone_quantizer(coefficients, self.quantization_step), self.quantization_step)
        decomposition = pywt.array_to_coeffs(quantized_coeffs, slices, output_format="wavedec")
        reconstructed_chunk = pywt.waverec(decomposition, wavelet=self.wavelet, mode="per")
        return reconstructed_chunk
    '''

    def pack(self, chunk_number, chunk):
        #analyzed_chunk = self.MST_analyze(chunk)
        decomposition = pywt.wavedec(chunk, wavelet=self.wavelet, level=self.levels, mode="per")
        coefficients, self.slices = pywt.coeffs_to_array(decomposition)
        self.quantized_coeffs = self.deadzone_quantizer(coefficients, self.quantization_step)
        return super().pack(chunk_number, chunk)

    def unpack(self, packed_chunk, dtype=minimal.Minimal.SAMPLE_TYPE):
        chunk_number, chunk = super().unpack(packed_chunk, dtype)
        quantized_coeffs = self.deadzone_dequantizer(self.quantized_coeffs, self.quantization_step)
        decomposition = pywt.array_to_coeffs(quantized_coeffs, self.slices, output_format="wavedec")
        reconstructed_chunk = pywt.waverec(decomposition, wavelet=self.wavelet, mode="per")
        #synthesize_chunk = self.MST_synthesize(chunk)
        return chunk_number, reconstructed_chunk

class Temporal_decorrelation__verbose(Temporal_decorrelation, spatial.Spatial_decorrelation__verbose):
    def __init__(self):
        if __debug__:
            print("Running Temporal_decorrelation__verbose.__init__")
        super().__init__()
        self.variance = np.zeros(self.NUMBER_OF_CHANNELS) # Variance of the chunks_per_cycle chunks.
        self.entropy = np.zeros(self.NUMBER_OF_CHANNELS) # Entropy of the chunks_per_cycle chunks.
        self.bps = np.zeros(self.NUMBER_OF_CHANNELS) # Bits Per Symbol of the chunks_per_cycle compressed chunks.
        self.chunks_in_the_cycle = []

        self.average_vself, variance = np.zeros(self.NUMBER_OF_CHANNELS)
        self.average_entropy = np.zeros(self.NUMBER_OF_CHANNELS)
        self.average_bps = np.zeros(self.NUMBER_OF_CHANNELS)
        
    def stats(self):
        string = super().stats()
        #string += " {}".format(['{:4.1f}'.format(self.quantized_step)])
        return string

    def first_line(self):
        string = super().first_line()
        #string += "{:8s}".format('') # quantized_step
        return string

    def second_line(self):
        string = super().second_line()
        #string += "{:>8s}".format("QS") # quantized_step
        return string

    def separator(self):
        string = super().separator()
        #string += f"{'='*(20)}"
        return string

    def averages(self):
        string = super().averages()
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
        super().cycle_feedback()


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
        intercom = Temporal_decorrelation__verbose()
    else:
        intercom = Temporal_decorrelation__verbose()
    try:
        intercom.run()
    except KeyboardInterrupt:
        minimal.parser.exit("\nInterrupted by user")

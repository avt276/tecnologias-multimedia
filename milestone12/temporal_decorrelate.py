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
import intra_frame_decorrelation
'''
pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html#maximum-decomposition-level-dwt-max-level-dwtn-max-level
'''
class Temporal_decorrelation(intra_frame_decorrelation.Intra_frame_decorrelation):
    def __init__(self):
        if __debug__:
            print("Running Temporal_decorrelation.__init__")
        super().__init__()
        if __debug__:
            print("InterCom (Temporal_decorrelation) is running")
        self.levels = 3
        self.wavelet_name = "db11"
        self.wavelet = pywt.Wavelet(self.wavelet_name)
        self.previous_chunk = super().generate_zero_chunk()
        self.current_chunk = super().generate_zero_chunk()
        self.next_chunk = super().generate_zero_chunk()
        self.extended_chunk = super().generate_zero_chunk()
        self.slices = 0
        
    def pack(self, chunk_number, chunk):
        self.next_chunk = chunk

        number_of_overlaped_samples = 1 << math.ceil(math.log(self.wavelet.dec_len * self.levels) / math.log(2))
        self.extended_chunk = np.concatenate([self.previous_chunk[len(self.previous_chunk) - number_of_overlaped_samples :], self.current_chunk, self.next_chunk[: number_of_overlaped_samples]])
        
        decomposition = pywt.wavedec(self.extended_chunk, wavelet=self.wavelet, level=self.levels, mode="per")
        coefficients, self.slices = pywt.coeffs_to_array(decomposition)
        #return super().pack(chunk_number, chunk)
        return super().pack(chunk_number, coefficients)
        #TODO: elegir pywavelet.

    def unpack(self, packed_chunk, dtype=minimal.Minimal.SAMPLE_TYPE):
        chunk_number, chunk = super().unpack(packed_chunk, dtype)
        decomposition = pywt.array_to_coeffs(chunk, self.slices, output_format="wavedec")
        reconstructed_chunk = pywt.waverec(decomposition, wavelet=self.wavelet, mode="per")
        self.previous_chunk = self.current_chunk
        self.current_chunk = self.next_chunk
        return chunk_number, reconstructed_chunk

class Temporal_decorrelation__verbose(Temporal_decorrelation, intra_frame_decorrelation.Intra_frame_decorrelation__verbose):
    def __init__(self):
        super().__init__()
        self.LH_variance = np.zeros(self.NUMBER_OF_CHANNELS)
        self.average_LH_variance = np.zeros(self.NUMBER_OF_CHANNELS)
        self.LH_chunks_in_the_cycle = []

    def stats(self):
        #string = super().stats()
        #string += " {}".format(['{:>5d}'.format(int(i/1000)) for i in self.LH_variance])
        super.stats(self)

    def _first_line(self):
        #string = super().first_line()
        #string += "{:19s}".format('') # LH_variance
        super._first_line(self)

    def first_line(self):
        #string = super().first_line()
        #string += "{:19s}".format('') # LH_variance
        super.first_line(self)

    def second_line(self):
        #string = super().second_line()
        #string += "{:>19s}".format("LH variance") # LH variance
        super.second_line(self)

    def separator(self):
        #string = super().separator()
        #string += f"{'='*19}"
        #return string
        super.separator(self)

    def averages(self):
        #string = super().averages()
        #string += " {}".format(['{:>5d}'.format(int(i/1000)) for i in self.average_LH_variance])
        #return string
        super.averages(self)

    def cycle_feedback(self):
        #try:
            #concatenated_chunks = np.vstack(self.LH_chunks_in_the_cycle)
            #self.LH_variance = np.var(concatenated_chunks, axis=0)
        #except ValueError:
            #pass
        #self.average_LH_variance = self.moving_average(self.average_LH_variance, self.LH_variance, self.cycle)
        super().cycle_feedback()
        #self.LH_chunks_in_the_cycle = []

    def analyze(self, chunk):
        analyzed_chunk = super().analyze(chunk)
        self.LH_chunks_in_the_cycle.append(analyzed_chunk)
        return analyzed_chunk

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
        intercom = Temporal_decorrelation()
    try:
        intercom.run()
    except KeyboardInterrupt:
        minimal.parser.exit("\nInterrupted by user")
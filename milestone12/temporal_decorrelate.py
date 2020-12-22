#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

''' Real-time Audio Intercommunicator (lossless compression of the chunks). '''

import numpy as np
import pywt
import pywt.data
import math
try:
    import argcomplete  # <tab> completion for argparse.
except ImportError:
    print("Unable to import argcomplete")
import minimal
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
        self.wavelet_name = "db5"
        self.wavelet = pywt.Wavelet(self.wavelet_name)
        self.previous_chunk = super().generate_zero_chunk()
        self.current_chunk = super().generate_zero_chunk()
        self.next_chunk = super().generate_zero_chunk()
        self.extended_chunk = super().generate_zero_chunk()

        if(self.levels == 0):
            self.number_of_overlaped_samples = 0
        else:
            self.number_of_overlaped_samples = 1 << math.ceil(math.log(self.wavelet.dec_len * self.levels) / math.log(2))

        #print("Size current chunk:", self.current_chunk.shape)
        temp_decomposition = pywt.wavedec(self.current_chunk[:,0], wavelet=self.wavelet, level=self.levels, mode="per")
        temp_coefficients, self.slices = pywt.coeffs_to_array(temp_decomposition)
    
        
    def pack(self, chunk_number, chunk):
        self.next_chunk = chunk
        
        self.extended_chunk = np.concatenate([self.previous_chunk[len(self.previous_chunk) - self.number_of_overlaped_samples :], self.current_chunk, self.next_chunk[: self.number_of_overlaped_samples]])
        
        decomposition_left = pywt.wavedec(self.extended_chunk[:,0], wavelet=self.wavelet, level=self.levels, mode="per")
        coefficients_0, slices = pywt.coeffs_to_array(decomposition_left)
        decomposition_right = pywt.wavedec(self.extended_chunk[:,1], wavelet=self.wavelet, level=self.levels, mode="per")
        coefficients_1, slices = pywt.coeffs_to_array(decomposition_right)
        
        coefficients = np.empty_like(self.extended_chunk, dtype=np.int32)
        coefficients[:, 0] = coefficients_0[:]
        coefficients[:, 1] = coefficients_1[:]
        return super().pack(chunk_number, coefficients[self.number_of_overlaped_samples:-self.number_of_overlaped_samples])
        #TODO: elegir pywavelet.

    def unpack(self, packed_chunk, dtype=minimal.Minimal.SAMPLE_TYPE):
        chunk_number, chunk = super().unpack(packed_chunk, dtype)
        #print("Chunk: ", chunk.shape)
        #print("Slices: ", self.slices)
        decomposition_left = pywt.array_to_coeffs(chunk[:,0], self.slices, output_format="wavedec")
        decomposition_right = pywt.array_to_coeffs(chunk[:,1], self.slices, output_format="wavedec")
        reconstructed_chunk_left = pywt.waverec(decomposition_left, wavelet=self.wavelet, mode="per")
        reconstructed_chunk_right = pywt.waverec(decomposition_right, wavelet=self.wavelet, mode="per")
        reconstructed_chunk = np.empty((minimal.args.frames_per_chunk, 2), dtype=dtype)
        reconstructed_chunk[:, 0] = reconstructed_chunk_left[:]
        reconstructed_chunk[:, 1] = reconstructed_chunk_right[:]
        self.previous_chunk = self.current_chunk
        self.current_chunk = self.next_chunk
        return chunk_number, reconstructed_chunk

class Temporal_decorrelation__verbose(Temporal_decorrelation, intra_frame_decorrelation.Intra_frame_decorrelation__verbose):
    pass

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
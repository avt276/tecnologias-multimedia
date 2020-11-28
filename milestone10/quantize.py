## BR_Control
#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

import argparse
import sounddevice as sd
import numpy as np
import socket
import time
import psutil
import math
import struct
import threading
try:
    import argcomplete  # <tab> completion for argparse.
except ImportError:
    print("Unable to import argcomplete")
import minimal
import buffer as bf
import zlib as z
import compress_order 

class BR_Control(compress_order.Compress):

    def __init__(self):
        ''' Initializes the buffer. '''
        super().__init__()
        if minimal.args.buffering_time <= 0:
            minimal.args.buffering_time = 1 # ms
        print(f"buffering_time = {minimal.args.buffering_time} miliseconds")
        if __debug__:
            print("chunks_to_buffer =", self.chunks_to_buffer)     
    
    def pack(self, chunk_number, chunk):
        quantized_step = 700
        quantized_chunk = deadzone_quantizer(chunk, quantized_step)
        #print("---------Enviamos--------------")
        #print(quantized_chunk)
        packed_chunk = bf.Buffering.pack(chunk_number, quantized_chunk)
        #print("---------Packed_Chunk--------------")
        #print(packed_chunk)
        return packed_chunk

    def unpack(self, packed_chunk, dtype=minimal.Minimal.SAMPLE_TYPE):
        (chunk_number, ) = bf.Buffering.unpack(packed_chunk)
        quantized_chunk = packed_chunk[2:]
        #print("---------Recibimos y desempaquetamos--------------")
        #print(quantized_chunk)
        quantized_step = 700
        chunk = deadzone_dequantizer(quantize_chunk, quantized_step)
        
        return chunk_number, chunk
              
    def deadzone_quantizer(x, quantization_step):
        k = (x / quantization_step).astype(np.int)
        return k

    def deadzone_dequantizer(k, quantization_step):
        y = quantization_step * k
        return y  
    
    def run(self):
        print("Press CTRL+c to quit")
        self.played_chunk_number = 0
        with self.stream(self._record_send_and_play):

            first_received_chunk_number = self.receive_and_buffer()
            if __debug__:
                print("first_received_chunk_number =", first_received_chunk_number)

            self.played_chunk_number = (first_received_chunk_number - self.chunks_to_buffer) % self.cells_in_buffer
            # The previous selects the first chunk to be played the
            # one (probably emptty) that are in the buffer
            # self.chunks_to_buffer position before
            # first_received_chunk_number.

            while True:
                self.receive_and_buffer()

class BR_Control_verbose(BR_Control, compress_order.Compress__verbose):
    def __init__(self):
        super().__init__()
        thread = threading.Thread(target=self.feedback)
        thread.daemon = True # To obey CTRL+C interruption.
        thread.start()

    def feedback(self):
        while True:
            time.sleep(self.SECONDS_PER_CYCLE)
            self.cycle_feedback()

    def send(self, packed_chunk):
        BR_Control.send(self, packed_chunk)
        self.sent_bytes_count += len(packed_chunk)
        self.sent_messages_count += 1

    def receive(self):
        packed_chunk = super().receive()
        self.received_bytes_count += len(packed_chunk)
        self.received_messages_count += 1
        return packed_chunk

    def _record_send_and_play(self, indata, outdata, frames, time, status):
        if minimal.args.show_samples:
            self.show_indata(indata)

        super()._record_send_and_play(indata, outdata, frames, time, status)

        if minimal.args.show_samples:
            self.show_outdata(outdata)

    def run(self):
        '''.'''
        print("Press CTRL+c to quit")
        self.print_header()
        try:
            self.played_chunk_number = 0
            with self.stream(self._record_send_and_play):
                first_received_chunk_number = self.receive_and_buffer()
                if __debug__:
                    print("first_received_chunk_number =", first_received_chunk_number)
                self.played_chunk_number = (first_received_chunk_number - self.chunks_to_buffer) % self.cells_in_buffer
                while True:
                    self.receive_and_buffer()
        except KeyboardInterrupt:
            self.print_final_averages()

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
        intercom = BR_Control_verbose
    else:
        intercom = BR_Control()
    try:
        intercom.run()
    except KeyboardInterrupt:
        minimal.parser.exit("\nInterrupted by user")

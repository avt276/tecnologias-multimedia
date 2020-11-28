#
# Intercom_buffer
# |
# +- compress
#

from intercom_buffer import Intercom_buffer
import sounddevice as sd
import numpy as np
import psutil
import time
from multiprocessing import Process
import struct as st
import math
import zlib as z



class compress(Intercom_buffer):

     
            
    def pack(self):
        comprimido = z.compress(self)
        
        
        return comprimido
    
    def unpack(self):
        descomprimido = z.descompress(self)


        return descomprimido
    
    


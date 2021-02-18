#Bandpass filtering with FFT
#Calculation of cut-off frequency---------------------------------------------
import matplotlib.pylab as plt
import mne.filter as filter
import mne.time_frequency.tfr as tfr
import numpy as np
import pandas as pd  
import pywt
import scipy.io as sio
import scipy.linalg as linalg
from scipy.sparse import csr_matrix

from numpy import matlib
from numpy import linalg

#Test 3D array
a_3d_array = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(a_3d_array)

#Test 3D array2
import pprint
n = 3
distance = [[[0 for k in xrange(n)] for j in xrange(n)] for i in xrange(n)]
pprint.pprint(distance)
[[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
 [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
 [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]
distance[0][1]
[0, 0, 0]
distance[0][1][2]
0

#cut-off frequency (fc) = 1/(2PI*R*C)
#R: Resistance (Ohm) = ?
#C: Capacitance (µF) = ?
#sampling rate (fs) = 256 Hz
#length of input segment = 205
#samples = 800 ms
#fiter = 1-10 Hz bandpass

#Random numbers
j=2
omega=4
omega.c=5
R=10
C=12
H(j*omega)=1/(1+j.omega*R*C)
abs.H(j*omega.c)=1/(np.sqrt(1**2+omega.c**2*R**2*C**2))
abs.H(j*omega.c)=1/(np.sqrt(2))*abs(H(j*0))
abs.H(j*omega.c)=1/(np.sqrt(2))

#Transformation into frequency space------------------------------------------
coeffs=numpy.fft.fft(data)

#Removing coefficients--------------------------------------------------------

#Inverse Transformation into time-amplitude plane-----------------------------
filtdata=real(numpy.fft.ifft(coeffs))

#Complex Morlet Wavelet Example-----------------------------------------------

import numpy as np
import matplotlib.pylab as plt

#import wavelet 
#ermöglicht nutzen von funktoinen aus wavelet.py (Wie??)

x = np.arange(-5,5, 0.001)
sigma = 2
f = 1
s = np.sqrt(np.pi*sigma)*np.exp(2*np.pi*1j*f*x)*np.exp(-x**2/sigma)

#data = np.loadtxt("avrgPerChannel.npy") ??

plt.figure()

plt.plot(x, np.real(s), label="real")
plt.plot(x, np.imag(s), label="imag")
plt.plot(x, np.abs(s), label="abs")
plt.legend()
plt.show()
#-----------------------------------------------------------------------------

wavelet = np.loadtxt("wavelet.py")
print(wavelet)
#-----------------------------------------------------------------------------

coeffs, Ws = wavelet_coeffs(data,freqs,srate,nco)
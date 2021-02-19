#Wavelets and spectrogram
#Complex Morlet Wavelet Example-----------------------------------------------

import numpy as np
import matplotlib.pylab as plt

x = np.arange(-5,5, 0.001)
sigma = 2
f = 1
s = np.sqrt(np.pi*sigma)*np.exp(2*np.pi*1j*f*x)*np.exp(-x**2/sigma)

#data = np.loadtxt("avrgPerChannel.npy") ??
#import wavelet 
#ermöglicht nutzen von funktoinen aus wavelet.py (Wie??)

plt.figure()

plt.plot(x, np.real(s), label="real")
plt.plot(x, np.imag(s), label="imag")
plt.plot(x, np.abs(s), label="abs")
plt.legend()
plt.show()
#Inspect wavelets from Ws-----------------------------------------------------

wavelet = np.loadtxt("wavelet.py")
print(wavelet)

import wavelet.py
from wavelet import Ws
print(Ws)
#ERP Transformation-----------------------------------------------------------
import numpy as np
import pandas as pd

import avrgPerChannel.npy
import wavelet.npy
from wavelet import coeffs
from wavelet import wavelet_coeffs
from wavelet import data
from wavelet import freqs
from wavelet import srate
from wavelet import nco
from wavelet import Ws

#Can't import the real data
aPC_value = float(input(avrgPerChannel))

#Therefore I made my own data for testing
df=pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['a', 'b', 'c']) 
print(df)
df.transform(func = coeffs, Ws = wavelet_coeffs(data,freqs,srate,nco))
print(df)
#No output! Why?
#Outout is generated in a dedicated project tho.. ?

#Plotting the spectrogram-----------------------------------------------------
import numpy as np
import matplotlib.pylab as plt
from wavelet import power

df=pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['a', 'b', 'c']) 

fig,ax = plt.subplots()
cax = ax.imshow(power,cmap='jet',aspect='auto',\
extent=[0,power.shape[1],power.shape[0],0],origin='upper')
plt.show()

#-----------------------------------------------------------------------------


#Wavelets and spectrogram
#Complex Morlet Wavelet Example-----------------------------------------------

import numpy as np
import matplotlib.pylab as plt

x = np.arange(-5,5, 0.001)
sigma = 2
f = 1
s = np.sqrt(np.pi*sigma)*np.exp(2*np.pi*1j*f*x)*np.exp(-x**2/sigma)


plt.figure()

plt.plot(x, np.real(s), label="real")
plt.plot(x, np.imag(s), label="imag")
plt.plot(x, np.abs(s), label="abs")
plt.legend()
plt.xlabel('Time [ms]')
plt.ylabel('Amplitude [µ V]')
plt.show()
#Inspect wavelets from Ws-----------------------------------------------------
import numpy as np
from wavelet import wavelet_coeffs
from wavelet import wavelet_power
import matplotlib.pylab as plt

data=np.load(r'D:\Desktop\Studium\7. Semester\Brain-Machine Interfaces\Praktikum\bmi2020_tasks\MainContent\avrgPerChannel.npy')

print(data.shape)

plt.figure()
for i in range(10):
    for j in range(2):
        plt.plot(data[i,j,:])
#plt.plot(data[0,0,:])
#plt.plot(data[0,1,:])


freqs = np.arange(1, 256, 50)
srate = 256
nco = 2

coeffs, Ws = wavelet_coeffs(data,srate,freqs,nco)
#print(coeffs, Ws)
#print(len(coeffs))
#print(coeffs[0])

coeffs = np.array(coeffs)
power = wavelet_power(coeffs) 
print(power.shape)

#plt.plot(power[:,0,0,0])
#plt.plot(power[:,0,1,0])
#plt.plot(power[:,0,2,0])
#plt.plot(power[:,0,3,0])
#plt.plot(power[:,0,4,0])
#plt.show()

plt.xlabel('Samples')
plt.ylabel('Amplitude [µ V]')
plt.figure()
fig,ax = plt.subplots()
cax = ax.imshow(power[:,:,1,0],cmap='jet',aspect='auto',interpolation='none',\
extent=[0,power.shape[1],power.shape[0],0],origin='upper')

np.save(r'D:\Desktop\Studium\7. Semester\Brain-Machine Interfaces\Praktikum\bmi2020_tasks\MainContent\results_task1.npy', power)
plt.xlabel('Channel')
plt.ylabel('Samples')
plt.show()
#'Plot einer Frequenz mit wavelet für 10 samples'



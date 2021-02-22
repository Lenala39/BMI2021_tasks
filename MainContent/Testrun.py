
#Inspect wavelets from Ws-----------------------------------------------------
import numpy as np
from wavelet import wavelet_coeffs
from wavelet import wavelet_power
import matplotlib.pylab as plt

data=np.load(r'D:\Desktop\Studium\7. Semester\Brain-Machine Interfaces\Praktikum\bmi2020_tasks\MainContent\avrgPerChannel.npy')


freqs = np.arange(1, 256, 50)
srate = 256
nco = 2
fig,ax = plt.subplots()

coeffs, Ws = wavelet_coeffs(data,srate,freqs,nco)
coeffs = np.array(coeffs)
power = wavelet_power(coeffs) 
print(power.shape)

#plt.plot(power[:,0,0,0])
#plt.plot(power[:,0,1,0])
#plt.plot(power[:,0,2,0])
#plt.plot(power[:,0,3,0])
#plt.plot(power[:,0,4,0])

#plt.show()

cax = ax.imshow(power[:,:,1,0],cmap='jet',aspect='auto',interpolation='none',\
extent=[0,power.shape[1],power.shape[0],0],origin='upper')
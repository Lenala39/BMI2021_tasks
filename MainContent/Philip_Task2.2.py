#Task 2.2 - Bandpass filtering with FFT
#Calculation of cut-off frequency---------------------------------------------
import matplotlib.pylab as plt
import numpy as np

data=np.load(r'results_task1.npy')
s = data.shape
print(s)
data_filt = np.zeros(s)

#Transformation into frequency space------------------------------------------
#rescaling 256 samples to 205 scale
rescale = np.arange(0, 256, 256/205)
#filter limit setting (removing coefficients)---------------------------------
fltr1 = rescale>1
fltr2 = rescale<10
fltr = fltr1 & fltr2
for i in range(s[1]):
    for j in range(s[2]):
        for k in range (s[3]):
            coeffs=np.fft.fft(data[:,i,j,k])
            coeffs_filt = fltr*coeffs
            data_filt[:,i,j,k]=np.real(np.fft.ifft(coeffs_filt))
            
# 1 value = 1/800ms (x axis) after first transformation
#plt.figure()
#plt.plot(coeffs)

#print(fltr)
#plt.figure()
plt.plot(rescale,fltr*coeffs)

#Inverse Transformation into time-amplitude plane-----------------------------
plt.figure()
plt.plot(data_filt[:,5,1,0])

#origin data plot for comparison
plt.plot(data[:,5,1,0])
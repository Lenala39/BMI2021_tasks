#ERP Transformation-----------------------------------------------------------
import numpy as np
from wavelet import *

#Can't import the real data
aPC_value = float(input(avrgPerChannel))

#Therefore I made my own data for testing
testdata = np.random.randint(0, 100, size=(30, 10, 2))
print(testdata)

from wavelet import wavelet_coeffs
with open("D:\Desktop\Studium\7. Semester\Brain-Machine Interfaces\Praktikum\bmi2020_tasks\MainContent", "rb") as infile:
    data = np.load(infile)
srate = 256
freqs = 1
nco = 2

testdata.transform(func = wavelet_coeffs, Ws = wavelet_coeffs(data,freqs,srate,nco))
print(testdata)

aPC_value.transform(func = wavelet_coeffs, Ws = wavelet_coeffs(data,freqs,srate,nco))
print(aPC_value)

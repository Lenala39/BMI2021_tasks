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

aPC_value = float(input(avrgPerChannel))

df=pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['a', 'b', 'c']) 
print(df)
df.transform(func = coeffs, Ws = wavelet_coeffs(data,freqs,srate,nco))
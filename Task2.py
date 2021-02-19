import numpy as np
from toolbox import wavelet
import os

cwd = os.getcwd()
print(cwd)
full_path = os.path.join(cwd, "data", "avrgPerChannel.npy")
with open(full_path, "rb") as infile:
    data = np.load(infile)

# should be epochs x samples x channels
# channel, target/non-target, sample amount
print(data.shape)

c, ws = wavelet.wavelet_coeffs(data=data, srate=256, freqs=[1, 2, 3], nco=2)
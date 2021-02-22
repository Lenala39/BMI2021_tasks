import numpy as np
from math import floor
from math import ceil
from numpy import matlib
from math import sqrt
from numpy import linalg
import mne.time_frequency.tfr as tfr



def wavelet_coeffs(data,srate,freqs,nco=2):
    """
    Wavelet transform with complex morlet wavelet (i.e., a sinusiodial with a
    gaussian window). Useful to analyse signals in the oscillatory domain.
    Currently requires the mne toolbox to compute the convolution of the signal
    with the wavelet (this may change in the future).

    INPUT
        data    should be (epochs,samples,channels) OR single-channel vector 
                ...
    OUTPUT
        coeffs  complex-valued wavelet coefficients
        Ws      the wavelets 
    """

    # Generate the scaled wavelets for each frequency
    Ws = list()
    for k,f in enumerate(freqs):

        sigma_tf = nco/(2*np.pi*f) #f: frequency in Hz

        t = np.arange(0,5*sigma_tf,1/srate)
        t = np.r_[-t[::-1],t[1:]]

        osc = np.exp(2.*1j*np.pi*f*t)
        gauss = np.exp(-t**2/(2.*sigma_tf**2))
        W_tf = osc*gauss
        W_tf /= sqrt(.5)*linalg.norm(W_tf.ravel())
        Ws.append(W_tf)

    # Do the actual transform (e.g., convolve the signal with the  wavelets)
    # Caveat: This is currently only the single-job version of cwt !!!
    isVector = False
    if data.ndim==1:
        data = np.array([data,data])
        isVector = True

    if data.ndim==3:
        channels = data.shape[2]
        coeffs = list()
        for c in range(channels):
            dummy = tfr.cwt(data[:,:,c],Ws)
            coeffs.append(dummy)
    else:
        coeffs = tfr.cwt(data,Ws)

    if isVector:
        coeffs = np.squeeze(coeffs[0,:,:])
        print("Removed dummy dimension from coeffs!")

    return coeffs, Ws


def  wavelet_power(coeffs,logpower=False):
    """
    Obtain the time-frequency-bandpower representation from wavelet coefficients.

    """
    if 'numpy' in str(type(coeffs)):
        power = (coeffs*coeffs.conj()).real
        print("Received numpy array for processing!")
    else:
        epochs = coeffs[0].shape[0]
        power = list()
        channels = len(coeffs)
        for c in range(channels):
            dummy = np.zeros(coeffs[0].shape,dtype=np.float64)
            for i in range(epochs):
                dummy[i,:,:] = (coeffs[c][i,:,:]*coeffs[c][i,:,:].conj()).real
            power.append(dummy)

    if logpower:
        power = 10*np.log10(power)

    return power



def wavelet_phase(coeffs):
    """
    Obtain the time-frequency-phase representation from wavelet coefficients.

    """
    if 'numpy' in str(type(coeffs)):
        phase = np.angle(coeffs)
    else:
        epochs = coeffs[0].shape[0]
        phase = list()
        channels = len(coeffs)
        for c in range(channels):
            dummy = np.zeros(coeffs[0].shape,dtype=np.float64)
            for i in range(epochs):
                dummy[i,:,:] = np.angle(coeffs[c][i,:,:])
            phase.append(dummy)

    return phase



def toysig(freqs, sine=True, fac=None, noise=0, srate=256, scale=20, drift=False):
    """
    Generate a simulated oscillatory signal (1 channel).
    Useful for testing spectral methods (e.g., wavelets).
    """
    if fac is None:
        fac = (int)(floor(srate/2))

    freqs  = freqs.flatten()
    num = freqs.shape[0]*fac

    # signal with noise (gaussian)
    X = noise*np.matlib.randn(num)
    # Use this to add drift (simulated non-highpass filtered data):
    if drift:
        X -= 0.003*np.arange(0,num)

    X = np.asarray(X).flatten()

    for j in range(freqs.shape[0]):
        v = np.arange(1+j*fac,(j+1)*fac)
        if sine:
            X[v] = X[v]+np.sin(freqs[j]*2*np.pi*v/srate)*scale
        else:
            X[v] = X[v]+np.cos(freqs[j]*2*np.pi*v/srate)*scale

    return X

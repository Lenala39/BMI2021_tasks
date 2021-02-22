import numpy as np
import scipy.io as sio


def ecog_load_data_train():
    ##############################################
    #       eCoG Motor Imagery TRAIN data        #
    ##############################################
    datafile = "/home/lena/Apps/bmi2020_tasks/data/ecog_mi/Competition_train.mat"
    struct = sio.loadmat(datafile)
    training_data = struct['X'].astype(dtype=np.float64) # [278 trials x 64 channels x 3000 samples]
    training_label = struct['Y'].flatten() # -1 / 1
    return training_data, training_label

def ecog_load_data_test():
    ##############################################
    #       eCoG Motor Imagery TEST data        #
    ##############################################
    datafile = "/home/lena/Apps/bmi2020_tasks/data/ecog_mi/Competition_test.mat"
    labelfile = "/home/lena/Apps/bmi2020_tasks/data/ecog_mi/truelabel.mat"
    struct = sio.loadmat(datafile)
    testing_data = struct['X'].astype(dtype=np.float64) # [100 trials x 64 channels x 3000 samples]
    struct = sio.loadmat(labelfile)
    true_label = struct['truelabel'].flatten()
    return testing_data, true_label



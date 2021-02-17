import numpy as np
import scipy.io as sio

##############################################
#       eCoG Motor Imagery TRAIN data        #
##############################################
datafile = "/vol/pgbci/tools/data/ecog_mi/Competition_train.mat"
struct = sio.loadmat(datafile)
training_data = struct['X'].astype(dtype=np.float64) # [278 trials x 64 channels x 3000 samples]
training_label = struct['Y'].flatten() # -1 / 1


##############################################
#       eCoG Motor Imagery TEST data        #
##############################################
datafile = "/vol/pgbci/tools/data/ecog_mi/Competition_test.mat"
labelfile = "/vol/pgbci/tools/data/ecog_mi/truelabel.mat"
struct = sio.loadmat(datafile)
testing_data = struct['X'].astype(dtype=np.float64) # [100 trials x 64 channels x 3000 samples]
struct = sio.loadmat(labelfile)
true_label = struct['truelabel'].flatten()



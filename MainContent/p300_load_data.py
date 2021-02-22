import numpy as np

##############################################
#           P300 Speller data                #
##############################################
datafile = "/vol/pgbci/vlbmi_ws2019-20/data/p3bci_data.npz"
D = np.load(datafile, allow_pickle=True)
data = D['data']
onsets = D['onsets']
timestamps = D['timestamps']
flashseq = D['flashseq']
targets = D['targets']

from __future__ import division
import numpy as np
from math import floor
from numpy import matlib
from scipy import linalg

def downsample(data,fac,avg="pick",axis=0):
    """
    INPUT
        data    data array sized [trials x samples x channels] OR matrix
                 sized [trials x channels]
        fac     downsampling factor
        avg     string 'avg' (averaging, default) or 'pick' (no avg.)
    OUTPUT
        X       the downsampled signal array sized
                  [trials x floor(samples/fac) x channels]
    """

    tmp = np.zeros((2,4))
    dims = data.shape
    X = []

    if len(dims)==3:
        samples = dims[1]
        nums = (int)(floor(samples/fac))
        if avg=="avg":
            X = np.zeros((dims[0],nums,dims[2]),np.float64)
            for i in range(nums):
                X[:,i,:] = np.mean(data[:,i*fac:fac+i*fac,:],axis=1)
        elif avg=="pick":
            X = data[:,range(fac-1,fac*nums,fac),:]

    elif len(dims)==2:
        samples = dims[0]
        nums = (int)(floor(samples/fac))
        if avg=="avg":
            X = np.zeros((nums,dims[1]),np.float64)
            for i in range(nums):
                X[i,:] = np.mean(data[i*fac:fac+i*fac,:],axis=0)

        elif avg=="pick":
            X = data[range(fac-1,fac*nums,fac),:]
    else:
        print("wrong structure of input array! Required data array sized"
        + "trials x samples x channels OR matrix sized trials x channels.")

    return X


def cov_shrinkage(data,axis=0):
    """
    Implements algorithm for a shrinkage estimate of the covariance matrix.
    Uses the procedure as described in Schaefer and Strimmer (2005), which follows the
    Ledoit-Wolf Theorem.
    See P. 11, Target B  and Appendix
    INPUT
       data     ndarray (samples,dims) when axis=0 or (dims,samples) and axis=1
       axis     Whether features are contained in colums (default) or rows
    OUTPUT
       U       shrinkage estimate of covariance matrix (dims,dims)
    """
    print('covshape', data.shape)
    if axis==1:
        data = np.transpose(data)

    [rows,cols] = data.shape

    x_bar = np.sum(data,axis=0)/rows
    x_bar = np.matlib.repmat(x_bar,rows,1)


    w_ki = data-x_bar #factors for w_ijk

    s_ij = np.zeros((cols,cols),dtype=np.float64)
    var_s_ij = np.zeros((cols,cols),dtype=np.float64)

    #TODO MAKE column loops parallel
    for i in range(cols):
        for j in range(i,cols):
            w_kij = w_ki[:,i] * w_ki[:,j]
            #print(2, w_kij.shape)
            w_bar_ij = np.sum(w_kij,axis=0)/rows
            s_ij[i,j] = w_bar_ij * (rows/(rows-1))
            var_s_ij[i,j] = np.sum((w_kij - w_bar_ij)**2) * (rows/(rows-1)**3)


    mu = np.mean(np.diag(s_ij))

    T = np.eye(cols) * mu

    lambdaval = np.sum(np.reshape(np.triu(var_s_ij,0),(1,-1)))

    denom = np.sum(np.reshape(np.power(np.triu(s_ij,1),2),(1,-1))) + np.sum(np.power(np.diag(s_ij)-mu,2))

    lambdaval = lambdaval/denom

    U = lambdaval*T + (1-lambdaval)*np.cov(data,rowvar=False)

    return U


def calcTPscore(ys, flashseq, target, plot=False):
    """
    :param ys: real-valued outputs of the LDA classifier with shape (subtrials, row_column_number)
    :param flashseq: index of row/column flashing in epoch with shape (subtrials, row_column_number)
    :param target: tuple (row, column) of the target element
    :param plot: plot brightness matrix
    :return:
    """

    assert ys.shape == flashseq.shape,\
        f'ys and flashseq shapes should be equals, {ys.shape} != {flashseq.shape}'
    assert 0 <= target[0] <= 5
    assert 6 <= target[1] <= 11

    # print('target:', target)

    subtrials_number, row_col_number = ys.shape

    assert row_col_number % 2 == 0,\
        f'The total numbers of rows and column should be multiple of 2, but we have {row_col_number}'
    rows_number = ys.shape[1] // 2
    cols_number = rows_number

    # vector of M matrices for every subtrial
    M_vec = np.zeros((subtrials_number, rows_number, cols_number))

    subtrial_index = None

    # loop over the subtrials until the precision
    for subtrial_i in range(subtrials_number):

        for epoch_i in range(N_ROW_COL):
            flash_i = flashseq[subtrial_i, epoch_i]
            pred_score = ys[subtrial_i, epoch_i]
            M = M_vec[subtrial_i]

            # row flash, add value to row
            if flash_i < 6:
                M[flash_i,:] += pred_score
            # column flash, add value to column
            else:
                M[:,flash_i - 6] += pred_score

        # don't sum it up at the first iteration
        if subtrial_i > 0:
            M += M_vec[subtrial_i - 1]

        max_row, max_col = np.unravel_index(M.argmax(), M.shape)
        # print(max_row, max_col + rows_number)

        if plot:
            M_scaled = utils.scale(M_vec[subtrial_i].ravel(), 0, 1).reshape(rows_number, cols_number)
            plot_brightness_matrix(M_scaled, subtrial_i, target, row_col_number)

        # NOTE: columns of the P300 are indexed after rows
        if max_row == target[0] and max_col + rows_number == target[1]:
            # print('Found correct row and col in subtrial No. ', subtrial_i)
            subtrial_index = subtrial_i
            break

    if subtrial_index is not None:
        TPscore = M_vec[subtrial_index, max_row, max_col]
        M_thr = utils.scale(M_vec[subtrial_index].ravel(), 0, 1).sum()
    else:
        TPscore  = 0
        M_thr = 0

    return TPscore, M_thr, subtrial_index




def scale(data,lower=-1,upper=1,axis=0):
    """
    Scales the columns of a matrix to a defined interval.
    INPUT
        data        [trials x channels] Data will be scaled column-wise (default,axis=0)
                    OR single column vector
        lower       lower bound (default -1)
        upper       upper bound (default 1)
        axis 
      Lower and upper are included in the interval.
    OUTPUT
        data        Scaled data same size as input.
    """
    if lower >= upper:
        print('Lower bound must be smaller than upper bound!')
        
    if axis == 1:
        data = np.transpose(data)
    
    if np.ndim(data)==2:
        maxval,maxix = data.max(0),data.argmax(0)
        minval,minix = data.min(0),data.argmin(0)
        [rows,cols] = data.shape
        scaled = (data-np.ones((rows,1))*minval)*(np.ones((rows,1))*((upper-lower)*np.ones((1,cols))/(maxval-minval)))+lower
        if axis==1:
            scaled = np.transpose(scaled)
    elif np.ndim(data)==1:
        maxval,maxix = data.max(),data.argmax()
        minval,minix = data.min(),data.argmin()
        elem = data.shape
        scaled = (data-np.ones((elem))*minval)*(np.ones((elem))*((upper-lower))/(maxval-minval))+lower
    else:
        print("Input data must have either 1 or 2 dimensions.")
        scaled = -1

    return scaled


def calc_confusion(testlabel,truelabel,pos=1,neg=0,mat=False):
    """
    Calc the confusion matrix. Currently only for binary class.
    """
    # Init confmat entries
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    testlabel = testlabel.flatten()
    truelabel = truelabel.flatten()
    posixs = np.nonzero(truelabel==pos)[0]
    negixs = np.nonzero(truelabel==neg)[0]

    for l in range(posixs.shape[0]):
        if testlabel[posixs[l]]==truelabel[posixs[l]]:
            TP+=1
        else:
            FN+=1
    for l in range(negixs.shape[0]):
        if testlabel[negixs[l]]==truelabel[negixs[l]]:
            TN+=1
        else:
            FP+=1

    if mat:
        confmat = np.array([[TP,FP],[FN,TN]])
        return confmat
    else:
        return TP,FP,FN,TN


def fda_train(data, label):
    """
    Calc weight vector and bias for FDA classifier.
    INPUT
        data    matrix shaped [epochs, features]
        label   vector of class labels with 1 for the target class.
    OUTPUT
    
    """
    #targets = data[label==1, :]
    #non_targets = data[label!=1, :]
    targets = None
    non_targets = None
    for i in range(len(data)):
        if label[i] == 1:
            try:
                targets = np.vstack((targets, data[i]))
            except ValueError: # first iteration -> first vec is start of data-array
                targets = data[i]
        else:
            try:
                non_targets = np.vstack((non_targets, data[i]))
            except ValueError: # first iteration -> first vec is start of data-array
                non_targets = data[i]

    # mean of each class
    target_mean = np.mean(targets, axis=0)
    non_target_mean = np.mean(non_targets, axis=0)
    
    # covariance matrix within each class (scatter matrix)
    St = cov_shrinkage(targets)
    Snt = cov_shrinkage(non_targets)
    
    # (within-class-scatter)
    Sw = St + Snt
    
    invSw = linalg.inv(Sw)
    # Rayleigh coefficient
    fda_w = invSw @ (target_mean-non_target_mean).T
    print("fda.w size: ",fda_w.shape)

    fda_target_mu = np.dot(target_mean, fda_w)
    fda_non_target_mu = np.dot(non_target_mean, fda_w)
    # Fisher criterium
    fda_b = (fda_target_mu + fda_non_target_mu) / 2
    
    return fda_w, fda_b


def fda_test(data, fda_w, fda_b):
    """
    Calc the class scores for a test data set.
    INPUT
        data    matrix shaped [epochs, features]
        fda_w   weight vector shaped [features]
        fda_b   scalar bias
    OUTPUT
        scores  class scores shaped [epochs]
        label   class label shaped [epochs]
    """
    scores = data @ fda_w
    scores = (scores - scores.min()) / np.ptp(scores)
    label = np.sign(scores - fda_b)
    label = np.where(label < 0, 0, label) # convert -1 to 0
    return scores, label
 

def calc_AUC(roc):
    """
    Calc the Area under Curve (AUC) based on a/some given ROC(s).
    """
    if len(roc.shape)==3:
        auc = np.zeros((roc.shape[0],),dtype=np.float64)
        for f in range(roc.shape[0]):
            auc[f] = np.trapz(roc[f,0,:],roc[f,1,:])
    else:
        auc = np.trapz(roc[0,:],roc[1,:])

    return auc
    

def chan4plot():
    import mne
    import os
    # !!! Please adjust this path to point to the location where your p300speller.txt and biosemi32_mne-montage.txt is stored. 
    chanpath = "./"
    # Load channel infos (names + locations) and create info object for topo plots
    with open(os.path.join(chanpath,'p300speller.txt')) as f:
        chanlabel = f.read().splitlines()
    M = mne.channels.read_montage(os.path.join(chanpath,'biosemi32_mne-montage.txt'))  
    eeginfo = mne.create_info(chanlabel,256,'eeg',M)     
    return eeginfo
    
    
    
    
    

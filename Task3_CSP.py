from toolbox import utils, ecog_load_data
import mne.filter as filter
import numpy as np
import scipy.linalg as linalg
from matplotlib import pyplot as plt
import os



def filter_data_ecog(training_data_ecog):
    # sfreq = sample frequency, iir = forward-backward filtering (via filtfilt)
    filtered_data_ecog = filter.filter_data(data=training_data_ecog , sfreq=1000, l_freq=8, 
                                        h_freq=30, method="iir", n_jobs=6)
    return filtered_data_ecog


def group_data(filtered_data_ecog, training_label_ecog):
    # initialize ndarrays with correct dimensions -> 139 trials pos/neg
    data_ecog_pos = []
    data_ecog_neg = []
    
    # iterate over the filtered data
    for i in range(len(filtered_data_ecog)):
        # check the corresponding label -> negative = append to negative array
        if training_label_ecog[i] == -1:
            data_ecog_neg.append(filtered_data_ecog[i])
        elif training_label_ecog[i] == 1:
            data_ecog_pos.append(filtered_data_ecog[i])
        else:
            print("ERROR")

    # Convert back to ndarray
    data_ecog_neg = np.array(data_ecog_neg)
    data_ecog_pos = np.array(data_ecog_pos)
    assert data_ecog_neg.shape == (139, 64, 3000)
    return data_ecog_neg, data_ecog_pos

def input_formatting(data_ecog_neg, data_ecog_pos):
    # average over the epochs (trials) -> 278 trials, 64 channels, 3000 samples
    # result should be 64x3000 matrix
    filtered_data_ecog_pos = np.mean(data_ecog_pos, axis=0)
    filtered_data_ecog_neg = np.mean(data_ecog_neg, axis=0)
    return filtered_data_ecog_neg, filtered_data_ecog_pos

def compute_covariance_matrices(filtered_data_ecog_neg, filtered_data_ecog_pos):
    cov_pos = np.cov(filtered_data_ecog_pos)
    cov_neg = np.cov(filtered_data_ecog_neg)
    #cov_pos_norm = np.corrcoef(filtered_data_ecog_pos)
    #cov_neg_norm = np.corrcoef(filtered_data_ecog_neg)

    between_class_cov = cov_pos + cov_neg
    return cov_neg, cov_pos, between_class_cov

def compute_eigenvalues_and_vectors(between_class_cov):
    # compute eigenvalues (lamdba) and eigenvectors (V)
    eigenvalues, eigenvectors = np.linalg.eig(between_class_cov)

    # assert that factorization holds
    factor_1 = np.dot(eigenvectors, eigenvalues)
    transposed_vecs = eigenvectors.transpose()
    factor_2 = np.dot(eigenvalues, transposed_vecs)
    assert between_class_cov.all() == np.dot(factor_1, factor_2).all()

    return eigenvalues, eigenvectors

def whitening_transformation(eigenvalues, eigenvectors):
    inverted_eigenvalues = np.flip(eigenvalues)
    P_normalized = np.dot(np.sqrt(np.diag(inverted_eigenvalues)), np.transpose(eigenvectors))
    
    return P_normalized


def factorize_cov_matrix(P_normalized, cov_neg, cov_pos):
    # factorize cov matrices
    S_neg = np.dot(P_normalized, np.dot(cov_neg, np.transpose(P_normalized)))
    S_pos = np.dot(np.dot(P_normalized, cov_pos), np.transpose(P_normalized))

    assert S_neg.all() == np.dot(np.dot(cov_neg, P_normalized), P_normalized.transpose()).all()
    # TODO: assert that eigenvalues + eigenvalues = identity
    return S_neg, S_pos

def sort_eigenvectors(S_neg, S_pos):
    # compute generalized eigenvector problem
    common_eigenvalues, common_eigenvectors = linalg.eig(S_neg, S_pos)
    # create an ordering with argsort
    ordering = np.argsort(common_eigenvalues)
    # apply sorting on the eigenvalues and -vectors
    common_eigenvalues = common_eigenvalues[ordering]
    common_eigenvectors = common_eigenvectors[ordering]

    return common_eigenvalues, common_eigenvectors

def data_preprocessing(training_data_ecog, training_label_ecog):
    # 1.0 apply mne filter method to data
    filtered_data_ecog = filter_data_ecog(training_data_ecog)
    # 2.0 group data according to label
    ecog_neg, ecog_pos = group_data(filtered_data_ecog, training_label_ecog)
    return ecog_neg, ecog_pos
    
def CSP(ecog_neg, ecog_pos, cols=3):
    # 2.1 "flatten" input into correct shape
    ecog_neg, ecog_pos = input_formatting(ecog_neg, ecog_pos)
    # 2.2 covariance matrices
    cov_neg, cov_pos, between_class_cov = compute_covariance_matrices(ecog_neg, ecog_pos)
    # 2.3 factorize eigenvalues 
    eigenvals, eigenvecs = compute_eigenvalues_and_vectors(between_class_cov)
    # 2.5 normalization with withening transform
    P = whitening_transformation(eigenvalues=eigenvals, eigenvectors=eigenvecs)
    # 2.6 factorization
    S_neg, S_pos = factorize_cov_matrix(P_normalized=P, cov_neg=cov_neg, cov_pos=cov_pos)
    # 2.7 sort in descending order
    common_eigenvalues, common_eigenvectors = sort_eigenvectors(S_neg, S_pos)

    # 2.8 compute projection matrix W
    W = np.dot(np.transpose(common_eigenvectors),P).transpose() # todo: nochmal T
    
    # REDUCE W
    # makes list from 0 to cols-1 (e.g. want 3 cols -> [0,1,2])
    keep_beginning = np.arange(0, cols)
    # makes list from -cols to -1 (e.g. want 3 cols -> [-3, -2, -1])
    keep_end = np.arange(0-cols, 0)

    # slice away only wanted number of cols (see lists above)
    first_two_cols = W[:, list(keep_beginning)]
    last_two_cols =W[:, list(keep_end)]
    
    # concatenate again
    W_reduced = np.concatenate((first_two_cols, last_two_cols), axis=1)

    return W_reduced
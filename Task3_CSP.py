from toolbox import utils, ecog_load_data
import mne.filter as filter
import numpy as np
import scipy.linalg as linalg
from matplotlib import pyplot as plt
import os



def filter_data_ecog(training_data_ecog):
    """Filters data using mne.filter
       Cutoff frequencies set to 8/30 Hz 
    Args:
        training_data_ecog (np.ndarray): contains the loaded data from .mat-files

    Returns:
        np.ndarray: filtered data
    """
    filtered_data_ecog = filter.filter_data(data=training_data_ecog , sfreq=1000, l_freq=8, 
                                        h_freq=30, method="iir", n_jobs=6)
    return filtered_data_ecog


def group_data(filtered_data_ecog, training_label_ecog):
    """Groups the filtered data according to the training labels (0,1)

    Args:
        filtered_data_ecog (np.ndarray): data to be grouped
        training_label_ecog (np.ndarray): corresponding labels

    Returns:
        np.ndarray(s): negative and positive instances in seperate np.ndarrays
    """
    # init lists to store entries 
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

    return data_ecog_neg, data_ecog_pos

def input_formatting(data_ecog_neg, data_ecog_pos):
    """ average over the epochs (trials) -> 278 trials, 64 channels, 3000 samples
        result should be 64x3000 matrix

    Args:
        data_ecog_neg (np.ndarray): array containing all negative instances
        data_ecog_pos (np.ndarray): array containing all positive instances

    Returns:
        np.ndarray(s): averaged arrays over axis 0 -> shape reduced by one dimension
    """
    # average over the trials (axis=0)
    averaged_data_pos = np.mean(data_ecog_pos, axis=0)
    averaged_data_neg = np.mean(data_ecog_neg, axis=0)

    return averaged_data_neg, averaged_data_pos

def compute_covariance_matrices(filtered_data_ecog_neg, filtered_data_ecog_pos):
    """Computes the covariance matrices of the respective input arrays

    Args:
        filtered_data_ecog_neg (np.ndarray): averaged, filtered negative instances in matric
        filtered_data_ecog_pos (np.ndarray): averaged, filtered negative instances in matric

    Returns:
        np.ndarray(s): covariance matrices for each class and the between-class-covariance matrix
    """
    # compute covariance matrix
    cov_pos = np.cov(filtered_data_ecog_pos)
    cov_neg = np.cov(filtered_data_ecog_neg)

    # compute sum of matrices -> between class covariance
    between_class_cov = cov_pos + cov_neg
    return cov_neg, cov_pos, between_class_cov

def compute_eigenvalues_and_vectors(between_class_cov):
    """Computes eigenvalues and vectors

    Args:
        between_class_cov (np.ndarray): between class covariance matrix

    Returns:
        np.ndarray(s): eigenvalues and vectors
    """

    # compute eigenvalues (lamdba) and eigenvectors (V)
    eigenvalues, eigenvectors = np.linalg.eig(between_class_cov)

    # assert that factorization holds
    factor_1 = np.dot(eigenvectors, eigenvalues)
    transposed_vecs = eigenvectors.transpose()
    factor_2 = np.dot(eigenvalues, transposed_vecs)
    assert between_class_cov.all() == np.dot(factor_1, factor_2).all()

    return eigenvalues, eigenvectors

def whitening_transformation(eigenvalues, eigenvectors):
    """Apply whitening transformation to recieve P

    Args:
        eigenvalues (np.ndarray): eigenvalues
        eigenvectors (np.ndarray): eigenvectors

    Returns:
        np.ndarray: Projection matrix P (normalized)
    """ 
    inverted_eigenvalues = np.flip(eigenvalues)
    P_normalized = np.dot(np.sqrt(np.diag(inverted_eigenvalues)), np.transpose(eigenvectors))
    
    return P_normalized


def factorize_cov_matrix(P_normalized, cov_neg, cov_pos):
    """Factorize Covariance matrices to respective eigenvectors

    Args:
        P_normalized (np.ndarray): projection matrix
        cov_neg (np.ndarray): covariance matrix for negative class
        cov_pos (np.ndarray): covariance matrix for positive class

    Returns:
        np.ndarray(s): eigenvalues as result of factorization (S1 and S2)
    """
    # factorize cov matrices
    S_neg = np.dot(P_normalized, np.dot(cov_neg, np.transpose(P_normalized)))
    S_pos = np.dot(np.dot(P_normalized, cov_pos), np.transpose(P_normalized))

    return S_neg, S_pos

def sort_eigenvectors(S_neg, S_pos):
    """Sort the eigenvectors from biggest to smallest value

    Args:
        S_neg (np.ndarray): Eigenvectors for negative class
        S_pos (np.ndarray): Eigenvectors for positive class

    Returns:
        np.ndarray: common eigenvalues and -vectors
    """
    # compute generalized eigenvector problem to find common eigen...
    common_eigenvalues, common_eigenvectors = linalg.eig(S_neg, S_pos)

    # create an ordering with argsort -> list of indices in correct order
    ordering = np.argsort(common_eigenvalues)
    
    # apply sorting on the eigenvalues and -vectors
    common_eigenvalues = common_eigenvalues[ordering]
    common_eigenvectors = common_eigenvectors[ordering]

    return common_eigenvalues, common_eigenvectors

def data_preprocessing(training_data_ecog, training_label_ecog):
    """Apply filtering and grouping to the data

    Args:
        training_data_ecog (np.ndarray): ecog data 
        training_label_ecog (np.ndarray): labels for the data

    Returns:
        np.ndarray(s): filtered and grouped data
    """
    # 1.0 apply mne filter method to data
    filtered_data_ecog = filter_data_ecog(training_data_ecog)
    # 2.0 group data according to label
    ecog_neg, ecog_pos = group_data(filtered_data_ecog, training_label_ecog)
    return ecog_neg, ecog_pos
    
def CSP(ecog_neg, ecog_pos, cols=3):
    """Compute the projection matrix W using CSP algorith,

    Args:
        ecog_neg (np.ndarray): negative class data
        ecog_pos (np.ndarray): positive class data
        cols (int, optional): Number of cols to keep on each side of W. Defaults to 3.

    Returns:
        np.ndarray: reduced projection matrix W
    """
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
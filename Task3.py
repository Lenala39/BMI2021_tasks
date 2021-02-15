from toolbox import utils, ecog_load_data
import mne.filter as filter
import numpy as np
import scipy.linalg as linalg

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
    print("Shape of input: ", data_ecog_neg.shape)
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
    
    #print("P: ", P_normalized.shape)
    #print("C: ", cov_neg.shape)
    #print("P_t: ", P_normalized.transpose().shape)
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
    
def CSP(ecog_neg, ecog_pos):
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
    print(W.shape)
    return W

def cross_validation_ecog(ecog_neg, ecog_pos):
    """
    Implements the split for the cross-validation for the eCoG data 

    Results in 5 tuples (one per fold) that each contain a tuple with the
    positive and negative instances 
    all_folds[i] = (training_data, testing_data) 
        with training_data = (neg_data, pos_data) 
        and testing_data = (neg_data, pos_data)

    ecog_neg: only negatively labeled instances (139, 64)
    ecog_pos: only positively labeled instances (139, 64)

    returns: all_folds (dict): dict of len 5 with the data for each fold split in training and test
    """
    all_folds = {}
    for i in range(1,6):
        # get the starting index for the test set 
        try:
            test_start = int(len(ecog_neg) * ((i-1)/5))
            # if it is the first fold -> start at 0
        except ZeroDivisionError:
            test_start = 0
        # end index is 1/5, 2/5, ... of the data 
        test_end = int(len(ecog_neg) * (i/5)) - 1 
        # make tuple for the test containing neg and pos arrays
        test = (ecog_neg[test_start:test_end], ecog_pos[test_start:test_end])
        print(f"Test set for fold number {i} starts at {test_start} and ends at {test_end}!")

        # concat the training data 
        # take sections before the test index (empty in first fold) and after (empty in last fold)
        train_neg = np.concatenate( (ecog_neg[0:test_start], (ecog_neg[test_end+1:138])) ) 
        train_pos = np.concatenate( (ecog_pos[0:test_start], (ecog_pos[test_end+1:138])) ) 
        assert len(train_neg) == len(train_pos)
        
        print("Shape of training data neg ", train_neg.shape)
        print(f"{len(train_neg)} training samples and {len(test[0])} test samples with a ratio of {round(len(test[0])/len(train_neg),2)}!")

        train = (train_neg, train_pos)
        # append to the folds
        all_folds[i] = (train, test)
    return all_folds

if __name__ == "__main__":
    # 0. load data from file
    training_data_ecog, training_label_ecog = ecog_load_data.ecog_load_data_train()

    # preprocessing
    # filtering using mne + grouping according to label
    ecog_neg, ecog_pos = data_preprocessing(training_data_ecog, training_label_ecog)
    print(ecog_neg.shape)

    all_folds = cross_validation_ecog(ecog_neg, ecog_pos)
    for fold, data in all_folds.items():
        training_data_neg = data[0][0]
        training_data_pos = data[0][1]
        
        # input formatting (average over all epochs)
        # neg, pos = input_formatting(training_data_neg, training_data_pos)

        # compute projection matrix W
        W = CSP(training_data_neg, training_data_pos)
        first_two_cols = W[:, [0,1]]
        last_two_cols =W[:, [-2,-1]]
        W_reduced = np.concatenate((first_two_cols, last_two_cols), axis=1)

        # apply W to training and test
        for i in range(len(training_data_ecog)):
            epoch = training_data_ecog[i]
            epoch_label = training_label_ecog[i]
            out = np.dot(epoch.transpose(), W_reduced)
            featurevec = np.log(np.var(out, axis=0))
from toolbox import utils, ecog_load_data
import mne.filter as filter
import numpy as np
import scipy.linalg as linalg
from matplotlib import pyplot as plt

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
    print("input formatting")
    filtered_data_ecog_pos = np.mean(data_ecog_pos, axis=0)
    filtered_data_ecog_neg = np.mean(data_ecog_neg, axis=0)
    return filtered_data_ecog_neg, filtered_data_ecog_pos

def compute_covariance_matrices(filtered_data_ecog_neg, filtered_data_ecog_pos):
    print("compute cov")
    cov_pos = np.cov(filtered_data_ecog_pos)
    cov_neg = np.cov(filtered_data_ecog_neg)
    #cov_pos_norm = np.corrcoef(filtered_data_ecog_pos)
    #cov_neg_norm = np.corrcoef(filtered_data_ecog_neg)

    between_class_cov = cov_pos + cov_neg
    return cov_neg, cov_pos, between_class_cov

def compute_eigenvalues_and_vectors(between_class_cov):
    print("compute eigenvals")
    # compute eigenvalues (lamdba) and eigenvectors (V)
    eigenvalues, eigenvectors = np.linalg.eig(between_class_cov)

    # assert that factorization holds
    factor_1 = np.dot(eigenvectors, eigenvalues)
    transposed_vecs = eigenvectors.transpose()
    factor_2 = np.dot(eigenvalues, transposed_vecs)
    assert between_class_cov.all() == np.dot(factor_1, factor_2).all()

    return eigenvalues, eigenvectors

def whitening_transformation(eigenvalues, eigenvectors):
    print("whitening")
    inverted_eigenvalues = np.flip(eigenvalues)
    P_normalized = np.dot(np.sqrt(np.diag(inverted_eigenvalues)), np.transpose(eigenvectors))
    
    return P_normalized


def factorize_cov_matrix(P_normalized, cov_neg, cov_pos):
    print("factorize cov matrices")
    # factorize cov matrices
    S_neg = np.dot(P_normalized, np.dot(cov_neg, np.transpose(P_normalized)))
    S_pos = np.dot(np.dot(P_normalized, cov_pos), np.transpose(P_normalized))

    assert S_neg.all() == np.dot(np.dot(cov_neg, P_normalized), P_normalized.transpose()).all()
    # TODO: assert that eigenvalues + eigenvalues = identity
    return S_neg, S_pos

def sort_eigenvectors(S_neg, S_pos):
    print("sort eigenvals")
    # compute generalized eigenvector problem
    common_eigenvalues, common_eigenvectors = linalg.eig(S_neg, S_pos)
    # create an ordering with argsort
    ordering = np.argsort(common_eigenvalues)
    # apply sorting on the eigenvalues and -vectors
    common_eigenvalues = common_eigenvalues[ordering]
    common_eigenvectors = common_eigenvectors[ordering]

    return common_eigenvalues, common_eigenvectors

def data_preprocessing(training_data_ecog, training_label_ecog):
    print("data filtering and grouping")
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
    print("compute W")
    W = np.dot(np.transpose(common_eigenvectors),P).transpose() # todo: nochmal T
    
    # reduce projection matrix to 2 cols on each side
    first_two_cols = W[:, [0,1]]
    last_two_cols =W[:, [-2,-1]]
    W_reduced = np.concatenate((first_two_cols, last_two_cols), axis=1)


    return W_reduced

def cross_validation_ecog(ecog_neg, ecog_pos, print_status=False):
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
        

        # concat the training data 
        # take sections before the test index (empty in first fold) and after (empty in last fold)
        train_neg = np.concatenate( (ecog_neg[0:test_start], (ecog_neg[test_end+1:138])) ) 
        train_pos = np.concatenate( (ecog_pos[0:test_start], (ecog_pos[test_end+1:138])) ) 
        assert len(train_neg) == len(train_pos)

        temp = {
            "train_neg": train_neg,
            "train_pos": train_pos,
            "test_neg": ecog_neg[test_start:test_end],
            "test_pos": ecog_neg[test_start:test_end]
        }

        # append to the folds
        all_folds[i] = temp

        if print_status:
            print(f"Test set for fold number {i} starts at {test_start} and ends at {test_end}!")
            print("Shape of training data neg ", train_neg.shape)
            print(f"{len(train_neg)} training samples and {len(test[0])} test samples with a ratio of {round(len(test[0])/len(train_neg),2)}!")
            print(f"appended to fold {i}")
        # TODO: change back when handing in: Faster Debugging
        return all_folds
    return all_folds

def apply_projection_matrix(data, W):
    # apply W to training and test
    projected_vectors = None
    for i in range(len(data)):
        epoch = data[i]
        #epoch_label = labels[i]
        out = np.dot(epoch.transpose(), W)
        featurevec = np.log(np.var(out, axis=0))
        try:
            projected_vectors = np.vstack((projected_vectors, featurevec))
        except ValueError: # first iteration -> first vec is start of data-array
            projected_vectors = featurevec
    return projected_vectors    

def concat_split_data(training_data_neg, training_data_pos, testing_data_neg, testing_data_pos):
    # concatenate all pos/neg in train or test to apply to W    
    fold_training_data = np.concatenate( (training_data_neg, training_data_pos) )
    fold_testing_data = np.concatenate( (testing_data_neg, testing_data_pos) )
    
    # create label-vecs for the concatenated train/test vecs in the fold
    label_neg_train = np.full((len(training_data_neg),1), 0)
    label_pos_train = np.full((len(training_data_pos), 1), 1)
    labels_train = np.concatenate( (label_neg_train, label_pos_train ) ).flatten()

    assert len(fold_training_data) == len(labels_train)
    
    # shuffle around the instances
    ordering_training = np.arange(len(fold_training_data))
    np.random.shuffle(ordering_training)
    fold_training_data = fold_training_data[ordering_training]
    labels_train = labels_train[ordering_training]
    
    # concat the labels
    label_neg_test = np.full((len(testing_data_neg),1), 0)
    label_pos_test = np.full((len(testing_data_pos),1), 1)
    labels_test = np.concatenate( (label_neg_test, label_pos_test ) ).flatten()
    
    assert len(fold_testing_data) == len(labels_test)

    # shuffle around the instances
    ordering_testing = np.arange(len(fold_testing_data))
    np.random.shuffle(ordering_testing)
    fold_testing_data = fold_testing_data[ordering_testing]
    labels_test = labels_test[ordering_testing]

    return fold_training_data, fold_testing_data, labels_train, labels_test


def compute_tps_and_fps(ys, absolute=False):
    bias = np.linspace(0,1,100)

    true_positives = []
    false_positives = []
    for i in bias:
        # make everything below bias 0, rest 1
        ys_bias = [0 if elem < i else 1 for elem in list(ys)]
        ys_bias = np.array(ys_bias)
        TP,FP,FN,TN = utils.calc_confusion(ys_bias, labels_test)
        
        # return the absolute number of TP and FPs
        if absolute:
            true_positives.append(TP)
            false_positives.append(FP)
        # return the TP/FP-rate as given by the formula
        else:
            true_positives.append(TP / (TP + FN))
            false_positives.append(FP / (TN + FP))

    return true_positives, false_positives


if __name__ == "__main__":
    # 0. load data from file
    training_data_ecog, training_label_ecog = ecog_load_data.ecog_load_data_train()

    # preprocessing
    # filtering using mne + grouping according to label
    ecog_neg, ecog_pos = data_preprocessing(training_data_ecog, training_label_ecog)
    all_folds = cross_validation_ecog(ecog_neg, ecog_pos)
    for fold, data in all_folds.items():

        print(f"Working on fold {fold}:")

        training_data_neg = data["train_neg"]
        training_data_pos = data["train_pos"]
        testing_data_neg = data["test_neg"]
        testing_data_pos = data["test_pos"]
        # input formatting (average over all epochs)
        # neg, pos = input_formatting(training_data_neg, training_data_pos)

        # compute (already reduced) projection matrix W
        # W = CSP(training_data_neg, training_data_pos)

        # TODO: remove when handing in: only for quicker debugging!!!
        with open("/home/lena/Apps/bmi2020_tasks/W_save.npy", "rb") as infile:
            W = np.load(infile)
        
        # get the concatenated data (not split in pos and neg) + labels
        fold_training_data, fold_testing_data, labels_train, labels_test = concat_split_data(training_data_neg, training_data_pos, testing_data_neg, testing_data_pos)

        # Project the training and test data
        projected_vecs_train = apply_projection_matrix(data=fold_training_data, W=W)
        projected_vecs_test = apply_projection_matrix(data=fold_testing_data, W=W)

        # apply the fda classifier
        fda_w, fda_b = utils.fda_train(data=projected_vecs_train, label=labels_train)
        
        # compute the ys (classifications)
        # ys = np.matmul(projected_vecs_test, fda_w)
        
        # normalize the ys to [0,1]
        # ys = (ys - ys.min()) / np.ptp(ys)

        
        # compute test classification over 100 bias values
        true_positives = []
        false_positives = []
        true_positives_rate = []
        false_positives_rate = []
        bias = np.linspace(0,1,10)
        for b in bias:
            scores, labels = utils.fda_test(projected_vecs_test, fda_w, fda_b=b)
            TP,FP,FN,TN = utils.calc_confusion(labels, labels_test)
            true_positives_rate.append(TP / (TP + FN))
            try:
                false_positives_rate.append(FP / (TN + FP)) 
            except ZeroDivisionError:
                false_positives_rate.append(0.0)
            true_positives.append(TP)
            false_positives.append(FP)

        # true_positives, false_positives = compute_tps_and_fps(ys=ys, absolute=True)
        plt.plot(true_positives, false_positives)
        #plt.plot(true_positives_rate, false_positives_rate)
        plt.show(block=True)

        roc = np.vstack( (true_positives, false_positives ))
        auc = utils.calc_AUC(roc)
        print(auc)
            
            

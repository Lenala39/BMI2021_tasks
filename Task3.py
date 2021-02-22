from toolbox import utils, ecog_load_data
import mne.filter as filter
import numpy as np
import scipy.linalg as linalg
from matplotlib import pyplot as plt
import os
import Task3_CSP as CSP

def cross_validation_ecog(ecog_neg, ecog_pos, print_status=False):
    """
    Implements the split for the cross-validation for the eCoG data 

    ecog_neg: only negatively labeled instances (139, 64)
    ecog_pos: only positively labeled instances (139, 64)

    returns: (dict): indices of the test dataset for each fold
    """
    all_folds = {}
    for i in range(1,6):
        # get the starting index for the test set (i-1)/5
        try:
            test_start = int(len(ecog_neg) * ((i-1)/5))
            # if it is the first fold -> start at 0
        except ZeroDivisionError:
            test_start = 0
        # end index is (1/5)-1, (2/5)-1, ... of the data 
        test_end = int(len(ecog_neg) * (i/5)) - 1 

        temp = {
            "test_start": test_start, 
            "test_end": test_end, 
        }

        # append to the folds
        all_folds[i] = temp

        if print_status:
            print(f"Test set for fold number {i} starts at {test_start} and ends at {test_end}!")
            print(f"appended to fold {i}")
        
    return all_folds

def apply_projection_matrix(data, W):
    """Iterate over the data and apply the projection matrix

    Args:
        data (np.ndarray): data that should be reduced
        W (np.ndarray): projection matrix for reduction

    Returns:
        np.ndarrays: dim-reduced data
    """
    # apply W to data
    projected_vectors = None
    for i in range(len(data)):
        epoch = data[i]

        # compute dim-reduced vector
        out = np.dot(epoch.transpose(), W)
        featurevec = np.log(np.var(out, axis=0))
        # concat the vectors to a nd.array -> stacking vertically
        try:
            projected_vectors = np.vstack((projected_vectors, featurevec))
        except ValueError: # first iteration -> first vec is start of data-array
            projected_vectors = featurevec
    return projected_vectors    

def concat_split_data(training_data_neg, training_data_pos, testing_data_neg, testing_data_pos, test_is_none=False):
    """Concatenates the previously grouped data into a single np.ndarray again 
       (for test and train respectively)

    Args:
        training_data_neg (np.ndarray): negative training data
        training_data_pos (np.ndarray): positive training data
        testing_data_neg (np.ndarray): negative testing data
        testing_data_pos (np.ndarray): positive testing data
        test_is_none (bool, optional): When used without cross-validation -> Test is None. Defaults to False.

    Returns:
        np.ndarray(s): concatinated test and training data with the respective labels 
    """
    # concatenate all pos/neg in train or test to apply to W    
    fold_training_data = np.concatenate( (training_data_neg, training_data_pos) )

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
    

    fold_testing_data = None
    labels_test = None
    if not test_is_none:
        # concatenate all pos/neg in test to apply to W
        fold_testing_data = np.concatenate( (testing_data_neg, testing_data_pos) )

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

        # set to None as gc
        training_data_neg = None
        training_data_pos = None
        testing_data_neg = None
        testing_data_neg = None

    return fold_training_data, fold_testing_data, labels_train, labels_test


def training_cross_val(all_folds, ecog_neg, ecog_pos, load_from_file=False):
    """Train the classifier in a cross-val scheme

    Args:
        all_folds (dict): contains indices for test set in each fold
        ecog_neg (np.ndarray): positive data instances
        ecog_pos (np.ndarray): negative data instances
        load_from_file (bool, optional): Projection matrix W can be loaded from file for debugging. Defaults to False.

    Returns:
        np.ndarray(s): Best projection matrix, best fda_w (weights) and fda_b (bias) 
    """
    # init the vars for storing best values
    best_weight_set = False
    best_weight = None
    best_W = None
    best_bias = None
    last_auc = None
    
    # use if i want to be quicker while debugging
    if load_from_file:
        cwd = os.getcwd()
        # number in file-name stands for columns that were kept at each end
        path_to_stored_W = os.path.join(cwd, "W4_save.npy")
        with open(path_to_stored_W, "rb") as infile:
            W = np.load(infile) # Todo: move up so only loaded once

    # iterate over folds
    for fold, index_dict in all_folds.items():
        
        print(f"Working on fold {fold}:")

        # SPLITTING INTO TRAIN AND TEST
        # get the correct subsets for training and testing
        test_start = index_dict["test_start"]
        test_end = index_dict["test_end"]
        
        testing_data_neg = ecog_neg[test_start:test_end]
        testing_data_pos =  ecog_pos[test_start:test_end]
        
        # concat the training data 
        # take sections before the test index (empty in first fold) and after (empty in last fold)
        training_data_neg = np.concatenate( (ecog_neg[0:test_start], (ecog_neg[test_end+1:138])) ) 
        training_data_pos = np.concatenate( (ecog_pos[0:test_start], (ecog_pos[test_end+1:138])) )

        # COMPUTE W
        # compute (already reduced) projection matrix W
        if not load_from_file:
            W = CSP.CSP(training_data_neg, training_data_pos)
        
        # CONCAT THE DATA FOR CLASSIFICATION
        # get the concatenated data (not split in pos and neg) + labels
        fold_training_data, fold_testing_data, labels_train, labels_test = concat_split_data(training_data_neg, training_data_pos, testing_data_neg, testing_data_pos)

        # set to None as garbage collection (gc)
        training_data_neg = None
        training_data_pos = None
        testing_data_neg = None
        testing_data_pos = None

        # PROJECT USING W
        # Project the training and test data
        projected_vecs_train = apply_projection_matrix(data=fold_training_data, W=W)
        projected_vecs_test = apply_projection_matrix(data=fold_testing_data, W=W)
        
        # set to None as gc
        fold_training_data = None
        fold_testing_data = None
        
        # CLASSIFY
        # apply the fda classifier to the training data -> train classifies
        fda_w, fda_b = utils.fda_train(data=projected_vecs_train, label=labels_train)

        # GET FP AND TP USING fda_train()
        # compute tps and fps over number of steps 
        true_positives, false_positives = compute_tps_fps(projected_vecs_test, labels_test, fda_w, absolute=True, steps=100)

        # set to None as gc
        projected_vecs_test = None
        projected_vecs_train = None

        # PLOT
        # plot the roc
        plot_roc(true_positives, false_positives)
        auc = compute_auc(true_positives, false_positives)

        # SELECT BEST fda_w
        # use size of auc as metric (larger = better)
        if not best_weight_set:
            best_weight = fda_w
            best_bias = fda_b
            last_auc = auc
            best_W = W
            best_weight_set = True
        else:
            if auc > last_auc:
                best_weight = fda_w
                best_bias = fda_b
                best_W = W
            last_auc = auc
    
    return best_W, best_weight, best_bias


def training_full_data(ecog_neg, ecog_pos):
    """Training method if no cross validation is required

    Args:
        ecog_neg (np.ndarray): negative training instances
        ecog_pos (np.ndarray): positive training instances

    Returns:
        np.ndarray(s): projection matrix W, computed weights and bias
    """
    # compute W using the data
    W = CSP.CSP(ecog_neg, ecog_pos, cols=3)
    # concat the split data (only need return values for training, so test is _)
    training_data, _, labels_train, _ = concat_split_data(training_data_neg=ecog_neg, training_data_pos=ecog_pos, testing_data_neg=None, testing_data_pos=None, test_is_none=True)
    # project the vectors using W
    projected_vecs_train = apply_projection_matrix(data=training_data, W=W)
    # apply fda_classifier -> training phase
    fda_w, fda_b = utils.fda_train(data=projected_vecs_train, label=labels_train)

    return W, fda_w, fda_b


def compute_tps_fps(projected_vecs_test, labels_test, fda_w, absolute=True, steps=100):
    """Computes true positives and false positives by applying the classifier (fda_test)

    Args:
        projected_vecs_test (np.ndarray): projected vectors that should be classified
        labels_test (np.ndarray): labels for the data
        fda_w (np.ndarray): computed fda_w in training phase
        absolute (bool, optional): Return the absolute TP/FPs or the rate. Defaults to True.
        steps (int, optional): Number of bias steps to compute the TPs/FPs for. Defaults to 100.

    Returns:
        np.ndarray: [description]
    """
    true_positives = []
    false_positives = []
    true_positives_rate = []
    false_positives_rate = []

    # compute test classification over 100 bias values
    bias = np.linspace(0,1,steps)

    for b in bias:
        # use fda_test to compute the classification with the bias
        scores, labels = utils.fda_test(projected_vecs_test, fda_w, fda_b=b)
        # compute TP, ...
        TP,FP,FN,TN = utils.calc_confusion(labels, labels_test)
        
        # only want the values (not rates)
        if absolute:
            true_positives.append(TP)
            false_positives.append(FP)

        else:
            # compute the rates
            true_positives_rate.append(TP / (TP + FN))
            try:
                false_positives_rate.append(FP / (TN + FP)) 
            except ZeroDivisionError:
                false_positives_rate.append(0.0)
    
    if absolute:
        # need sorting, otherwise auc will be negative
        true_positives.sort()
        false_positives.sort()
        return true_positives, false_positives 
    else:
        # need sorting, otherwise auc will be negative
        true_positives_rate.sort()
        false_positives_rate.sort()
        return true_positives_rate, false_positives_rate

def plot_roc(true_positives, false_positives):
    """Uses matplotlib to plot the ROC curve

    Args:
        true_positives (list): list of fp-values for each bias shift
        false_positives (list): list of fp-values for each bias shift
    """
    # get the AUC for the title
    auc = compute_auc(true_positives, false_positives)
    # init the subplot
    plt.subplot(211)
    # plot the two lists as x- and y-axis respectively
    plt.plot(false_positives, true_positives, label="ROC")
    # insert legend
    plt.legend(loc=0)
    # insert title
    plt.title(label=f"ROC Curve with AUC of {auc}", loc="center")
    # show the plot
    plt.show(block=True)

def compute_auc(true_positives, false_positives):
    """Compute the area under curve (AUC)

    Args:
        true_positives (list): list of fp-values for each bias shift
        false_positives (list): list of fp-values for each bias shift

    Returns:
        float: computed AUC value
    """
    # stack the lists as two columns for the functino in utils
    roc = np.vstack( (true_positives, false_positives ))
    auc = utils.calc_AUC(roc)
    return auc


if __name__ == "__main__":
    
    # set the param for cross-val or not
    cross_val = False
    
    # 0. load data from file
    training_data_ecog, training_label_ecog = ecog_load_data.ecog_load_data_train()

    # PREPROCESSING
    # filtering using mne + grouping according to label
    ecog_neg, ecog_pos = CSP.data_preprocessing(training_data_ecog, training_label_ecog)
    
    # reset for saving space
    training_data_ecog = None
    training_label_ecog = None

    # TRAINING PHASE
    if cross_val:
        # GET INDICES FOR CROSS-VAL 
        all_folds = cross_validation_ecog(ecog_neg, ecog_pos)
    
        # TRAINING PHASE USING CROSS-VAL
        W, fda_w, fda_b = training_cross_val(all_folds=all_folds, ecog_neg=ecog_neg, ecog_pos=ecog_pos, load_from_file=False)
    else:
        # TRAINING WITH ALL DATA
        W, fda_w, fda_b = training_full_data(ecog_neg, ecog_pos, load_from_file=False)

    print("DONE TRAINING >> STARTING TEST PHASE")
    
    # TESTING PHASE (on competition_test.mat)
    testing_data_ecog, testing_label_ecog = ecog_load_data.ecog_load_data_test()

    # make all -1 labels 0
    testing_label_ecog = np.where(testing_label_ecog == -1, 0, testing_label_ecog)

    # filter the data with MNE bandpass
    filtered_testing_data_ecog = CSP.filter_data_ecog(testing_data_ecog)

    # apply W -> recieved from either cross-val (best W) or application on all training data
    projected_vecs_test = apply_projection_matrix(filtered_testing_data_ecog, W)

    # compute ROC and auc
    FP, TP = compute_tps_fps(projected_vecs_test=projected_vecs_test, labels_test=testing_label_ecog, fda_w=fda_w)
    plot_roc(TP, FP)
    auc = compute_auc(TP, FP)
    print("AUC of testing phase: " , auc)




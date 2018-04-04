#!/usr/bin/python3.5
import os
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.externals import joblib

DATA_DIR = './data'
OUTPUT_DIR = './data'
POSITIVE_FEATURE_PATH = os.path.join(DATA_DIR, 'specialized_corners.mat')
NEGATIVE_FEATURE_PATH = os.path.join(DATA_DIR, 'features_neg.mat')

features_positive = pickle.load(open(POSITIVE_FEATURE_PATH, 'rb'), encoding='latin1')
features_negative = pickle.load(open(NEGATIVE_FEATURE_PATH, 'rb'), encoding='latin1')

# Seperate the training and testing data
def seperate_training_data(arr, training_ratio):
    training_amount = max(min(round(len(arr) * training_ratio), len(arr)), 0)

    return arr[:training_amount], arr[training_amount:]

def create_training_and_test_data(positive_train_ratio = 0.7, negative_train_ratio = 0.8, create_test_data = True):
    fp_train, fp_test = seperate_training_data(features_positive, positive_train_ratio)
    fn_train, fn_test = seperate_training_data(features_negative, negative_train_ratio)

    X_train = np.concatenate([ fp_train, fn_train ])
    Y_train = np.concatenate([ np.ones((len(fp_train), )), np.zeros((len(fn_train), )) ])

    if create_test_data:
        X_test = np.concatenate([ fp_test, fn_test ])
        Y_test = np.concatenate([ np.ones((len(fp_test), )), np.zeros((len(fn_test), )) ])

        return (X_train, Y_train), (X_test, Y_test)

    return (X_train, Y_train), ([], [])

def create_models(training_data, test_data, score=True):
    X_train, Y_train = training_data
    X_test, Y_test = test_data

    lr = LogisticRegression().fit(X_train, Y_train)
    dt = DecisionTreeClassifier().fit(X_train, Y_train)
    rf = RandomForestClassifier().fit(X_train, Y_train)
    svm = SVC().fit(X_train, Y_train)

    if score:
        print("Printing scores below...")
        for model in [lr, dt, rf, svm]:
            print("'%s' score: %f" % (model.__class__.__name__, model.score(X_test, Y_test)))

    return lr, dt, rf, svm

####for i in range(0, 5):
####    for j in range(0, 5):
####        positive_train_ratio = 0.5 + (i * 0.1)
####        negative_train_ratio = 0.5 + (j * 0.1)
####        training_data, test_data = create_training_and_test_data(positive_train_ratio, negative_train_ratio, create_test_data=True)

####        print("Checking scores for train ratios: %f (positive), %f (negative)." % (positive_train_ratio, negative_train_ratio))
####        create_models(training_data, test_data, score=True)

training_data, test_data = create_training_and_test_data(0.9, 0.7, create_test_data=True)
_, _, rf, _ = create_models(training_data, test_data, score=True)

rf_path = os.path.join(OUTPUT_DIR, 'corner-classifier-rf.pkl')
print("Dumping the learned RF classifier to '%s'..." % (rf_path))
joblib.dump(rf, rf_path)
print('Done!')

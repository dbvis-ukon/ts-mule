import sys
import dill

import numpy as np
import pandas as pd

import argparse

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

from sklearn import linear_model, metrics

sys.path.insert(0, './../../')

from tsmule.xai.lime import LimeTS
from tsmule.xai.evaluation import PerturbationAnalysis


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Start evaluation of Beijing 2.5PM models.')
    parser.add_argument('-perturb', type=str, default='', nargs='?', help='do the perturbation analysis')
    parser.add_argument('-end', type=int, default=1000, nargs='?', help='an integer for the end index')
    parser.add_argument('-start', type=int, default=0, nargs='?', help='an integer for the start index')
    parser.add_argument('-model', type=str, default='cnn', nargs='?', help='a model name for the evaluation cnn | rnn | dnn')
    
    args = parser.parse_args()
    end = args.end
    start = args.start
    model = args.model
    
    perturb = args.perturb
    
    print('Args:', perturb, start, end, model)
    
    if perturb == '':
        
        if model == 'cnn':
            model = keras.models.load_model('./beijing_air_2_5_cnn_model.h5')
        elif model == 'dnn':
            model = keras.models.load_model('./beijing_air_2_5_dnn_model.h5')
        elif model == 'rnn':
            model = keras.models.load_model('./beijing_air_2_5_rnn_model.h5')
        else:
            print('No model specified')
            exit

        with open('./beijing_air_2_5_test_data.dill', 'rb') as f:
            dataset_test = dill.load(f)

        X = dataset_test[0][start:end]
        y = dataset_test[1][start:end]

        def predict_fn(x):
            if len(x.shape) == 2:
                prediction = model.predict(x[np.newaxis]).ravel()
            else:
                prediction = model.predict(x).ravel()
            return prediction

        relevances = []
        for x in X:
            explainer = LimeTS()
            relevances.append(explainer.explain(x, predict_fn))
            
        relevances = np.asarray(relevances)
        
        with open('./beijing_air_2_5_test_data_relevances_' + start + '_' + end + '_' + model + '_' + '.dill', 'wb') as f:
            dill.dump(relevances, f)
        
    else:
        
        pass

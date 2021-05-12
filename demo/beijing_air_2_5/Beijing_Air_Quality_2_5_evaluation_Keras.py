import sys
import dill

import numpy as np
import pandas as pd

import argparse

from os import listdir
from os.path import isfile, join

import tensorflow as tf
from tensorflow import keras

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
    model_name = args.model
    
    perturb = args.perturb
    
    print('Args:', perturb, start, end, model_name)
    
    if model_name == 'cnn':
        model = keras.models.load_model('./beijing_air_2_5_cnn_model.h5')
    elif model_name == 'dnn':
        model = keras.models.load_model('./beijing_air_2_5_dnn_model.h5')
    elif model_name == 'rnn':
        model = keras.models.load_model('./beijing_air_2_5_rnn_model.h5')
    else:
        print('No model specified')
        exit


    def predict_fn(x):
        if len(x.shape) == 2:
            prediction = model.predict(x[np.newaxis]).ravel()
        else:
            prediction = model.predict(x).ravel()
        return prediction


    if perturb == '':

        with open('./beijing_air_2_5_test_data.dill', 'rb') as f:
            dataset_test = dill.load(f)

        X = dataset_test[0][start:end]
        y = dataset_test[1][start:end]
        
        relevances = []

        for i, x in enumerate(X):
            explainer = LimeTS()
            lasso_classifier = linear_model.Lasso(alpha=0.001)
            explainer._kernel = lasso_classifier
            
            relevances.append(explainer.explain(x, predict_fn))
            
            print('Done:', i)
            
        relevances = np.asarray(relevances)
        relevances_data = [X, y, relevances]
        
        with open('./beijing_air_2_5_test_data_relevances_' + str(start) + '_' + str(end) + '_' + model_name + '_' + '.dill', 'wb') as f:
            dill.dump(relevances_data, f)
        
    else:
        
        X = []
        y = []
        relevances = []
        
        cwd = './'
        files = [f for f in listdir(cwd) if isfile(join(cwd, f))]

        for f in files:
            if 'relevances' in f and model_name in f:
                with open(join(cwd, f), 'rb') as fd:
                    relevance = dill.load(fd)
                    X.append(relevance[0])
                    y.append(relevance[1])
                    relevances.append(relevance[2])

        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)
        relevances = np.concatenate(relevances, axis=0)
        
        pa = PerturbationAnalysis()
        zero_scores = pa.analysis_relevance(X, y, relevances, predict_fn=predict_fn, replace_method='zeros', eval_fn=metrics.mean_squared_error, percentile=90)
        
        pa = PerturbationAnalysis()
        inverse_max_scores = pa.analysis_relevance(X, y, relevances, predict_fn=predict_fn, replace_method='inverse_max', eval_fn=metrics.mean_squared_error, percentile=90)
        
        pa = PerturbationAnalysis()
        inverse_mean_scores = pa.analysis_relevance(X, y, relevances, predict_fn=predict_fn, replace_method='inverse_mean', eval_fn=metrics.mean_squared_error, percentile=90)
        
        pa = PerturbationAnalysis()
        global_mean_scores = pa.analysis_relevance(X, y, relevances, predict_fn=predict_fn, replace_method='global_mean', eval_fn=metrics.mean_squared_error, percentile=90)
        
        pa = PerturbationAnalysis()
        local_mean_scores = pa.analysis_relevance(X, y, relevances, predict_fn=predict_fn, replace_method='local_mean', eval_fn=metrics.mean_squared_error, percentile=90)

        perturbation_data = {
            'zero_scores': zero_scores,
            'inverse_max_scores': inverse_max_scores,
            'inverse_mean_scores': inverse_mean_scores,
            'global_mean_scores': global_mean_scores,
            'local_mean_scores': local_mean_scores,
        }
        
        with open('./beijing_air_2_5_test_data_relevances_perturbation_' + model_name + '_' + '.dill', 'wb') as f:
            dill.dump(perturbation_data, f)

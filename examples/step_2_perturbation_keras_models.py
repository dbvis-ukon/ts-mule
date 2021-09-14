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

import logging 
logging.getLogger('stumpy').setLevel(logging.ERROR)

sys.path.insert(0, './../')

from tsmule.xai.lime import LimeTS
from tsmule.xai.evaluation import PerturbationAnalysis
from tsmule.sampling.segment import SegmentationPicker


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Start the perturbation analysis of pre-trained models in a directory.')
    parser.add_argument('-path', type=str, default='', nargs='?', help='path to the folder to evaluate')
    parser.add_argument('-model', type=str, default='cnn', nargs='?', help='a model name for the evaluation [cnn | rnn | dnn]')
    
    args = parser.parse_args()
    path = args.path
    model_name = args.model

    files = []

    dnn_path = ''
    cnn_path = ''
    rnn_path = ''
    for f in listdir(path):
        if isfile(join(path, f)):
            files.append(f)

            if 'dnn' in f:
                dnn_path = join(path, f)
            elif 'cnn' in f:
                cnn_path = join(path, f)
            elif 'rnn' in f:
                rnn_path = join(path, f)
            elif 'test_data' in f:
                data_path = join(path, f)

    print('Args:', path, model_name)
    
    if model_name == 'dnn' and not dnn_path == '':
        model = keras.models.load_model(dnn_path)
    elif model_name == 'cnn' and not cnn_path == '':
        model = keras.models.load_model(cnn_path)
    elif model_name == 'rnn' and not rnn_path == '':
        model = keras.models.load_model(rnn_path)
    else:
        print('No model specified or found')
        exit


    def predict_fn(x):
        if len(x.shape) == 2:
            prediction = model.predict(x[np.newaxis]).ravel()
        else:
            prediction = model.predict(x).ravel()
        return prediction


    X = []
    y = []

    relevances = {}

    for f in files:
        if 'relevances' in f and model_name in f:
            with open(join(path, f), 'rb') as fd:
                data = dill.load(fd)
                X.append(data[0])
                y.append(data[1])
                    
                for key in data[2]:
                    if key not in relevances:
                        relevances[key] = []
                    relevances[key].append(data[2][key])

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    for key in relevances:
        relevances[key] = np.concatenate(relevances[key], axis=0)
            
    perturbation_data = {}
        
    for key in relevances:
            
        rel = relevances[key]
            
        zero_scores = None
        inverse_max_scores = None
        inverse_mean_scores = None
        global_mean_scores = None
        local_mean_scores = None
        
        try:
            pa = PerturbationAnalysis()
            zero_scores = pa.analysis_relevance(X, y, rel, predict_fn=predict_fn, replace_method='zeros', eval_fn=metrics.mean_squared_error, percentile=90)
                
            pa = PerturbationAnalysis()
            inverse_max_scores = pa.analysis_relevance(X, y, rel, predict_fn=predict_fn, replace_method='inverse_max', eval_fn=metrics.mean_squared_error, percentile=90)
                
            pa = PerturbationAnalysis()
            inverse_mean_scores = pa.analysis_relevance(X, y, rel, predict_fn=predict_fn, replace_method='inverse_mean', eval_fn=metrics.mean_squared_error, percentile=90)
                
            pa = PerturbationAnalysis()
            global_mean_scores = pa.analysis_relevance(X, y, rel, predict_fn=predict_fn, replace_method='global_mean', eval_fn=metrics.mean_squared_error, percentile=90)
                
            pa = PerturbationAnalysis()
            local_mean_scores = pa.analysis_relevance(X, y, rel, predict_fn=predict_fn, replace_method='local_mean', eval_fn=metrics.mean_squared_error, percentile=90)
            
        except:
            print('Error with:', key)

        perturbation_data_for_key = {
            'zero_scores': zero_scores,
            'inverse_max_scores': inverse_max_scores,
            'inverse_mean_scores': inverse_mean_scores,
            'global_mean_scores': global_mean_scores,
            'local_mean_scores': local_mean_scores,
        }
            
        perturbation_data[key] = perturbation_data_for_key
        
    with open(join(path, 'test_data_perturbation_' + model_name + '_' + '.dill'), 'wb') as f:
        dill.dump(perturbation_data, f)

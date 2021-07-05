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

sys.path.insert(0, './../../')

from tsmule.xai.lime import LimeTS
from tsmule.xai.evaluation import PerturbationAnalysis
from tsmule.sampling.segment import SegmentationPicker


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Start evaluation of Beijing 2.5PM models.')
    parser.add_argument('-perturb', type=str, default='', nargs='?', help='do the perturbation analysis')
    parser.add_argument('-end', type=int, default=1000, nargs='?', help='an integer for the end index')
    parser.add_argument('-start', type=int, default=0, nargs='?', help='an integer for the start index')
    parser.add_argument('-model', type=str, default='cnn', nargs='?', help='a model name for the evaluation [cnn | rnn | dnn]')
    parser.add_argument('-segm', type=str, default='all', nargs='?', help='a segmentation name for the evaluation [all | matrix | sax | window]')
    
    
    segmentation_classes = {
        'matrix': ['slopes-sorted', 'slopes-not-sorted', 'bins-min', 'bins-max'], 
        'sax': [''], 
        'window': ['uniform', 'exponential']}
    
    
    args = parser.parse_args()
    end = args.end
    start = args.start
    model_name = args.model
    segmentation_name = args.segm
    
    perturb = args.perturb
    
    print('Args:', perturb, start, end, model_name, segmentation_name)
    
    if model_name == 'cnn':
        model = keras.models.load_model('./beijing_air_multi_site_cnn_model.h5')
    elif model_name == 'dnn':
        model = keras.models.load_model('./beijing_air_multi_site_dnn_model.h5')
    elif model_name == 'rnn':
        model = keras.models.load_model('./beijing_air_multi_site_rnn_model.h5')
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

        with open('./beijing_air_multi_site_test_data.dill', 'rb') as f:
            dataset_test = dill.load(f)

        X = dataset_test[0][start:end]
        y = dataset_test[1][start:end]
        
        segmentations_to_process = {segmentation_name: segmentation_classes[segmentation_name]} if segmentation_name != 'all' else segmentation_classes
        
        relevances = {}
        for segmentation in segmentations_to_process:
            parameters_to_process = segmentations_to_process[segmentation]
            for parameter in parameters_to_process:
                
                relevances_temp = []

                for i, x in enumerate(X):
                    explainer = LimeTS()
                    
                    # set linear model
                    lasso_classifier = linear_model.Lasso(alpha=0.001)
                    explainer._kernel = lasso_classifier
                    
                    # estimate partitions and take 10% of time series length
                    ts_len = max(x.shape)
                    partitions = int(ts_len * 0.1 + 1)
                    
                    # set segmentation
                    segmentation_class = SegmentationPicker().select(segmentation, partitions)
                    explainer._segmenter = segmentation_class
                    
                    try:
                        explanation = explainer.explain(x, predict_fn, segmentation_method=parameter)
                    except:
                        print('Error')
                        explanation = None

                    relevances_temp.append(explanation)
                    
                    print('Done:', i)

                relevances[segmentation + '-' + parameter] = np.asarray(relevances_temp)
                
                print('-----------------')
                print('Done:', parameter)
                
            print('-----------------')
            print('Done:', segmentation)

        relevances_data = [X, y, relevances]
    
        with open('./beijing_air_multi_site_test_data_relevances_' + str(start) + '_' + str(end) + '_' + model_name + '_' + '.dill', 'wb') as f:
            dill.dump(relevances_data, f)
        
    else:
        
        X = []
        y = []
        relevances = []
        
        cwd = './'
        files = [f for f in listdir(cwd) if isfile(join(cwd, f))]
        
        relevances = {}

        for f in files:
            if 'relevances' in f and model_name in f:
                with open(join(cwd, f), 'rb') as fd:
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
        
        with open('./beijing_air_multi_site_test_data_perturbation_' + model_name + '_' + '.dill', 'wb') as f:
            dill.dump(perturbation_data, f)

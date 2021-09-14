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

    parser = argparse.ArgumentParser(description='Start evaluation of pre-trained models in a directory.')
    parser.add_argument('-path', type=str, default='', nargs='?', help='path to the folder to evaluate')
    parser.add_argument('-start', type=int, default=0, nargs='?', help='an integer for the start index')
    parser.add_argument('-end', type=int, default=1000, nargs='?', help='an integer for the end index')
    parser.add_argument('-model', type=str, default='cnn', nargs='?', help='a model name for the evaluation [cnn | rnn | dnn]')
    parser.add_argument('-segm', type=str, default='all', nargs='?', help='a segmentation name for the evaluation [all | matrix-slope | matrix-bins | sax | window]')
    
    
    segmentation_classes = {
        'matrix-slope': ['slopes-sorted', 'slopes-not-sorted'],
        'matrix-bins': ['bins-min', 'bins-max'], 
        'sax': [''], 
        'window': ['uniform', 'exponential']}


    args = parser.parse_args()
    path = args.path
    end = args.end
    start = args.start
    model_name = args.model
    segmentation_name = args.segm

    data_path = ''

    dnn_path = ''
    cnn_path = ''
    rnn_path = ''
    for f in listdir(path):
        if isfile(join(path, f)):
            if 'dnn' in f:
                dnn_path = join(path, f)
            elif 'cnn' in f:
                cnn_path = join(path, f)
            elif 'rnn' in f:
                rnn_path = join(path, f)
            elif 'test_data' in f:
                data_path = join(path, f)

    print('Args:', path, start, end, model_name, segmentation_name)

    if data_path == '':
        print('No data found')
        exit
    
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


    with open(data_path, 'rb') as f:
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
                explainer = LimeTS(n_samples=1000)
                    
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
                except Exception as e:
                    print(e)
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
    
    with open(join(path, 'test_data_relevances_' + str(start) + '_' + str(end) + '_' + model_name + '_' + '.dill'), 'wb') as f:
        dill.dump(relevances_data, f)

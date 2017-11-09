# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 22:24:08 2017

@author: 罗骏
"""

import os
import numpy as np
import _read_data_method
import _adjusted_wavelets_combination
import _dwt_construct_array
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

opposite_path = os.path.abspath('')
key_length = 64 #Adjusted minimum length of amino acid sequence

def main():
    global key_length
    mod = 1  #
    
    species = ['Arabidopsis_thaliana'] #
    for name_ in species:
        # read data
        locate_feature = _read_data_method.transfer_feature()
        locate_fasta = _read_data_method.transfer_fasta(name_)
        locate_dip_matrix_POS = _read_data_method.dip_turn_local(name_, 'interact')  
        locate_dip_matrix_NEG = _read_data_method.dip_turn_local(name_, 'uninteract')
        output_data = []
        # wavelets_comb is a list of wavelets' names that will be used
        wavelets_comb, file_name = _adjusted_wavelets_combination.adjusted_wavelets_combination(mod)
        for DWT_name in wavelets_comb:   
            output_data += [DWT_name]
            dwt_Positive_feature_rows = _dwt_construct_array.dwt_construct_array(locate_dip_matrix_POS, locate_feature, locate_fasta, '0', key_length, DWT_name, 'Ture')
            dwt_Negative_feature_rows = _dwt_construct_array.dwt_construct_array(locate_dip_matrix_NEG, locate_feature, locate_fasta, '1', key_length, DWT_name, 'Ture')
            dwt_feature = dwt_Negative_feature_rows + dwt_Positive_feature_rows
            S = len(dwt_feature)
            print('\n', DWT_name,'\nHave constructed dwt feature')
            
            row_number = -1
            for rows in dwt_feature:
                row_number += 1 
                if (row_number == 0):
                    data = np.zeros((S,len(rows[0:])))
                    data[row_number,:] = np.array(rows[0:]).astype(np.float32)
                    continue
                data[row_number,:] = np.array([rows[0:]]).astype(np.float32)
            featureList = data[:,0:-1]
            labelList = data[:,-1]
            print('feature, label constructed', '\ndata.size:', data.size, len(labelList), '\n')
            
            X_train, X_test, y_train, y_test = train_test_split(featureList, labelList, test_size=0.20)
            
            param_grid_sets = [
                    {'weights' : ['uniform','distance'],
                     'algorithm' : ['ball_tree', 'kd_tree'],
                     'leaf_size' : [x for x in range(10, 110, 10)]},
                    {'n_estimators': [x for x in range(20, 220, 20)],
                     'min_samples_split': [2, 3, 4],
                     'max_depth': [x for x in range(6, 12, 2)]},
                    {'loss' : ['deviance', 'exponential'],
                     'learning_rate' : [x/10 for x in range(1, 9, 2)],
                     'n_estimators' : [x for x in range(50, 260, 50)],
                     'max_depth' : [x for x in range(3, 8, 1)]},
                     {'penalty' : ['l1', 'l2'],
                      'loss' : ['hinge', 'squared_hinge'],
                      'C' : [x for x in range(1, 9, 2)]}
                    ]
            classifiers = [KNeighborsClassifier(), ExtraTreesClassifier(), GradientBoostingClassifier(), LinearSVC()]
            
            for param_grid, clf in zip(param_grid_sets, classifiers):
                grid_search = GridSearchCV(estimator = clf, param_grid = param_grid, cv = 10, n_jobs = -1)
                grid_search.fit(X_train, y_train)
                print ('Best score: %0.3f' % grid_search.best_score_)
                print ('Best parameters set:')
                best_parameters = grid_search.best_estimator_.get_params()
                for param_name in sorted(param_grid.keys()):
                    print ('\t%s: %r' % (param_name, best_parameters[param_name]))
                predictions = grid_search.predict(X_test)
                print (classification_report(y_test, predictions))
    return 0

if __name__ == '__main__':
    main()
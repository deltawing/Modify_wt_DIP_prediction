# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 05:59:43 2017

@author: 罗骏
"""
import csv
import os
import numpy as np
#import time
import _read_data_method
import _adjusted_wavelets_combination
import _dwt_construct_array
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
S = 0
opposite_path = os.path.abspath('')

key_length = 64 #Adjusted minimum length of amino acid sequence

'''
This script is focusing on revealing the influences of different number of wavelets used in constitute a feature.
Output file in ./data/test_result
'''

def main():
    global key_length
    #timea = time.time()
    print('How many wavelets to generate the feature? \n (Currently we support 1~5 wavelets) \n Please input a number.')
    mod = input()
    if mod not in ['1','2','3','4','5']:
        print('Please input a number between 0 and 6')
        return 0
    else:
        mod = int(mod)
        
    #Species' names of the dataset that will be tested
    species = ['Arabidopsis_thaliana','Bos_taurus','Caenorhabditis_elegans','Drosophila_melanogaster',
               'Homo_sapiens','Mus_musculus','Rattus_norvegicus','Saccharomyces_cerevisiae']
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
            
            clf2 = ExtraTreesClassifier(n_estimators=200)
            clf4 = QuadraticDiscriminantAnalysis()
            clf = [clf2,clf4]
            clf_name = ['ExtraTrees','QuadraticDiscriminant']
            num = 0
            for i in clf:
                predicted = cross_val_predict(i,featureList,labelList,cv=10,n_jobs=-1 )
                score1 = metrics.confusion_matrix(labelList,predicted).tolist()
                score1_mod = score1[0]+score1[1]
                score2 = metrics.f1_score(labelList,predicted)
                score4 = metrics.matthews_corrcoef(labelList,predicted)
                print (clf_name[num],"Classifier:\nconfusion_matrix",score1,"\nf1_score",np.average(score2),"\nmatthews_corrcoef",np.average(score4))
                output_data += [[clf_name[num]],['confusion_matrix']+score1_mod,['f1_score']+[np.average(score2)],['matthews_corrcoef']+[np.average(score4)]]
                num += 1
                #print(time.time()-timea)
                
        if not os.path.exists(os.path.join(opposite_path, 'data','test_result')):
            os.makedirs(os.path.join(opposite_path, 'data','test_result'))                
        with open (os.path.join(opposite_path, 'data','test_result','adv_WT_'+file_name+name_+'.csv'),'w',newline='')as T: 
            featured_array5 = csv.writer(T)
            featured_array5.writerows(output_data)
        
if __name__ == '__main__':
    main()
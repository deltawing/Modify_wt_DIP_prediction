# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 05:59:43 2017

@author: 罗骏
"""
import csv
import os
import numpy as np
import _read_data_method
import _dwt_construct_array
import _compare_algorithm
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


'''
This script is used to compare the predictive performance of other algorithms.
Output file in ./data/test_result, filename prefix is "same_spec_comparison_"
'''

S = 0
opposite_path = os.path.abspath('')
featureList = []
labelList = []
key_length = 64  

def construct_array(temp_interact_hash, locate_feature, locate_fasta, label):
    PseAAC_array = []
    AC_array = []
    LD_array = []
    CT_array = []
    print("constructing array with label %s..." % label)
    for temp_interact_row in temp_interact_hash:
        interact_proteinA = temp_interact_row[0:temp_interact_row.find('M')]
        interact_proteinB = temp_interact_row[temp_interact_row.find('M')+1:]
        dip_proteinA = _dwt_construct_array.search_match(interact_proteinA, locate_fasta)
        dip_proteinB = _dwt_construct_array.search_match(interact_proteinB, locate_fasta)
        if (len(dip_proteinA)<key_length or len(dip_proteinB)<key_length):
            continue      
        PseAAC_featured_proteinA = _compare_algorithm.pseaac(dip_proteinA, locate_feature)
        PseAAC_featured_proteinB = _compare_algorithm.pseaac(dip_proteinB, locate_feature)
        AC_featured_proteinA = _compare_algorithm.auto_Covariance(dip_proteinA, locate_feature)
        AC_featured_proteinB = _compare_algorithm.auto_Covariance(dip_proteinB, locate_feature)
        LD_featured_proteinA = _compare_algorithm.local_descriptors(dip_proteinA)
        LD_featured_proteinB = _compare_algorithm.local_descriptors(dip_proteinB)
        CT_featured_proteinA = _compare_algorithm.conjoint_triad(dip_proteinA)
        CT_featured_proteinB = _compare_algorithm.conjoint_triad(dip_proteinA)        
        PseAAC_array += [PseAAC_featured_proteinA + PseAAC_featured_proteinB + [label]]
        AC_array += [AC_featured_proteinA + AC_featured_proteinB + [label]]
        LD_array += [LD_featured_proteinA + LD_featured_proteinB + [label]]
        CT_array += [CT_featured_proteinA + CT_featured_proteinB + [label]]
    return AC_array, PseAAC_array, LD_array, CT_array    

    
def main():
    global key_length
    '''species = ['Arabidopsis_thaliana','Bos_taurus','Caenorhabditis_elegans','Drosophila_melanogaster',
               'Homo_sapiens','Mus_musculus','Rattus_norvegicus','Saccharomyces_cerevisiae']'''
    species = ['Rattus_norvegicus']
    for name_ in species:
        # read data
        locate_feature = _read_data_method.transfer_feature()
        locate_fasta = _read_data_method.transfer_fasta(name_)
        locate_dip_matrix_POS = _read_data_method.dip_turn_local(name_, 'interact')  
        locate_dip_matrix_NEG = _read_data_method.dip_turn_local(name_, 'uninteract')
        output_data = []
        AC_Positive_rows, PseAAC_Positive_rows, LD_Positive_rows, CT_Positive_rows = construct_array(locate_dip_matrix_POS,locate_feature,locate_fasta,'0')
        AC_Negative_rows, PseAAC_Negative_rows, LD_Negative_rows, CT_Negative_rows = construct_array(locate_dip_matrix_NEG,locate_feature,locate_fasta,'1')
        print('array constructed')
        AC_feature = np.array(AC_Positive_rows+AC_Negative_rows).astype(np.float32)
        PseAAC_feature = np.array(PseAAC_Positive_rows+PseAAC_Negative_rows).astype(np.float32)
        LD_feature = np.array(LD_Positive_rows+LD_Negative_rows).astype(np.float32)
        CT_feature = np.array(CT_Positive_rows+CT_Negative_rows).astype(np.float32)
        
        #Feature = [AC_feature, PseAAC_feature, LD_feature, CT_feature]
        Feature = [AC_feature, PseAAC_feature, LD_feature]
        clf1 = SVC(kernel='linear')
        clf2 = SVC(kernel='linear')
        clf3 = KNeighborsClassifier(weights='distance')
        classifiers = [clf1, clf2, clf3]
        Feature_name = ['AC', 'PseAAC', 'LD']
        for diff_feature, classifier, diff_feature_name in zip(Feature, classifiers, Feature_name):
            featureList = diff_feature[:,0:-1]
            labelList = diff_feature[:,-1]
            predicted = cross_val_predict(classifier,featureList,labelList,cv=10,n_jobs=-1)
            score1 = metrics.confusion_matrix(labelList,predicted).tolist()
            score1_mod = score1[0]+score1[1]
            score2 = metrics.f1_score(labelList,predicted)
            score4 = metrics.matthews_corrcoef(labelList,predicted)
            print (diff_feature_name,"Classifier:\nconfusion_matrix",score1,"\nf1_score",np.average(score2),"\nmatthews_corrcoef",np.average(score4))
            output_data += [[diff_feature_name],['confusion_matrix']+score1_mod,['f1_score']+[np.average(score2)],['matthews_corrcoef']+[np.average(score4)]]
            
        for DWT_name in ['db1', 'db2']:
            print(DWT_name)
            output_data += [['iPP-Esml('+DWT_name+')']]
            DWT_Positive_feature_rows = _dwt_construct_array.dwt_construct_array(locate_dip_matrix_POS, locate_feature, locate_fasta, '0', key_length, DWT_name)
            DWT_Negative_feature_rows = _dwt_construct_array.dwt_construct_array(locate_dip_matrix_NEG, locate_feature, locate_fasta, '1', key_length, DWT_name)
            DWT_feature = DWT_Negative_feature_rows+DWT_Positive_feature_rows
            S = len(DWT_feature)
            row_number = -1
            for rows in DWT_feature:
                row_number += 1 
                if (row_number == 0):
                    data = np.zeros((S,len(rows[0:-1]+[rows[-1]])))
                    data[row_number,:] = np.array(rows[0:-1]+[rows[-1]]).astype(np.float32)
                    continue
                data[row_number,:] = np.array([rows[0:-1]+[rows[-1]]])
            featureList = data[:,0:-1]
            labelList = data[:,-1] 
            #clf5 = RandomForestClassifier(criterion='entropy', n_estimators=100, random_state=1)
            clf7 = SVC(kernel='linear')
            predicted = cross_val_predict(clf7,featureList,labelList,cv=10,n_jobs=-1 )
            score1 = metrics.confusion_matrix(labelList,predicted).tolist()
            score1_mod = score1[0]+score1[1]
            score2 = metrics.f1_score(labelList,predicted)
            score4 = metrics.matthews_corrcoef(labelList,predicted)
            print ('RandomForest',"Classifier:\nconfusion_matrix",score1,"\nf1_score",np.average(score2),"\nmatthews_corrcoef",np.average(score4))
            output_data += [['confusion_matrix']+score1_mod,['f1_score']+[np.average(score2)],['matthews_corrcoef']+[np.average(score4)]]

        if not os.path.exists(os.path.join(opposite_path, 'data','test_result')):
            os.makedirs(os.path.join(opposite_path, 'data','test_result'))         
        with open (os.path.join(opposite_path, 'data', 'test_result', 'same_spec_comparison_'+name_+'.csv'),'w',newline='') as T: 
            featured_array5 = csv.writer(T)
            featured_array5.writerows(output_data)
                   
if __name__ == '__main__':
    main()
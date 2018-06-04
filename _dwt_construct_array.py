# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 13:08:12 2017

@author: 罗骏
"""
import pywt
import numpy as np

'''
First, using fasta file to transform protein pairs(e.g.'297M653') into corresponding amino acid sequences,
then transform these amino acid sequences using wavelet transform. The last number of each row in return file is 
label: '0' means original protein pairs having interaction, '1' means original protein pairs having no interaction
return list example:
    [[3.39932, 0.346153846154, 3.39676, 1.15384615385, 2.8426, 0.576923076923, 3.09421, ..., 1],
    [3.5353, 2.75510204082, -3.34545, 3.0, -2.486, 1.59183673469, 1.51651, ..., 1],
    ...
    [-1.26737, 2.0, -1.24171, 2.5, -1.02878, 0.5, 0.730649, ..., 0]
    ]
'''    
def dwt_construct_array(temp_interact_hash, locate_feature, locate_fasta, label, key_length, dwt_name = 'empty', modify = 'false'):    
    dwt_array = []    
    for temp_interact_row in temp_interact_hash:
        interact_proteinA = temp_interact_row[0:temp_interact_row.find('M')]
        interact_proteinB = temp_interact_row[temp_interact_row.find('M')+1:]
        dip_proteinA = search_match(interact_proteinA, locate_fasta)
        dip_proteinB = search_match(interact_proteinB, locate_fasta)
        if (len(dip_proteinA)<key_length or len(dip_proteinB)<key_length):
            continue 
        if modify == 'false':
            dwt_featured_proteinA = iPP_Esml(dip_proteinA, locate_feature, dwt_name, key_length)
            dwt_featured_proteinB = iPP_Esml(dip_proteinB, locate_feature, dwt_name, key_length)
        else:
            dwt_featured_proteinA = discrete_wavelet_transform(dip_proteinA, locate_feature, dwt_name, key_length)
            dwt_featured_proteinB = discrete_wavelet_transform(dip_proteinB, locate_feature, dwt_name, key_length)
        dwt_array += [dwt_featured_proteinA + dwt_featured_proteinB + [label]]
    return dwt_array


'''
Identify the maximum level that can be decomposed and gather the coefficient series output from high/low-pass filter
return list example:
    [2.22688, 2.41414141414, -2.09604, 1.52525252525, 2.04729, 0.555555555556, 0.649754, 0.704495, ..., 2.7836]
'''
def discrete_wavelet_transform(protein_array, locate_feature, dwt_name, key_length):
    return_DWT_array = []
    for dwt_name_use in dwt_name:
        for q in range(7):
            test_DWT_array = []
            for i in range(len(protein_array)):
                if (protein_array[i]=='X' or protein_array[i]=='U' or protein_array[i]==' '):
                    continue
                test_DWT_array += [locate_feature[protein_array[i]][q]]
            max_level = pywt.dwt_max_level(key_length, pywt.Wavelet(dwt_name_use))
            if max_level >= 4:   
                cA4, cD4, cD3, cD2, cD1 = pywt.wavedec(test_DWT_array, dwt_name_use, level=4)
                cA4array = take_obvious_item(cA4)
                cD4array = take_obvious_item(cD4)
                cD3array = take_obvious_item(cD3)
                cD2array = take_obvious_item(cD2)
                cD1array = take_obvious_item(cD1)
                #cD1array = np.array(cD1).astype(np.float32)
                #cD1array = [cD1array.mean(), cD1array.var()]
                return_DWT_array += cA4array + cD4array + cD3array + cD2array + cD1array
            elif max_level == 3:   
                cA3, cD3, cD2, cD1 = pywt.wavedec(test_DWT_array, dwt_name_use, level=3)
                cA3array = take_obvious_item(cA3)    
                cD3array = take_obvious_item(cD3)
                cD2array = take_obvious_item(cD2)
                cD1array = take_obvious_item(cD1)
                #cD1array = np.array(cD1).astype(np.float32)
                #cD1array = [cD1array.mean(),cD1array.var()]
                return_DWT_array += cA3array + cD3array + cD2array + cD1array
    return return_DWT_array


'''
Pick up (1) 3 samples which have the biggest absolute value of the subsequence and their relative locations
    (2) mean of the wavelet coefficients in the subsequence 
    (3) standard deviation of the wavelet coefficients in the subsequence, 
return list example:
    [3.7269, 2.86567164179, 3.2064, 0.985074626866, 2.98516, 0.783582089552, 1.15042, 4.35581]
'''
def take_obvious_item(X):
    Y = []
    temp_X = np.array(X).astype(np.float32)
    Q = np.array(X).astype(np.float32)
    Q = Q**2
    count = 3     #3 samples which have the biggest absolute value
    #max_number = np.max(Q)**0.5
    me_mean = Q.mean()
    me_var = Q.var()
    for i in range(count):
        Y += [temp_X[np.argmax(Q)]]
        #Y += [(np.argmax(Q)+1)*max_number/Q.size]   
        Y += [(np.argmax(Q)+1)*3/Q.size]   #relative position
        temp_X[np.argmax(Q)] = 0
        Q[np.argmax(Q)] = 0  
    Y += [me_mean, me_var]
    return Y
  
'''
Implement iPP_Esml algorithm 
'''
def iPP_Esml(protein_array, locate_feature, DWT_name, key_length):
    return_DWT_array=[]
    for q in range(7):
        test_DWT_array=[]
        for i in range(len(protein_array)):
            if (protein_array[i]=='X' or protein_array[i]=='U' or protein_array[i]==' ' ):
                continue
            test_DWT_array+=[locate_feature[protein_array[i]][q]]
        max_level = pywt.dwt_max_level(key_length, pywt.Wavelet(DWT_name))
        if max_level >= 4:
            cA4, cD4, cD3, cD2, cD1 = pywt.wavedec(test_DWT_array, DWT_name, level=4)
            for alter in [cA4, cD4, cD3, cD2, cD1]:
                return_DWT_array+=[max(alter)]+[min(alter)]+[np.array(alter).mean()]+[np.array(alter).var()]
    return return_DWT_array

    
def search_match(protein_number,locate_fasta):
    target_protein=''
    if '>dip:DIP-'+protein_number+'N|' in locate_fasta:
        target_protein += locate_fasta['>dip:DIP-'+protein_number+'N|']
    return target_protein
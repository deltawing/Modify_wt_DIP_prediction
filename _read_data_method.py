# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 22:54:10 2017

@author: 罗骏
"""

import csv
import os
opposite_path = os.path.abspath('')
DWT_name = ['bior1.1', 'bior1.3', 'bior2.2', 'bior3.1', 'bior3.3','coif1', 'db1', 'db2', 'db3', 'db4',
                'haar', 'rbio1.1', 'rbio1.3','rbio2.2', 'rbio3.1', 'rbio3.3', 'sym2', 'sym3', 'sym4']
clf_name = ['GradientBoosting','ExtraTrees','KNeighbors','QuadraticDiscriminant','RandomForestClassifier','Stacking']
Compare_method_name = ['AC','PseAAC','LD','iPP-Esml(db1)','iPP-Esml(db2)']
metrics_name = ['confusion_matrix','f1_score','matthews_corrcoef']

'''
read physicochemical property data(have been standardized) from normalized_feature.csv and add them to hash table
    physicochemical property including hydrophobicity, hydrophilicity, the volume of side chains,
    polarity, polarizability, solvent accessible surface area, net charge index of side chains
return dicts example:
    {   'A':[0.636250588, -0.151878007, -1.591936431, -0.085799934, -1.364091085, -1.417615945, -0.452654194],
        ...
    }
'''
def transfer_feature():
    with open(os.path.join(opposite_path,'data','orig','normalized_feature.csv')) as C:
        normalized_feature=csv.reader(C)
        feature_hash = {}
        amino_acid = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
        for charater in amino_acid:
            feature_hash[charater] = []
        for row in normalized_feature:
            i = 0
            for charater in amino_acid:
                feature_hash[charater] += [float(row[i])]
                i += 1
    return feature_hash


'''
read each Species' fasta data from corresponding SPECIES'NAME_fasta_out file and construct 
corresponding relationship between dip number and its amino acid sequence
return dicts example:
    {'>dip:DIP-48601N|': 'MDARDKQVLRSLRLELGAEVLVEGLVLQYLYQEGILTENHIQEIKAQTTGLRKTMLLLDILPSRGP
    KAFDTFLDSLQEFPWVREKLEKAREEATAELPTAQPRKDHLPEKPFIIRLITAAPREAAPESLSPVAETAQSGQKGRLDQSNSCPSAP
    VAATEACLDQS',
    ...
    }
'''
def transfer_fasta(name_):
    with open(os.path.join(opposite_path,'data','orig',name_+'_fasta_out')) as fasta:
        fasta_hash = {}
        dip_number_row = []
        protein_array_row = ['']
        for row in fasta:
            row = row.strip("\n ")
            if (len(row)>0 and row.find('dip:DIP-')>=0):
                if (len(dip_number_row)>0 and len(protein_array_row[0])>0):
                    fasta_hash[dip_number_row] = protein_array_row[0]
                    protein_array_row = ['']
                dip_number_row = row
            elif (len(row)>0 and row.find('dip:DIP-')<0):
                protein_array_row[0] += row
    return fasta_hash


'''
read protein interact data from corresponding interact_same_spec_SPECIES'NAME.csv file
erases 'DIP-' and 'N|', return file only have the number of each pair of protein
one pair of protein separate with word 'M'
return dicts example:
    {'297M653': 0, '1071M60851': 0, '44817M653': 0, ...
    }
'''
def dip_turn_local(name_, act):
    with open(os.path.join(opposite_path, 'data', 'inter_no_act_location',act+'_same_spec_'+name_+'.csv')) as A:
        dip_matrix=csv.reader(A)
        temp_interact_hash = {}
        for row in dip_matrix:
            if (len(row) <= 0):
                continue
            dip_proteinA, dip_proteinB = row[0], row[1]
            short_itemA = dip_proteinA[dip_proteinA.find('DIP-')+4:dip_proteinA.find('N|')]
            short_itemB = dip_proteinB[dip_proteinB.find('DIP-')+4:dip_proteinB.find('N|')]
            if (short_itemA > short_itemB):
                short_itemA, short_itemB = short_itemB, short_itemA
            temp_interact_hash[str(short_itemA)+'M'+str(short_itemB)] = 0 #use dicts to remove duplicates
    return temp_interact_hash


'''read data from test result file and turn it to dicts
   return dicts example: {('bior1.1','bior1.3'):
        {'KNeighbors':
            {'f1_score': [0.850947687642],
            'confusion_matrix': [2020.0, 831.0, 152.0, 2806.0],
            'matthews_corrcoef': [0.67897255645]
            }, 
        'GradientBoosting': 
            {'f1_score': [0.928848641656],
            'confusion_matrix': [2497.0, 354.0, 86.0, 2872.0],
            'matthews_corrcoef': [0.851850349531]},
        'QuadraticDiscriminant': 
            ...
        'ExtraTrees':
            ...
        'Stacking':
            ...
        'RandomForestClassifier':
            ...
        }
    } 
'''
def read_test_result(datapath):
    local_performance = []
    arrangement = {}
    global DWT_name
    global clf_name
    global metrics_name
    temp_DWT_name = []
    temp_clf_name = []
    temp_metrics_name = []
    
    with open (datapath) as A:
        my_wavelets_performance = csv.reader(A)
        for row in my_wavelets_performance:
            temp_row = []
            for i in range(len(row)):
                if len(row[i])>0:
                    temp_row += [row[i]]
            row = temp_row
            if len(row[0].strip('\n'))>0:
                local_performance += row
                
    u, t, s = 0, 0, 0
    for row_num in range(len(local_performance)):
        if local_performance[row_num] in DWT_name and u == 0:
            if t == 0:
                del temp_DWT_name
                temp_DWT_name = []
            temp_DWT_name += [local_performance[row_num]]
            t += 1
            s = 0
        elif local_performance[row_num] in clf_name:
            if s == 0:
                temp_DWT_name = tuple(temp_DWT_name)
                arrangement[temp_DWT_name] = {}
            temp_clf_name = local_performance[row_num]
            arrangement[temp_DWT_name][temp_clf_name] = {}
            u, s = 1, 1
        elif local_performance[row_num] in metrics_name:
            temp_metrics_name = local_performance[row_num]
            arrangement[temp_DWT_name][temp_clf_name][temp_metrics_name] = []   
            u = 1
        else :
            number = float(local_performance[row_num])
            arrangement[temp_DWT_name][temp_clf_name][temp_metrics_name] += [number]
            u, t = 0, 0
    return arrangement

'''
read data from compare methods result files and turn it to dicts
return dicts example:{AC:
    {confusion_matrix:[26.0, 186.0, 21.0, 554.0],
    f1_score:[0.842585551]
    matthews_corrcoef:[0.1612278]}
    LD: ...
    PseAAC: ...
    iPP-Esml(db1): ...
    iPP-Esml(db2): ...
}
'''
def read_compare_result(datapath):
    compare_local_performance = []
    compare_arrangement = {} 
    global Compare_method_name
    global metrics_name
    temp_method_name = []
    temp_metrics_name = []
    
    with open (datapath) as A:
        compare_performance = csv.reader(A)
        for row in compare_performance:
            temp_row = []
            for i in range(len(row)):
                if len(row[i])>0:
                    temp_row += [row[i]]
            row = temp_row
            if len(row[0].strip(' '))>0:
                compare_local_performance += row
                
    for row_num in range(len(compare_local_performance)):
        if compare_local_performance[row_num] in Compare_method_name:
            temp_method_name = compare_local_performance[row_num]
            compare_arrangement[temp_method_name] = {}
        elif compare_local_performance[row_num] in metrics_name:
            temp_metrics_name = compare_local_performance[row_num]
            compare_arrangement[temp_method_name][temp_metrics_name] = []
        else:
            number = float(compare_local_performance[row_num])
            compare_arrangement[temp_method_name][temp_metrics_name] += [number]
    return compare_arrangement
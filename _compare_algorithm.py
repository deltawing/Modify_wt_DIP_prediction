# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 16:16:00 2017

@author: 罗骏
"""
import math

def aac_7_number_description(protein_array):
    local_operate_array=[]
    for i in range(len(protein_array)):
        if (protein_array[i] in 'AGV'):
            local_operate_array+=[1]
        elif (protein_array[i] in 'ILFP'):
            local_operate_array+=[2]
        elif (protein_array[i] in 'YMTS'):
            local_operate_array+=[3]
        elif (protein_array[i] in 'HNQW'):
            local_operate_array+=[4]
        elif (protein_array[i] in 'RK'):
            local_operate_array+=[5]
        elif (protein_array[i] in 'DE'):
            local_operate_array+=[6]
        elif (protein_array[i]=='C'):
            local_operate_array+=[7]
        else :
            local_operate_array+=[7]
    return local_operate_array

'''
Implement pse-aac algorithm
'''    
def pseaac(protein_array, locate_feature):
    nambda=15   #
    omega=0.05  #
    AA_frequency={'A':[0],'C':[0],'D':[0],'E':[0],'F':[0],'G':[0],'H':[0],'I':[0],'K':[0],'L':[0],
                  'M':[0],'N':[0],'P':[0],'Q':[0],'R':[0],'S':[0],'T':[0],'V':[0],'W':[0],'Y':[0]}
    A_class_feature = [0 for v in range(20)]
    B_class_feature = []
    sum_frequency = 0
    sum_occurrence_frequency = 0
    for i in range(len(protein_array)):
        if (protein_array[i]=='X' or protein_array[i]=='U'):
            continue
        AA_frequency[protein_array[i]][0] += 1
    for j in AA_frequency:
        sum_frequency += AA_frequency[j][0]
    for m in AA_frequency:
        if (sum_frequency == 0):
            s = [0 for b in range(35)]
            return s
        else:    
            AA_frequency[m][0] /= sum_frequency
    for o in AA_frequency:
        sum_occurrence_frequency += AA_frequency[o][0]
    
    for k in range(1,nambda+1):
        B_class_feature += [thet(protein_array, locate_feature, k)]
    Pu_under = sum_occurrence_frequency + omega * sum(B_class_feature)
    for l in range(nambda):
        B_class_feature[l] = (B_class_feature[l] * omega / Pu_under) * 100
    number_range = range(len(AA_frequency))
    for charater, number in zip(AA_frequency, number_range):
        A_class_feature[number] = AA_frequency[charater][0] / Pu_under * 100
    class_feature = A_class_feature + B_class_feature
    return class_feature

def thet(protein_array, locate_feature, t):
    sum_comp = 0
    for i in range(len(protein_array)-t):
        sum_comp += comp(protein_array[i], protein_array[i+t], locate_feature)
    sum_comp /= (len(protein_array) - t)
    return sum_comp
    
def comp(Ri, Rj, locate_feature):
    theth=0
    if (Ri=='X' or Rj=='X' or Ri=='U' or Rj=='U'):
        return 0
    else:
        for i in range(3):
            theth += pow(locate_feature[Ri][i]-locate_feature[Rj][i],2)
        theth=theth/3
        return theth
    
    
'''
Implement local descriptors algorithm
'''
def local_descriptors(protein_array):
    local_operate_array = aac_7_number_description(protein_array)
    A_point = math.floor(len(protein_array)/4)-1
    B_point = A_point*2+1
    C_point = A_point*3+2
    part_vector = []
    part_vector += construct_63_vector(local_operate_array[0:A_point])
    part_vector += construct_63_vector(local_operate_array[A_point:B_point])
    part_vector += construct_63_vector(local_operate_array[B_point:C_point])
    part_vector += construct_63_vector(local_operate_array[C_point:])
    part_vector += construct_63_vector(local_operate_array[0:B_point])
    part_vector += construct_63_vector(local_operate_array[B_point:])
    part_vector += construct_63_vector(local_operate_array[A_point:C_point])
    part_vector += construct_63_vector(local_operate_array[0:C_point])
    part_vector += construct_63_vector(local_operate_array[A_point:])
    part_vector += construct_63_vector(local_operate_array[math.floor(A_point/2):math.floor(C_point-A_point/2)])
    return part_vector
    
def construct_63_vector(part_array):
    simple_7 = [0 for n in range(7)]
    marix_7_7 = [[0 for n in range(7)] for m in range(7)]
    simple_21 = [0 for n in range(21)]
    simple_35 = [0 for n in range(35)]
    for i in range(len(part_array)):
        simple_7[part_array[i]-1] += 1
        if (i<(len(part_array)-1) and part_array[i]!=part_array[i+1]):
            if(part_array[i]>part_array[i+1]):
                j,k = part_array[i+1],part_array[i]
            else:
                j,k = part_array[i],part_array[i+1]
            marix_7_7[j-1][k-1] += 1
    i = 0
    for j in range(7):
        for k in range(j+1,7):
            simple_21[i] = marix_7_7[j][k]
            i += 1
    residue_count = [0,0,0,0,0,0,0]
    for q in range(len(part_array)):
        residue_count[part_array[q]-1] += 1
        if (residue_count[part_array[q]-1] == 1):
            simple_35[5*(part_array[q]-1)] = q+1
        elif(residue_count[part_array[q]-1] == math.floor(simple_7[part_array[q]-1]/4)):
            simple_35[5*(part_array[q]-1)+1] = q+1
        elif(residue_count[part_array[q]-1] == math.floor(simple_7[part_array[q]-1]/2)):
            simple_35[5*(part_array[q]-1)+2] = q+1
        elif(residue_count[part_array[q]-1] == math.floor(simple_7[part_array[q]-1]*0.75)):
            simple_35[5*(part_array[q]-1)+3] = q+1
        elif(residue_count[part_array[q]-1] == simple_7[part_array[q]-1]):
            simple_35[5*(part_array[q]-1)+4] = q+1
    for o in range(7):
        simple_7[o] /= len(part_array)
    for p in range(21):       
        simple_21[p] /= len(part_array)
    for m in range(35):       
        simple_35[m] /= len(part_array)
    simple_63_vector = simple_7 + simple_21 + simple_35
    return simple_63_vector

'''
Implement conjoint triad algorithm
'''
def conjoint_triad(protein_array):
    local_operate_array = aac_7_number_description(protein_array)
    vector_3_matrix = [[a,b,c,0] for a in range(1,8) for b in range(1,8) for c in range(1,8)]
    for m in range(len(local_operate_array)-2):
        vector_3_matrix[(local_operate_array[m]-1)*49+(local_operate_array[m+1]-1)*7+(local_operate_array[m+2]-1)][3] += 1
    CT_array=[]
    for q in range(343):
        CT_array+=[vector_3_matrix[q][3]]
    return CT_array


'''
Implement Auto Covariance algorithm
'''
def auto_Covariance(protein_array, locate_feature):
    lg = 30 #will affect 'ac_array' down below
    AC_array = [[0 for u in range(lg)] for v in range(7)]
    mean_feature = [0,0,0,0,0,0,0]
    for j in range(len(mean_feature)):
        for i in range(len(protein_array)):
            if (protein_array[i]=='X' or protein_array[i]=='U' or protein_array[i]==' '):
                continue
            mean_feature[j] += locate_feature[protein_array[i]][j]
    for k in range(len(mean_feature)):
        mean_feature[k] /= len(protein_array)
    for lag in range(lg):
        for ac_fea in range(len(mean_feature)):
            AC_array[ac_fea][lag] = acsum(protein_array, lag, mean_feature, ac_fea, locate_feature) 
    Auto_Covariance_feature = []
    for o in range(len(AC_array)):
        for p in range(len(AC_array[0])):
            Auto_Covariance_feature += [AC_array[o][p]]
    return Auto_Covariance_feature

def acsum(protein_array, lag, mean_feature, ac_fea, locate_feature):
    phychem_sum = 0
    for i in range (len(protein_array)-lag):
        if(protein_array[i]=='X' or protein_array[i+lag]=='X' or protein_array[i]=='U' or protein_array[i+lag]=='U' or protein_array[i]==' ' or protein_array[i+lag]==' '):
            continue
        phychem_sum += (locate_feature[protein_array[i]][ac_fea]-mean_feature[ac_fea]) * (locate_feature[protein_array[i+lag]][ac_fea]-mean_feature[ac_fea])
    phychem_sum /= (len(protein_array)-lag)
    return phychem_sum
 
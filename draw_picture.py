# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 09:51:58 2017

@author: 罗骏
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import _read_data_method
import _adjusted_wavelets_combination

'''
The histogram will shows the performance gap between the different wavelet (or set of wavelets)
and the contrastive feature vector construction methods, the evaluation parameters are:
Accuracy, Recall, F1 score, Mathew’s correlation coefficient
'''

opposite_path = os.path.abspath('')

DWT_name = ['bior1.1', 'bior1.3', 'bior2.2', 'bior3.1', 'bior3.3','coif1', 'db1', 'db2', 'db3', 'db4',
                'haar', 'rbio1.1', 'rbio1.3','rbio2.2', 'rbio3.1', 'rbio3.3', 'sym2', 'sym3', 'sym4']
Compare_method_name = ['AC','PseAAC','LD','iPP-Esml(db1)','iPP-Esml(db2)']
color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896',
                  '#9467bd', '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', 
                  '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']           

def show_picture(local_performance_data, local_compare_performance_data, species_name, metric, mod = 1):
    global opposite_path
    global color_sequence
    
    global Compare_method_name
    if mod == 1:
        global DWT_name
        clf_name = ['GradientBoosting','ExtraTrees','KNeighbors','QuadraticDiscriminant','RandomForestClassifier','Stacking']
    else:
        DWT_name, file_name = _adjusted_wavelets_combination.adjusted_wavelets_combination(mod)
        num_DWT_name = len(DWT_name)
        clf_name = ['ExtraTrees','QuadraticDiscriminant']
    num_DWT_name = len(DWT_name)
    x = np.arange(num_DWT_name)
    total_width = 2.4 #
    width = total_width/num_DWT_name
    x = x - (total_width - width) / len(clf_name)
    parameters = [] #
    for temp_clf_name in clf_name:
        single_clf_parameters = []
        for temp_DWT_name in DWT_name:
            if mod == 1:
                temp_DWT_name = (temp_DWT_name,)
            q = local_performance_data[temp_DWT_name][temp_clf_name]['confusion_matrix'] #
            if metric == 'Accuracy':
                single_clf_parameters += [(q[0]+q[3]) / (q[0]+q[1]+q[2]+q[3])] 
            elif metric == 'Recall':
                single_clf_parameters += [q[0] / (q[0]+q[1])]
            elif metric == 'f1_score' or metric == 'matthews_corrcoef':
                single_clf_parameters +=  local_performance_data[temp_DWT_name][temp_clf_name][metric]
            if mod == 1:
                del temp_DWT_name
        parameters += [single_clf_parameters]
        
    compare_parameters = [] #
    for temp_method_name in Compare_method_name:
        q = local_compare_performance_data[temp_method_name]['confusion_matrix'] #
        if (q[0]+q[1]+q[2]+q[3]) == 0:
            compare_parameters += [0]
        elif metric == 'Accuracy' and (q[0]+q[1]+q[2]+q[3]) != 0:
            compare_parameters += [(q[0]+q[3]) / (q[0]+q[1]+q[2]+q[3])]
        elif metric == 'Recall' and (q[0]+q[1]+q[2]+q[3]) != 0:
            compare_parameters += [q[0] / (q[0]+q[1])]
        elif (metric == 'f1_score' or metric == 'matthews_corrcoef') and (q[0]+q[1]+q[2]+q[3]) != 0:
            compare_parameters += local_compare_performance_data[temp_method_name][metric]
        
    for i in range(len(clf_name)):
        if mod == 1:
            plt.bar(x + i*width, parameters[i], width = width, color = color_sequence[2*i], label = clf_name[i])
        else:
            plt.bar(x + i*2*width + 1.0, parameters[i], width = 2*width, color = color_sequence[2*i], label = clf_name[i])
    for j in range(len(Compare_method_name)):
        if mod == 1:
            plt.bar(num_DWT_name+ j*8*width, compare_parameters[j], width = 2*width, color = color_sequence[19-2*j])
        else:
            plt.bar(num_DWT_name+ j*6.2*width, compare_parameters[j], width = 2*width, color = color_sequence[19-2*j])
    if mod == 1:
        temp_DWT_name = ('bior2.2',)     
    if metric == 'Accuracy':
        mark1 = local_performance_data[temp_DWT_name]['QuadraticDiscriminant']['confusion_matrix']
        mark2 = local_performance_data[temp_DWT_name]['ExtraTrees']['confusion_matrix']
        a = [(mark1[0]+mark1[3]) / (mark1[0]+mark1[1]+mark1[2]+mark1[3]), (mark2[0]+mark2[3]) / (mark2[0]+mark2[1]+mark2[2]+mark2[3])]
    elif metric == 'Recall':
        mark1 = local_performance_data[temp_DWT_name]['QuadraticDiscriminant']['confusion_matrix']
        mark2 = local_performance_data[temp_DWT_name]['ExtraTrees']['confusion_matrix']
        a = [mark1[0] / (mark1[0]+mark1[1]), mark2[0] / (mark2[0]+mark2[1])]
    elif metric == 'f1_score' or metric == 'matthews_corrcoef':
        mark1 = local_performance_data[temp_DWT_name]['QuadraticDiscriminant'][metric]
        mark2 = local_compare_performance_data['iPP-Esml(db1)'][metric]
        a = [mark1[0], mark2[0]]
    a.sort()
    y_mark = a[1]
    
    plt.title('Comparison of different methods in '+species_name)   #
    plt.xlabel('different method')                  #
    plt.ylabel(metric)
    plt.xticks(range(num_DWT_name+len(Compare_method_name)),(DWT_name+Compare_method_name),rotation=45) #
    
    if y_mark>=0.9:
        plt.ylim((0, 1))
    else:
        plt.ylim((0, y_mark+0.1))
    plt.legend(loc = 'lower left')
    fig = plt.gcf()
    fig.set_size_inches(18.0, 6)
    plt.show()
    #fig.savefig(opposite_path+'/picture_old/'+species_name+'_'+metric+'.png', bbox_inches='tight', dpi=fig.dpi)
    return 0
        
def main():
    species = ['Arabidopsis_thaliana','Bos_taurus','Caenorhabditis_elegans','Drosophila_melanogaster',
              'Homo_sapiens','Mus_musculus','Rattus_norvegicus','Saccharomyces_cerevisiae']
    four_metrics = ['Accuracy','Recall','f1_score','matthews_corrcoef']
    
    print("The histogram will shows the performance gap between the different wavelet(or set of wavelets) "
          "\nand the contrastive feature vector construction methods, the evaluation parameters are:"
          "\nAccuracy, Recall, F1 score, Mathew’s correlation coefficient"
          "\n"
          "\nHere we can randomly select N wavelets from the candidate wavelets and"
          "\nobserve the predicted performance when combinate N wavelets to providing the features."
          "\nPlease enter the 'N'（Currently support 1~5）"
          "\n"
          "\n(This program is for drawing purposes only.)"
          "\n(Experiment_WT.py needs to be run before use this script, with the same numerical 'wavelets to generate the feature')"
          "\n(We provide the experimental results of single wavelet as an example, you can enter 1 to see demo)"
          )
    mod = input()
    if mod not in ['1','2','3','4','5']:
        print('Please input a number between 0 and 6')
        return 0
    else:
        mod = int(mod)    
        
    for name in species:
        if mod == 1:
            wavelets_performance = os.path.join(opposite_path, 'data', 'test_result', 'same_spec_'+name+'.csv')
        else:
            wavelets_performance = os.path.join(opposite_path, 'data', 'test_result', 'adv_WT_mod'+str(mod)+'_'+name+'.csv')
        local_performance_data = _read_data_method.read_test_result(wavelets_performance)
        compare_result_path = os.path.join(opposite_path, 'data', 'test_result','same_spec_comparison_'+name+'.csv' )
        local_compare_performance_data = _read_data_method.read_compare_result(compare_result_path) 
        for metrics in four_metrics:
            show_picture (local_performance_data, local_compare_performance_data, name, metrics, mod)   
            
if __name__ == '__main__':
    main()
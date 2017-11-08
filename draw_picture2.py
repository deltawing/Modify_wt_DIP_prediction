# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 20:14:54 2017

@author: 罗骏
"""

import matplotlib.pyplot as plt
import os
import _read_data_method
opposite_path = os.path.abspath('')
color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896',
                  '#9467bd', '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', 
                  '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

'''
This script is to demonstrate the impact of different number of wavelets used on predictor performance.
We offer the data from one species for illustration.
'''

def collect_feature_to_draw(raw_arrangement, classifier):
    global metrics_name
    Accuracy = []
    Recall = []
    matthews_corrcoef = []
    f1_score = []
    for key in raw_arrangement:
        matrix = raw_arrangement[key][classifier]['confusion_matrix']
        Accuracy += [(matrix[0]+matrix[3]) / (matrix[0]+matrix[1]+matrix[2]+matrix[3])]
        Recall += [matrix[0] / (matrix[0]+matrix[1])]
        matthews_corrcoef += raw_arrangement[key][classifier]['matthews_corrcoef']
        f1_score += raw_arrangement[key][classifier]['f1_score']        
    return Accuracy, Recall, matthews_corrcoef, f1_score

def draw_picture (all_data, name, classifier):
    global color_sequence
    fs = 10  # fontsize
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 9))

    bplot1 = axes[0, 0].boxplot(all_data[0], vert=True, patch_artist=True)
    axes[0, 0].set_title('Accuracy', fontsize=fs)
    bplot2 = axes[0, 1].boxplot(all_data[1], vert=True, patch_artist=True)   
    axes[0, 1].set_title('Recall', fontsize=fs)
    bplot3 = axes[1, 0].boxplot(all_data[2], vert=True, patch_artist=True)   
    axes[1, 0].set_title('matthews_corrcoef', fontsize=fs)
    bplot4 = axes[1, 1].boxplot(all_data[3], vert=True, patch_artist=True)   
    axes[1, 1].set_title('f1_score', fontsize=fs)
    # fill with colors
    color = color_sequence[::2]
    for bplot in (bplot1, bplot2, bplot3, bplot4):
        for patch, this_color in zip(bplot['boxes'], color):
            patch.set_facecolor(this_color)
    for ax1 in axes:
        for ax2 in ax1:
            ax2.yaxis.grid(True)
            ax2.set_ylabel('score(%)')
    # add x-tick labels
    plt.setp(axes, xticks=[y+1 for y in range(len(all_data[0]))],
        xticklabels = ['1wave', '2waves', '3waves', '4waves', '5waves'])
    fig.suptitle("The affection of the number of waves used to decompose "+name+" using "+classifier)
    fig.subplots_adjust(hspace=0.3)
    plt.show()

def main():
    local = {}
    species = ['Caenorhabditis_elegans'] #the data from one species
    clf_we_care = ['ExtraTrees','QuadraticDiscriminant']
    for name in species:
        for classifier in clf_we_care:
            Total_Accuracy, Total_Recall, Total_matthews_corrcoef, Total_f1_score = [], [], [], []
            datapath1 = os.path.join(opposite_path,'data','test_result','same_spec_'+name+'.csv')
            local = _read_data_method.read_test_result(datapath1)
            Accuracy, Recall, matthews_corrcoef, f1_score = collect_feature_to_draw(local, classifier)
            Total_Accuracy += [Accuracy]
            Total_Recall += [Recall]
            Total_matthews_corrcoef += [matthews_corrcoef]
            Total_f1_score += [f1_score]
            for mod in ['mod2', 'mod3', 'mod4', 'mod5']:
                datapath2 = os.path.join(opposite_path,'data','test_result','adv_WT_'+ mod +'_'+ name +'.csv')
                local = _read_data_method.read_test_result(datapath2)
                Accuracy, Recall, matthews_corrcoef, f1_score = collect_feature_to_draw(local, classifier)
                Total_Accuracy += [Accuracy]
                Total_Recall += [Recall]
                Total_matthews_corrcoef += [matthews_corrcoef]
                Total_f1_score += [f1_score]
            draw_picture([Total_Accuracy, Total_Recall, Total_matthews_corrcoef, Total_f1_score], name, classifier)

if __name__ == '__main__':
    main()
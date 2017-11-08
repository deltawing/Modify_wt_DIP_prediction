# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 23:08:01 2017

@author: 罗骏
"""
def adjusted_wavelets_combination(mod):
    DWT = ['bior1.1','db2','bior2.2','db3','bior1.3','rbio3.1'] #
    wavelets_comb = []
    if mod == 1:
        arragement = [[m] for m in range(6)]
        file_name = 'mod1_'
    elif mod == 2:
        arragement = [[m,n] for m in range(6) for n in range(m+1, 6)]
        file_name = 'mod2_'
    elif mod == 3:
        arragement = [[m,n,o] for m in range(6) for n in range(m+1, 6) for o in range(n+1, 6)]
        file_name = 'mod3_'
    elif mod == 4:
        arragement = [[m,n,o,p] for m in range(6) for n in range(m+1, 6) for o in range(n+1, 6) for p in range(o+1, 6)]
        file_name = 'mod4_'
    elif mod == 5:
        arragement = [[m,n,o,p,q] for m in range(6) for n in range(m+1, 6) for o in range(n+1, 6) for p in range(o+1, 6) for q in range(p+1, 6)]
        file_name = 'mod5_'
    
    for i in arragement:
        if mod == 1:
            DWT_name = (DWT[i[0]],)
        elif mod == 2:
            DWT_name = (DWT[i[0]], DWT[i[1]])
        elif mod == 3:
            DWT_name = (DWT[i[0]], DWT[i[1]], DWT[i[2]])
        elif mod == 4:
            DWT_name = (DWT[i[0]], DWT[i[1]], DWT[i[2]], DWT[i[3]])
        elif mod == 5:
            DWT_name = (DWT[i[0]], DWT[i[1]], DWT[i[2]], DWT[i[3]], DWT[i[4]])
        wavelets_comb += [DWT_name]
        del DWT_name
        
    return wavelets_comb, file_name
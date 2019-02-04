#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 19:29:06 2017

@author: luke

Levi-Civita in python:
https://stackoverflow.com/questions/20908754/how-to-speed-up-a-vector-cross-product-calculation

getitem with multiple arguments:
    https://stackoverflow.com/questions/1685389/possible-to-use-more-than-one-argument-on-getitem
"""
import numpy as np

def make_Levi_Civita():
    eijk = np.zeros((3, 3, 3))
    eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
    eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
    return eijk

eijk = make_Levi_Civita()

class LeviCivita(object):
    def __init__(self):
        self.lc = eijk
    def __call__(self,i,j,k):
        return self.lc[i,j,k]
    
    def __getitem__(self, pos):
        i,j,k = pos
        return self.lc[i,j,k]

    def __repr__(self):
        return 'LeviCivita({})'.format(self.lc)

    def __str__(self):
        return 'LeviCivita({})'.format(self.lc)
    
if __name__ == '__main__':
    self = LeviCivita()
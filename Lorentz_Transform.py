#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 17:04:38 2017

@author: luke

nice derivation of gamma:
    https://www.youtube.com/watch?v=6Tts3gxs_cM
    
"""

import numpy as np
import matplotlib.pyplot as plt

c = 1.*1.e8

def gamma(v,c=1.*1.e8):
    return 1./np.sqrt(1.-(np.linalg.norm(v)/c)**2)

def beta(v,c=1.*1.e8):
    """
    Bx = B[1]/c
    By = B[2]/c
    Bz = B[3]/c
    """
    return (v/c)

def projection(r,v):
    magv = np.linalg.norm(v)
    return np.dot(r,v)*v/(magv**2)


class LorentzTransform(object):
    """
        v : 3 vector
        p : 3 momentum
        V : 4 vector
        EP : energy-momentum
    """
    def __init__(self,v,x):
        self.v = v
        self.x = x
        self.Lambda = self.compose_lambda(v)
        
    def compose_lambda(self,v4=None):
        """
        v4 is a 4 vector in some reference from
        to be transformed to the primed frame
        
        he General Lorentz Transformation:
        https://www.youtube.com/watch?v=iWABqiuIX6c
        
        or Ryder, Intro to GR, page 23
        """
        if v4 is None: 
            v4 = self.v
        else:
            v4 = v4-self.v
        gma = gamma(v4)
        B = beta(v4)
        normB = np.linalg.norm(B)
        nBs = normB**2
        
        outer = np.outer(B[1:],B[1:])
        outer = ((gma-1.)/nBs)*outer
        ident = np.identity(3)
        outer = ident + outer
        
        row1 = -np.ones((4,1),float).T*gma
        row1[0,0] = -row1[0,0]
        row1[0,1:] = np.multiply(row1[0,1:],B[1:])
        col1 = row1.T
        
        R = np.zeros((4,4),float)
        
        R[:,0] = col1.T
        R[0,:] = row1
        
        return R
    
    def x_mu_prime(self, xp, vp):
        """
        """
        Lambda = self.compose_lambda(vp)
        return np.dot(Lambda,xp)
    

if __name__ == "__main__":
    v4 = np.asarray([1.,
                     .1,
                     .0,
                     .0])
    x0 = np.asarray([0.,
                     0.,
                     0.,
                     0.])
    LT = LorentzTransform(v=v4,x=x0)
    
    xp = np.asarray([.2,.5,.5,0.])
    v1 = np.asarray([1.,.4,0.,0.])
    xup = LT.x_mu_prime(xp,v1)
    print xup
    
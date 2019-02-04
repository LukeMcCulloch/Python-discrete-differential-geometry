#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 17:13:08 2018

@author: lukemcculloch
"""
import numpy as np
import time
import random
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse.linalg import dsolve
from scipy.sparse import csr_matrix, csc_matrix

# cholesky sparse is here:
#https://pypi.org/project/scikit-sparse/#files
#https://github.com/scikit-sparse/scikit-sparse/
import sksparse.cholmod as skchol
#https://scikit-sparse.readthedocs.io/en/latest/cholmod.html

import weakref
#https://eli.thegreenplace.net/2009/06/12/
#               safely-using-destructors-in-python/


class HodgeDecomposition(object):
    
    def __init__(self, mesh):
        #self.mesh = weakref.ref(mesh)
        self.mesh = mesh
        self.edgeIndex   = self.mesh.enumerateEdges
        self.vertexIndex = self.mesh.enumerateVertices
        self.faceIndex   = self.mesh.enumerateFaces
        self.buildDEC()
    
    def buildDEC(self):
        mesh = self.mesh
        
        
        omega = np.zeros((mesh.nedges),float)
        for edge in mesh.edges:
            i = self.edgeIndex[edge]
            omega[i] = edge.omega
        self.omega = omega
        
        self.hodge0  = mesh.buildHodgeStar0Form(self.vertexIndex)
        self.hodge1  = mesh.buildHodgeStar1Form(self.edgeIndex)
        self.hodge2  = mesh.buildHodgeStar2Form(self.faceIndex)
        
        self.ihodge0 = mesh.diagonalInverse(self.hodge0)
        self.ihodge1 = mesh.diagonalInverse(self.hodge1)
        self.ihodge2 = mesh.diagonalInverse(self.hodge2)
        
        self.d0  = mesh.buildExteriorDerivative0Form(  
                                           edgeIndex=self.edgeIndex, 
                                           vertexIndex=self.vertexIndex)
        self.d1  = mesh.buildExteriorDerivative1Form( 
                                           faceIndex=self.faceIndex, 
                                           edgeIndex=self.edgeIndex)
        self.d0T = self.d0.T
        self.d1T = self.d1.T
        
        #// construct 0-form laplace matrix
        #// shift the matrix by a small constant (1e-8) to make it positive definite
        
    def decomposeField(self):
        print("Begin Field Decomposition.")
        t0master = time.time()
        self.dAlpha = self.computeExactComponent(self.omega)
        self.deltaBeta = self.computeCoExactComponent(self.omega)
        #self.dAlpha = np.zeros_like(self.deltaBeta)
        #self.deltaBeta = np.zeros_like(self.dAlpha)
        self.computeHarmonicComponent(self.omega, 
                                      self.dAlpha, 
                                      self.deltaBeta)  
        tSolve = time.time() - t0master
        print("...Decomposition completed.")
        print("Total Time {:.5f} seconds.".format(tSolve))
    
    
        return
        
    def computeExactComponent(self, omega):
        print 'dAlpha'
        print 'build LHS...'
        t0 = time.time()
        #
        #LHS = np.dot(self.hodge1,self.d0.todense())
        #ss = np.shape(omega)[0]
        #LHS = csc_matrix(LHS)
        #LHS = self.d0T.dot(LHS)
        #LHS = LHS #+ (1.e-8 * csc_matrix(np.identity(ss,float)))
        ##
        LHS = self.d0T.dot(self.hodge1)
        ss = np.shape(LHS)[0]
        LHS = csc_matrix(LHS)
        LHS = LHS.dot(self.d0)
        LHS = LHS #- (1.e-8 * csc_matrix(np.identity(ss,float)))
        #LHS = LHS + (1.e-8 * np.identity(ss,float))
        #LHS = csc_matrix(LHS)
        #
        tSolve = time.time() - t0
        print("...sparse alpha LHS completed.")
        print("alpha LHS build took {:.5f} seconds.".format(tSolve))
        
        
        print 'build RHS...'
        t0 = time.time()
        RHS = self.d0T.dot(self.hodge1.dot(omega) )
        #RHS = self.d0T.dot(omega) 
        tSolve = time.time() - t0
        print("...sparse alpha RHS completed.")
        print("alpha RHS build took {:.5f} seconds.".format(tSolve))
        
        
        print 'solve dAlpha'
        #dAlpha = dsolve.spsolve(LHS, RHS , use_umfpack=True)
        #dAlpha = scipy.sparse.linalg.cg(LHS, RHS)[0]
        
        #sparse LU solve:
        #DLU = scipy.sparse.linalg.splu(LHS)
        #dAlpha = DLU.solve(RHS)
        
        #sparse Cholesky solve:
        llt = skchol.cholesky_AAt(LHS) #factor
        dAlpha = llt(RHS)
        
        # now push alpha to a 1 form using d:
        dAlpha = self.d0.dot(dAlpha)
        #alpha = np.dot(self.d0.todense(),
        #               alpha)
        return dAlpha
        
        
        
    def computeCoExactComponent(self, omega):
        print 'system 2, Beta'
        #solve system 2 for delta Beta
        # page 117-118-119
        # scipy.linalg.lu
        print 'build LHS...'
        t0 = time.time()
        LHS = self.d1.dot(self.ihodge1)
        #ss = np.shape(LHS)[0]
        LHS = csc_matrix(LHS)
        LHS = LHS.dot(self.d1T)
        #LHS = LHS + 1.e-8 * csc_matrix(np.identity(ss,float))
        #
        tSolve = time.time() - t0
        print("...sparse Beta LHS build completed.")
        print("Beta LHS build took {:.5f} seconds.".format(tSolve))
        
        
        print 'build RHS...'
        t0 = time.time()
        #RHS = np.matmul(d1,omega)
        RHS = self.d1.dot(omega)
        tSolve = time.time() - t0
        print("...sparse Beta RHS completed.")
        print("Beta RHS build took {:.5f} seconds.".format(tSolve))
        
        
        print 'solve'
        Beta = dsolve.spsolve(LHS, RHS , 
                               use_umfpack=True)
        
        #LU decomposition:
        #DLU = scipy.sparse.linalg.splu(LHS)
        #Beta = DLU.solve(RHS)
        
        #        print 'solve complete, transform'
        #        Beta = np.dot(ihodge2,Beta)
        #        print 'transform complete, Beta complete'
        #        
        # store exact, coexact, harmonic components on the mesh edges.
        print 'decomposition field to mesh'
        
        
        
        #now pull back to a 1 form using the codifferential *d*
        # *d* Beta => *d0*
        #Beta = np.dot(hodge0,np.dot(d0,
        #              np.dot(hodge2,Beta)))
        # the easy way:
        Beta = self.d1T.dot(Beta)
        print 'solve complete, transform Beta'
        Beta = np.dot(self.ihodge1,Beta)
        print 'transform complete, Beta complete'
        return Beta
    
    def computeHarmonicComponent(self, 
                        omega, dAlpha, deltaBeta):
        """
        Also puts all components on the edges of the mesh!
        """
        self.gamma = omega - (dAlpha+deltaBeta)
        for edge in self.mesh.edges:
            i = self.edgeIndex[edge]
            edge.exactComponent    = dAlpha[i]
            edge.coexactComponent  = deltaBeta[i]
            edge.harmonicComponent = self.gamma[i]
            #edge.harmonicComponent = omega[i] - (dAlpha[i])
            #edge.harmonicComponent = omega[i] - (deltaBeta[i])
        print 'decomposition complete'
        return
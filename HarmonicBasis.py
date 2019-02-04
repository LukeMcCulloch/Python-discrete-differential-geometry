#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 16:24:22 2018

@author: lukemcculloch
"""
import numpy as np

import weakref
#https://eli.thegreenplace.net/2009/06/12/
#               safely-using-destructors-in-python/

class HarmonicBases(object):
    def __init__(self, mesh):
        #self.mesh = weakref.ref(mesh)
        self.mesh = mesh
        
    def buildClosedPrimalOneForm(self, hgenerator, edgeIndex):
        """
            Builds a closed, but not exact, primal 1-form ω.
            private
            method module:Projects.HarmonicBases#buildClosedPrimalOneForm
            param {module:Core.Halfedge[]} hgenerator: An array of halfedges 
            representing a homology generator of the input mesh.
                @link https://en.wikipedia.org/wiki/Homology_(mathematics)#Surfaces homology generator}
            
            param {Object} edgeIndex: A dictionary mapping 
            each edge of the input mesh to a unique index.
            returns {module:LinearAlgebra.DenseMatrix}
        #"""
        E = len(self.mesh.edges)
        omega = np.zeros((E,1),float)
        for h in hgenerator:
            i = edgeIndex[h.edge]
            if h.edge.anyHalfEdge is h:
                sign = 1
            else:
                sign = -1
            omega[i,0] = sign
        return omega
    
    def compute(self, hodgeDecomposition):
        """
            Computes the harmonic bases [γ1, γ2 ... γn] of the input mesh.
            method module:Projects.HarmonicBases#compute
            param {module:Projects.HodgeDecomposition} hodgeDecomposition: 
                A hodge decomposition object that
                can be used to compute the exact component 
                of the closed, but not exact, primal
                1-form ω.
            returns {module:LinearAlgebra.DenseMatrix[]}
        """
        gammas = []
        generators = self.mesh.generators
        if len(generators)>0:
            edgeIndex = self.mesh.enumerateEdges
            for generator in generators:
                omega = self.buildClosedPrimalOneForm(
                        generator, edgeIndex)
                dAlpha = hodgeDecomposition.computeExactComponent(omega)
                gammas.append(omega-dAlpha)
        return gammas
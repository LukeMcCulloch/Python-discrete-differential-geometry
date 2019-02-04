#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 01:56:29 2018

@author: lukemcculloch
"""
from HodgeDecomposition import HodgeDecomposition
from TreeCotree import TreeCotree
from HarmonicBasis import HarmonicBases

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

import weakref
#https://eli.thegreenplace.net/2009/06/12/
#               safely-using-destructors-in-python/

class TrivialConnections(object):
    """
	/**
	 * This class implements the {@link https://www.cs.cmu.edu/~kmcrane/Projects/DDG/paper.pdf trivial connections} algorithm to compute a smooth
	 * 1-form vector fields on a surface mesh.
	 * @constructor module:Projects.TrivialConnections
	 * @param {module:Core.Geometry} geometry The input geometry of the mesh this class acts on.
	 * @property {Object} vertexIndex A dictionary mapping each vertex of the input mesh to a unique index.
	 * @property {Object} edgeIndex A dictionary mapping each edge of the input mesh to a unique index.
	 * @property {module:LinearAlgebra.DenseMatrix[]} bases The harmonic bases [γ1, γ2 ... γn] of the input mesh.
	 * @property {module:LinearAlgebra.SparseMatrix} P The period matrix of the input mesh.
	 * @property {module:LinearAlgebra.SparseMatrix} A The 0-form laplace matrix d0^T star1 d0 of the input mesh.
	 * @property {module:LinearAlgebra.SparseMatrix} hodge1 The hodge star 1-form matrix of the input mesh.
	 * @property {module:LinearAlgebra.SparseMatrix} d0 The exterior derivaitve 0-form matrix of the input mesh.
	 */
    """
    def __init__(self, mesh):
        #self.mesh = weakref.ref(mesh)
        self.mesh = mesh
        self.edgeIndex   = mesh.enumerateEdges
        self.vertexIndex = mesh.enumerateVertices
        
        hodgeDecomposition = HodgeDecomposition(mesh)
        
        treeCotree = TreeCotree(mesh)
        treeCotree.buildGenerators()
        
        harmonicBases = HarmonicBases(mesh)
        self.bases = harmonicBases.compute(hodgeDecomposition)
        
        
        self.A = harmonicBases.d0T.dot(harmonicBases.hodge1)
        self.A = self.A.dot(harmonicBases.d0)
        ss = mesh.nedges
        self.A = self.A + (1.e-8 * csc_matrix(np.identity(ss,float)))
        self.hodge1 = harmonicBases.hodge1
        self.d0 = harmonicBases.d0
        #// build period matrix and store relevant DEC operators
        
    
    def buildPeriodMatrix(self):
        return 
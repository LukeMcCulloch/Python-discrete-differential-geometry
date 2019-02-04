#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 16:20:39 2018

@author: lukemcculloch
"""

import weakref
#https://eli.thegreenplace.net/2009/06/12/
#               safely-using-destructors-in-python/

class TreeCotree(object):
    """
    See page 120 of Sigraph DDG notes, author Keanan Crane
    
	 * This class computes the {@link https://www.cs.cmu.edu/~kmcrane/Projects/DDG/paper.pdf tree cotree} decomposition of a surface mesh
	 * to build its {@link https://en.wikipedia.org/wiki/Homology_(mathematics)#Surfaces homology generators}.
	 * @constructor module:Projects.TreeCotree
	 * @param {module:Core.Mesh} mesh The input mesh this class acts on.
	 * @property {module:Core.Mesh} mesh The input mesh this class acts on.
	 * @property {vertexParent} vertexParent A dictionary mapping each vertex of the input mesh to
	 * its parent in the primal spanning tree.
	 * @property {faceParent} faceParent A dictionary mapping each face of the input mesh to
	 * its parent in the dual spanning tree.
     
    """
    def __init__(self, mesh, vertexParent = None, faceParent=None):
        #self.mesh = weakref.ref(mesh)
        self.mesh = mesh
        if vertexParent is None:
            self.vertexParent = {}
        else:
            self.vertexParent = vertexParent
        if faceParent is None:
            self.faceParent = {}
        else:
            self.faceParent = faceParent
        # get ready to plot:
        self.treePositions = []
        self.cotreePositions = []
        self.generatorPositions = []
    
    def buildPrimalSpanningTree(self):
        """
         /**
        	 * Builds a primal spanning tree on a boundaryless mesh.
          *  private usage (not enforced)
          *  method module:Projects.TreeCotree#buildPrimalSpanningTree
          */
        Do
            -procB: first vertex
            -While bag is not empty Do
                pull a vertex V from the bag
                loop over V's neighbors, N
                    -For all unvisited neighbors:
                        procB(N)
                        procA(N)
            
            procA:
                -add the vert-neighbor edge to the tree
            procB:
                -put a vertex in a bag here named queue
                 marking that vertex as visited
        """
        #dummy tree:
        for v in self.mesh.verts:
            self.vertexParent[v] = v
        
        root = self.mesh.verts[0]
        queue = [root] #anything in the queue named 'bag' is visited
        while len(queue) != 0:
            u = queue.pop(0)
            for v in u.adjacentVerts():         #neighbor loop
                if self.vertexParent[v] is v and v is not root:
                    self.vertexParent[v] = u    #add to tree
                    queue.append(v)             #mark as visted
                    
        #corrected tree: self.vertexParent
        return
    
    def inPrimalSpanningTree(self, halfedge):
        """
          /**
           * Checks whether a halfedge is in the primal spanning tree.
           * private
           * method module:Projects.TreeCotree#inPrimalSpanningTree
           * param {module:Core.Halfedge} h A halfedge on the input mesh.
           * returns {boolean}
        """
        u = halfedge.vertex
        v = halfedge.twin.vertex
        return self.vertexParent[u] is v or self.vertexParent[v] is u
    
    def buildDualSpanningCotree(self):
        """
        	*
         * Builds a dual spanning tree on a boundaryless mesh.
         * private
         * method module:Projects.TreeCotree#buildDualSpanningCotree
        """
        for f in self.mesh.faces:
            self.faceParent[f] = f
        
        root = self.mesh.faces[0]
        queue = [root] #anything in the queue named 'bag' is visited
        g = None
        while len(queue) != 0:
            f = queue.pop(0)
            for h in f.adjacentHalfEdges():          #neighbor loop
                if not self.inPrimalSpanningTree(h): #edges cannot cross!
                    g = h.twin.face
                    
                    if (self.faceParent[g] is g) and (g is not root):
                        self.faceParent[g] = f      #add to tree
                        queue.append(g)             #mark as visted
        return
    
    
    def inDualSpanningTree(self, halfedge):
        """
        	 /**
            Checks whether a halfedge is in the dual spanning tree.
            private
            method module:Projects.TreeCotree#inDualSpanningTree
            param {module:Core.Halfedge} h A halfedge on the input mesh.
            returns {boolean}
    	      */
        """
        f = halfedge.face
        g = halfedge.twin.face
        return self.faceParent[f] is g or self.faceParent[g] is f
    
    def sharedHalfedge(self, f, g):
        for h in f.adjacentHalfEdges():
            if h.twin.face is g:
                return h
        assert(False),  "Line 1020, sharedHalfedge, HalfEdgeMesh - List Implementation:"+\
                        " Code should not reach here!"
                        
    def buildGenerators(self):
        self.buildPrimalSpanningTree() #T  -build T first so that 
        self.buildDualSpanningCotree() #T*   dual edges do not 
                                       #      cross edges in T
        generators = []
        # for each edge in e 
        for e in self.mesh.edges:
            h = e.anyHalfEdge
            
            #for each e that is not in T nor crossed by T*
            if not self.inPrimalSpanningTree(h) and not self.inDualSpanningTree(h):
                #"""
                #    -follow both of its endpoints back to the root of T
                #    (traverse the nodes up the chain of parents)
                #    -The resulting loop is a generator!
                #"""
                tempGenerator1 = []
                f = h.face
                while self.faceParent[f] is not f:
                    parent = self.faceParent[f]
                    tempGenerator1.append(self.sharedHalfedge(f, parent))
                    f = parent
                
                tempGenerator2 = []
                f = h.twin.face
                while self.faceParent[f] is not f:
                    parent = self.faceParent[f]
                    tempGenerator2.append(self.sharedHalfedge(f, parent))
                    f = parent
                
                #"""
                #   Eliminate crossings:
                #"""
                m = len(tempGenerator1) - 1
                n = len(tempGenerator2) - 1
                maxit = self.mesh.nedges+1
                iternum = 0
                while tempGenerator1[m] is tempGenerator2[n] and iternum<maxit:
                    m -= 1
                    n -= 1
                    iternum +=1
                generator = [h]
                if iternum >= maxit: print (
                        'WARNING, forced terminiation of homology generator:', 
                        '\n iteration excedence at edge {}'.format(e) )
                for i in range(0,m+1): generator.append(tempGenerator1[i].twin)
                for i in range(n,0,-1): generator.append(tempGenerator2[i])
                
                generators.append(generator)
        return generators

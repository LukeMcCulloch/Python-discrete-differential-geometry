# Basic application to load a mesh from file and view it in a window

# Python imports
import sys, os
import euclid as eu
import time
import random
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse.linalg import dsolve
from scipy.sparse import csr_matrix, csc_matrix
import sksparse.cholmod as skchol

## Imports from this project
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core')) # hack to allow local imports without creaing a module or modifying the path variable
from InputOutput import *
from MeshDisplay import MeshDisplay
#from HalfEdgeMesh import *
from HalfEdgeMesh_ListImplementation import *
from Utilities import *
#from Solvers import solvePoisson



from HodgeDecomposition import HodgeDecomposition

"""
import pydec
import numpy as np
from scipy.linalg import lu
A = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
p, l, u = lu(A)
#"""
def main(inputfile, show=False, 
         StaticGeometry=False, partString='part1',
         is_simple=True):

    # Get the path for the mesh to load from the program argument
    if(len(sys.argv) == 3 and sys.argv[1] == 'simple'):
        filename = sys.argv[2]
        simpleTest = True
    elif(len(sys.argv) == 3 and sys.argv[1] == 'fancy'):
        filename = sys.argv[2]
        simpleTest = False 
    elif inputfile is not None:
        filename = inputfile
        simpleTest = is_simple
    else:
        print("ERROR: Incorrect call syntax. Proper syntax is 'python Assignment5.py MODE path/to/your/mesh.obj', where MODE is either 'simple' or 'fancy'")
        exit()

    # Read in the mesh
    mesh = HalfEdgeMesh(readMesh(filename))
    

    # Create a viewer object
    winName = 'DDG Assignment5 -- ' + os.path.basename(filename)
    meshDisplay = MeshDisplay(windowTitle=winName)
    meshDisplay.setMesh(mesh)


    ###################### BEGIN YOUR CODE
    
    # DDGSpring216 Assignment 5
    # 
    # In this programming assignment you will implement Helmholtz-Hodge decomposition of covectors.
    #
    # The relevant mathematics and algorithm are described in section 8.1 of the course notes.
    # You will also need to implement the core operators in discrete exterior calculus, described mainly in 
    # section 3.6 of the course notes.
    #
    # This code can be run with python Assignment5.py MODE /path/to/you/mesh.obj. MODE should be
    # either 'simple' or 'fancy', corresponding to the complexity of the input field omega that is given.
    # It might be easier to debug your algorithm on the simple field first. The assignment code will read in your input 
    # mesh, generate a field 'omega' as input, run your algorithm, then display the results.
    # The results can be viewed as streamlines on the surface that flow with the covector field (toggle with 'p'),
    # or, as actual arrows on the faces (toggle with 'l'). The keys '1'-'4' will switch between the input, exact,
    # coexact, and harmonic fields (respectively).
    # 
    # A few hints:
    #   - Try performing some basic checks on your operators if things don't seem right. For instance, applying the 
    #     exterior derivative twice to anything should always yield zero.
    #   - The streamline visualization is easy to look at, but can be deceiving at times. For instance, streamlines
    #     are not very meaningful where the actual covectors are near 0. Try looking at the actual arrows in that case
    #     ('l').
    #   - Many inputs will not have any harmonic components, especially genus 0 inputs. Don't stress if the harmonic 
    #     component of your output is exactly or nearly zero.
    
    
    # Implement the body of each of these functions...
   
#    def assignEdgeOrientations(mesh):
#        """
#        Assign edge orientations to each edge on the mesh.
#        
#        This method will be called from the assignment code, you do not need to explicitly call it in any of your methods.
#
#        After this method, the following values should be defined:
#            - edge.orientedHalfEdge (a reference to one of the halfedges touching that edge)
#            - halfedge.orientationSign (1.0 if that halfedge agrees with the orientation of its
#                edge, or -1.0 if not). You can use this to make much of your subsequent code cleaner.
#
#        This is a pretty simple method to implement, any choice of orientation is acceptable.
#        """
#        for edge in mesh.edges:
#            edge.orientedHalfEdge = edge.anyHalfEdge
#            edge.anyHalfEdge.orientationSign = -1.0
#            edge.anyHalfEdge.twin.orientationSign = 1.0
#        return

    def diagonalInverse(A):
        """
        Returns the inverse of a sparse diagonal matrix. Makes a copy of the matrix.
        
        We will need to invert several diagonal matrices for the algorithm, but scipy does
        not offer a fast method for inverting diagonal matrices, which is a very easy special
        case. As such, this is a useful helper method for you.

        Note that the diagonal inverse is not well-defined if any of the diagonal elements are
        0.0. This needs to be acconuted for when you construct the matrices.
        """
        ncol,nrow = np.shape(A)
        assert(ncol==nrow),'ERROR: Diagonal inverse only make sense for a symmetric matrix'
        
        #B = 1./np.diag(A)
        
        for i in range(ncol):
            A[i,i] = 1./A[i,i] #B[i]

        return A
    
    
#    @property
#    @cacheGeometry
#    def circumcentricArea(self):
#        """
#        Compute the area of the circumcentric dual cell for this vertex. 
#        Returns a positive scalar.
#
#        This gets called on a vertex, so 'self' will be a reference to the vertex.
#
#        The image on page 78 of the course notes may help you visualize this. 
#        (TLM:  not sure what this references any more)
#        
#        
#        TLM note for those like me who miss the obvious: 
#            You are not computing the circumcenter!
#            Go straight to the area!
#            
#        real source, slide 62:
#            http://brickisland.net/DDGFall2017/wp-content/uploads/2017/09/
#                                CMU_DDG_Fall2017_06_DiscreteExteriorCalculus.pdf
#        """
#        # vl      = list(self.adjacentVerts())
#        # fl      = list(self.adjacentFaces())
#        DualArea = 0.
#        for face in self.adjacentFaces():
#            #v1 = face.anyHalfEdge.vertex.position
#            #v2 = face.anyHalfEdge.next.vertex.position
#            #v3 = face.anyHalfEdge.next.next.vertex.position
#            l1 = norm(face.anyHalfEdge.vector)              #||v1-v3||
#            l2 = norm(face.anyHalfEdge.next.vector)         #||v2-v1||
#            l3 = norm(face.anyHalfEdge.next.next.vector)    #||v3-v2||
#            
#            s = .5*(l1+l2+l3)
#            DualArea +=  np.sqrt(s*(s-l1)*(s-l2)*(s-l3))
#        
#        
#        return DualArea
#    Vertex.circumcentricArea = circumcentricArea

#    @property
#    @cacheGeometry
#    def circumcentricDualArea(self):
#        """
#        Compute the area of the circumcentric dual cell for this vertex. 
#        Returns a positive scalar.
#
#        This gets called on a vertex, so 'self' will be a reference to the vertex.
#
#        The image on page 78 of the course notes may help you visualize this. 
#        (TLM:  not sure what this references any more)
#        
#        
#        TLM note for those like me who miss the obvious: 
#            You are not computing the circumcenter!
#            Go straight to the area!
#            
#        real source, slide 62:
#            http://brickisland.net/DDGFall2017/wp-content/uploads/2017/09/
#                                CMU_DDG_Fall2017_06_DiscreteExteriorCalculus.pdf
#        """
#        DualArea = 0.
#        for hedge in self.adjacentHalfEdges():
#            cak = hedge.cotan
#            lik = norm(hedge.vector)
#            caj = hedge.next.next.cotan
#            lij = norm(hedge.next.next.vector)
#            
#            DualArea +=  (lij**2 *cak) + (lik**2 * caj)
#        
#        
#        return DualArea/8.
#    Vertex.circumcentricDualArea = circumcentricDualArea


#    def buildHodgeStar0Form(mesh, vertexIndex):
#        """
#        Build a sparse matrix encoding the Hodge operator on 0-forms for this mesh.
#        Returns a sparse, diagonal matrix corresponding to vertices.
#
#        The discrete hodge star is a diagonal matrix where each entry is
#        the (area of the dual element) / (area of the primal element). You will probably
#        want to make use of the Vertex.circumcentricDualArea property you just defined.
#        
#        TLM as seen in notes:
#        By convention, the area of a vertex is 1.0.
#        """
#        nrows = ncols = len(mesh.verts)
#        vertex_area = 1.0
#        
#        Hodge0Form = np.zeros((nrows,ncols),float)
#        for i,vert in enumerate(mesh.verts):
#            vi = vertexIndex[vert]
#            Hodge0Form[vi,vi] = vert.circumcentricDualArea #/primal vertex_area
#            #Hodge0Form[vi,vi] = vert.barycentricDualArea #/primal vertex_area
#        return Hodge0Form
#
#    
#    def buildHodgeStar1Form(mesh, edgeIndex):
#        """
#        Build a sparse matrix encoding the Hodge operator on 1-forms for this mesh.
#        Returns a sparse, diagonal matrix corresponding to edges.
#        
#        The discrete hodge star is a diagonal matrix where each entry is
#        the (area of the dual element) / (area of the primal element). The solution
#        to exercise 26 from the previous homework will be useful here.
#        
#        TLM: cotan formula again.  see ddg notes page 89
#        see also source slide 56 (did you mean slide 62?):
#            http://brickisland.net/DDGFall2017/wp-content/uploads/2017/09/
#                                CMU_DDG_Fall2017_06_DiscreteExteriorCalculus.pdf
#
#        Note that for some geometries, some entries of hodge1 operator may be exactly 0.
#        This can create a problem when we go to invert the matrix. To numerically sidestep
#        this issue, you probably want to add a small value (like 10^-8) to these diagonal 
#        elements to ensure all are nonzero without significantly changing the result.
#        """
#        nrows = ncols = len(mesh.edges)
#        Hodge1Form = np.zeros((nrows,ncols),float)
#        
#        for i,edge in enumerate(mesh.edges):
#            ei = edgeIndex[edge]
#            w = (( edge.anyHalfEdge.cotan + edge.anyHalfEdge.twin.cotan ) *.5) + 1.e-8
#            #Hodge1Form[ei,ei] = edge.cotanWeight + 1.e-8
#            Hodge1Form[ei,ei] = w
#        return Hodge1Form
#    
#    
#    def buildHodgeStar2Form(mesh, faceIndex):
#        """
#        Build a sparse matrix encoding the Hodge operator on 2-forms for this mesh
#        Returns a sparse, diagonal matrix corresponding to faces.
#
#        The discrete hodge star is a diagonal matrix where each entry is
#        the (area of the dual element) / (area of the primal element).
#        
#        
#        TLM hint hint!, vertex is => (dual) vertex:
#        By convention, the area of a vertex is 1.0.
#        
#        
#        TLM: see also source slide 57:
#            http://brickisland.net/DDGFall2017/wp-content/uploads/2017/09/
#                                CMU_DDG_Fall2017_06_DiscreteExteriorCalculus.pdf
#        """
#        nrows = ncols = len(mesh.faces)
#        Hodge2Form = np.zeros((nrows,ncols),float)
#        
#        for i,face in enumerate(mesh.faces):
#            fi = faceIndex[face]
#            Hodge2Form[fi,fi] = 1./face.area
#            #Hodge2Form[fi,fi] = 1./face.AreaToDualVertexCicumcentric #circumcentric
#        return Hodge2Form
#
#    
#    def buildExteriorDerivative0Form(mesh, edgeIndex, vertexIndex):
#        """
#        Build a sparse matrix encoding the exterior derivative on 0-forms.
#        Returns a sparse matrix.
#
#        See section 3.6 of the course notes for an explanation of DEC.
#        
#        0form -> 1form
#        
#        In [2]: ed
#        Out[2]: <Edge #0>
#        
#        In [3]: ed.anyHalfEdge
#        Out[3]: <HalfEdge #11661>
#        
#        In [4]: ed.anyHalfEdge.vertex
#        Out[4]: <Vertex #0>
#        
#        In [5]: ed.anyHalfEdge.vertex.position
#        Out[5]: array([1.25, 0.  , 0.  ])
#        
#        In [6]: ed.anyHalfEdge.twin.vertex.position
#        Out[6]: array([ 1.246147,  0.      , -0.098074])
#        
#        In [7]: ed.anyHalfEdge.vertex.position - ed.anyHalfEdge.twin.vertex.position
#        Out[7]: array([0.003853, 0.      , 0.098074])
#        
#        In [8]: ed.anyHalfEdge.vector
#        Out[8]: array([0.003853, 0.      , 0.098074])
#        
#        ## so ed.anyHalfEdge.vector runs 
#            from anyHalfEdge.twin.vertex 
#            to   anyHalfEdge.vertex
#        
#        In [9]: ed.anyHalfEdge.twin.vector
#        Out[9]: array([-0.003853,  0.      , -0.098074])
#        """
#        vert_edge_incidence = np.zeros((mesh.nedges,mesh.nverts),float)
#        #        for vertex in mesh.verts:
#        #            vj = vertexIndex[vertex]
#        #            for edge in vertex.adjacentEdges():
#        #                ei = edgeIndex[edge]
#        #                
#        #                value = edge.orientedHalfEdge.orientationSign
#        #                if vertex is edge.anyHalfEdge.vertex:
#        #                    # then we are at edge.anyHalfEdge.vertex, 
#        #                    # i.e., the end of this half edge's vector. (not the start of the vector)
#        #                    value = -value
#        #                
#        #                vert_edge_incidence[ei,vj] = value
#                
#        
#        for edge in mesh.edges:
#            ei = edgeIndex[edge]
#            
#            vh1 = edge.orientedHalfEdge.vertex
#            vh2 = edge.orientedHalfEdge.twin.vertex
#            
#            ci = vertexIndex[vh1]
#            cj = vertexIndex[vh2]
#            
#            #value = edge.orientedHalfEdge.orientationSign
#            
#            vert_edge_incidence[ei,ci] =  1. #-value
#            vert_edge_incidence[ei,cj] = -1. #value
#        return csr_matrix( vert_edge_incidence )
#        #return vert_edge_incidence
#    
#    
#    def buildExteriorDerivative1FormOLD(mesh, faceIndex, edgeIndex):
#        """
#        Build a sparse matrix encoding the exterior derivative on 1-forms.
#        Returns a sparse matrix.
#         
#        See section 3.6 of the course notes for an explanation of DEC.
#        """
#        edge_face_incidence = np.zeros((mesh.nfaces,mesh.nedges),float)
#        for face in mesh.faces:
#            fi = faceIndex[face]
#            v = list(face.adjacentVerts()) #0,1,2
#            #tv = []
#            for edge in face.adjacentEdges():
#                ej = edgeIndex[edge]
#                value = edge.orientedHalfEdge.orientationSign
#                
#                #anyHalfEdge vector goes from 
#                #  anyHalfEdge.twin.vertex to anyHalfEdge.vertex
#                edge_start = edge.anyHalfEdge.twin.vertex
#                edge_end = edge.anyHalfEdge.vertex
#                if edge_start is v[0]:
#                    if edge_end is v[1]:
#                        value = value
#                    else:
#                        value = -value
#                elif edge_start is v[1]:
#                    if edge_end is v[2]:
#                        value = value
#                    else:
#                        value = -value
#                else:
#                    assert(edge_start is v[2])
#                    if edge_end is v[0]:
#                        value = value
#                    else:
#                        value = -value
#                
##                tv.append([edge,value])
#                edge_face_incidence[fi,ej] = value
#        return edge_face_incidence
#    def buildExteriorDerivative1Form(mesh, faceIndex, edgeIndex):
#        """
#        Build a sparse matrix encoding the exterior derivative on 1-forms.
#        Returns a sparse matrix.
#         
#        See section 3.6 of the course notes for an explanation of DEC.
#        """
#        edge_face_incidence = np.zeros((mesh.nfaces,mesh.nedges),float)
#        for face in mesh.faces:
#            fi = faceIndex[face]
#            #v = list(face.adjacentVerts()) #0,1,2
#            #tv = []
#            for he in face.adjacentHalfEdges():
#                ej = edgeIndex[he.edge]
#                #value = he.orientationSign
#                #tv.append([edge,value])
#                if he is he.edge.orientedHalfEdge:
#                    edge_face_incidence[fi,ej] =  1. #value
#                else:
#                    edge_face_incidence[fi,ej] = -1. #-value
#                    
#        #return edge_face_incidence
#        return csr_matrix( edge_face_incidence )

#    def decomposeField(mesh):
#        """
#        Decompose a covector in to exact, coexact, and harmonic components
#
#        The input mesh will have a scalar named 'omega' on its edges (edge.omega)
#        representing a discretized 1-form. This method should apply Helmoltz-Hodge 
#        decomposition algorithm (as described on page 107-108 of the course notes) 
#        to compute the exact, coexact, and harmonic components of omega.
#
#        This method should return its results by storing three new scalars on each edge, 
#        as the 3 decomposed components: edge.exactComponent, edge.coexactComponent,
#        and edge.harmonicComponent.
#
#        Here are the primary steps you will need to perform for this method:
#            
#            - Create indexer objects for the vertices, faces, and edges. Note that the mesh
#              has handy helper functions pre-defined 
#              for each of these: mesh.enumerateEdges() etc.
#            
#            - Build all of the operators we will need using 
#              the methods you implemented above:
#                  hodge0, hodge1, hodge2, d0, and d1. 
#              You should also compute their inverses and
#                  transposes, as appropriate.
#
#            - Build a vector which represents the input covector (from the edge.omega values)
#
#            - Perform a linear solve for the exact component, as described in the algorithm
#            
#            - Perform a linear solve for the coexact component, as described in the algorithm
#
#            - Compute the harmonic component as the part which is neither exact nor coexact
#
#            - Store your resulting exact, coexact, and harmonic components on the mesh edges
#
#        This method will be called by the assignment code, you do not need to call it yourself.
#        """
#
#        """1)Create indexer objects for the vertices, faces, and edges. Note that the mesh
#              has handy helper functions pre-defined for each of these: mesh.enumerateEdges() etc. """
#        
#        t0master = time.time()
#        edgeIndex   = mesh.enumerateEdges
#        vertexIndex = mesh.enumerateVertices
#        faceIndex   = mesh.enumerateFaces
#        
#        """2)Build all of the operators we will need using the methods you implemented above:
#              hodge0, hodge1, hodge2, d0, and d1. You should also compute their inverses and
#              transposes, as appropriate."""
#        hodge0  = mesh.buildHodgeStar0Form(vertexIndex)
#        ihodge0 = diagonalInverse(hodge0)
#        hodge1  = mesh.buildHodgeStar1Form( edgeIndex)
#        #hodge2  = buildHodgeStar2Form(mesh, faceIndex)
#        ihodge1 = diagonalInverse(hodge1)
#        #ihodge2 = diagonalInverse(hodge2)
#        d0  = mesh.buildExteriorDerivative0Form(  
#                                           edgeIndex=edgeIndex, 
#                                           vertexIndex=vertexIndex)
#        d0T = d0.T
#        d1  = mesh.buildExteriorDerivative1Form( 
#                                           faceIndex=faceIndex, 
#                                           edgeIndex=edgeIndex)
#        d1T = d1.T
#        
#        
#        print 'shape d0 = ',np.shape(d0)
#        print 'shape d1 = ',np.shape(d1)
#        #print 'shape hodge0 = ',np.shape(hodge0)
#        print 'shape hodge1 = ',np.shape(hodge1)
#        #print 'shape hodge2 = ',np.shape(hodge2)
#        
#        
#        
#        omega = np.zeros((mesh.nedges),float)
#        for edge in mesh.edges:
#            i = edgeIndex[edge]
#            omega[i] = edge.omega
#        
#        #solve system 1 for d alpha
#        # page 117-118-119
#        # scipy.linalg.cholesky
#        print 'system 1, alpha'
#        print 'build LHS...'
#        #LHS = np.matmul(d0T,
#        #             np.matmul(hodge1,d0))
#        t0 = time.time()
#        LHS = np.dot(ihodge0,
#                     d0T.dot(hodge1))
#        ss = np.shape(LHS)[0]
#        LHS = csr_matrix(LHS)
#        LHS = LHS.dot(d0)
#        LHS = LHS + (1.e-8 * csr_matrix(np.identity(ss,float)))
#        #llt = scipy.linalg.cholesky(LHS,lower=True)
#        tSolve = time.time() - t0
#        print("...sparse alpha LHS completed.")
#        print("alpha LHS build took {:.5f} seconds.".format(tSolve))
#        print 'build RHS...'
#        #RHS = np.matmul(d0T,
#        #             np.matmul(hodge1,omega))
#        RHS = np.dot(ihodge0,
#                     d0T.dot(hodge1.dot(omega))
#                     )
#        print 'type RHS = ',type(RHS)
#        print 'solve'
#        #alpha = np.linalg.solve(LHS,RHS)
#        alpha = dsolve.spsolve(LHS, RHS , 
#                               use_umfpack=True)
#        #alpha = scipy.sparse.linalg.cg(llt,RHS)
#        #alpha = dsolve.spsolve(csr_matrix(llt), RHS , 
#        #                       use_umfpack=True)
#        
#        print 'solve complete, alpha complete'
#        
#        
#        
#        
#        #solve system 2 for delta Beta
#        # page 117-118-119
#        # scipy.linalg.lu
#        print 'system 2, Beta'
#        print 'build LHS...'
#        #LHS = np.matmul(d1,
#        #             np.matmul(ihodge1,d1T))
#        t0 = time.time()
#        LHS = d1.dot(ihodge1)
#        
#        #ss = np.shape(LHS)[0]
#        
#        LHS = csr_matrix(LHS)
#        LHS = LHS.dot(d1T)
#        
#        #LHS = csr_matrix(LHS)
#        LHS = LHS #+ 1.e-8 * csr_matrix(np.identity(ss,float))
#        
#        tSolve = time.time() - t0
#        print("...sparse Beta LHS build completed.")
#        print("Beta LHS build took {:.5f} seconds.".format(tSolve))
#        print 'build RHS...'
#        #RHS = np.matmul(d1,omega)
#        RHS = d1.dot(omega)
#        print 'solve'
#        #Beta = np.linalg.solve(LHS,RHS)
#        Beta = dsolve.spsolve(LHS, RHS , 
#                               use_umfpack=True)
#        #        print 'solve complete, transform'
#        #        Beta = np.dot(ihodge2,Beta)
#        #        print 'transform complete, Beta complete'
#        #        
#        # store exact, coexact, harmonic components on the mesh edges.
#        print 'decomposition field to mesh'
#        
#        
#        print 'Now push alpha and Beta into 1 forms'
#        # now push alpha to a 1 form using d:
#        alpha = d0.dot(alpha)
#        """  Say we start with a primal 2-form on a primal face.
#        Applying the star operator takes us to a dual 0-form on a dual vertex.  
#        Taking the differential getsus to a dual 1-form on a dual edge.  
#        And finally, another star operator 
#        brings us to a primal 1-formon a primal edge."""
#        #now pull back to a 1 form using the codifferential *d*
#        # *d* Beta => *d0*
#        #Beta = np.dot(hodge0,
#        #              np.dot(d0,
#        #              np.dot(hodge2,Beta)))
#        # the easy way:
#        Beta = d1T.dot(Beta)
#        print 'solve complete, transform'
#        Beta = np.dot(ihodge1,Beta)
#        print 'transform complete, Beta complete'
#        
#        #Beta = np.zeros_like(alpha)
#        for edge in mesh.edges:
#            i = edgeIndex[edge]
#            edge.exactComponent    = alpha[i]
#            edge.coexactComponent  = Beta [i]
#            edge.harmonicComponent = omega[i] - (alpha[i] + Beta[i])
#            #edge.harmonicComponent = omega[i] - (Beta[i])
#        print 'decomposition complete'
#            
#        tSolve = time.time() - t0master
#        print("...Decomposition completed.")
#        print("Total Time {:.5f} seconds.".format(tSolve))
    
    

    def enumerateVertices(mesh):
        """
        Assign a unique index from 0 to (N-1) to each vertex in the mesh. Should
        return a dictionary containing mappings {vertex ==> index}.

        You will want to use this function in your solutions below.
        """
        #        index_map = {}
        #        index = 0
        #        for vv in mesh.verts:
        #            index_map[vv] = index
        #            index += 1
        return mesh.enumerateVertices
    
    
    
    @property
    @cacheGeometry
    def adjacency(self):
        index_map = enumerateVertices(self)
        nrows = ncols = len(mesh.verts)
        adjacency = np.zeros((nrows,ncols),int) 
        for vv in mesh.verts:
            ith = index_map[vv]
            avlist = list(vv.adjacentVerts())
            for av in avlist:
                jth = index_map[av]
                adjacency[ith,jth] = 1 
        return adjacency
        


    #################################
    # Part 1: Dense Poisson Problem #
    #################################
    # Solve a Poisson problem on the mesh. The primary function here
    # is solvePoissonProblem_dense(), it will get called when you run
    #   python Assignment3.py part1 /path/to/your/mesh.obj
    # and specify density values with the mouse (the press space to solve).
    #
    # Note that this code will be VERY slow on large meshes, because it uses
    # dense matrices.

    def buildLaplaceMatrix_sparse(mesh, index_map=None):
        """
        Build a Laplace operator for the mesh, with a dense representation

        'index' is a dictionary mapping {vertex ==> index}
        TLM renamed to index_map

        Returns the resulting matrix.
        """
        if index_map is None:
            index_map = mesh.enumerateVertices()
        
        nrows = ncols = len(mesh.verts)
        #        adjacency = np.zeros((nrows,ncols),int)
        #        for vv in mesh.verts:
        #            ith = index_map[vv]
        #            avlist = list(vv.adjacentVerts())
        #            for av in avlist:
        #                jth = index_map[av]
        #                adjacency[ith,jth] = 1 
                
                
        Laplacian = np.zeros((nrows,ncols),float)
        for vi in mesh.verts:
            ith = index_map[vi]
            ll = list(vi.adjacentEdgeVertexPairs())
            for edge, vj in ll:
                jth = index_map[vj]
                #                Laplacian[ith,jth] = np.dot(vj.normal,
                #                                             edge.cotanWeight*(vj.position - 
                #                                                       vi.position)
                #                                             )
                w1 = edge.anyHalfEdge.cotan
                w2 = edge.anyHalfEdge.twin.cotan
                W = .5*(w1+w2)
                #W = edge.cotanWeight
                if ith == jth:
                    pass 
                else:
                    Laplacian[ith,jth] = W
            
            Laplacian[ith,ith] = -(sum(Laplacian[ith]) )#+ 1.e-8)
            
        return csr_matrix(Laplacian)


    def buildMassMatrix_dense(mesh, index):
        """
        Build a mass matrix for the mesh.

        Returns the resulting matrix.
        """
        nrows = ncols = len(mesh.verts)
        
        #MassMatrix = np.zeros((nrows),float)
        MassMatrix = np.zeros((nrows,ncols),float)
        for vert in mesh.verts:
            i = index[vert]
            #MassMatrix[i,i] = 1./vert.dualArea
            MassMatrix[i,i] = vert.barycentricDualArea  
            #MassMatrix[i,i] = vert.circumcentricDualArea
        
        return MassMatrix


    def solvePoisson(mesh, densityValues):
        """
        Solve a Poisson problem on the mesh. The results should be stored on the
        vertices in a variable named 'solutionVal'. You will want to make use
        of your buildLaplaceMatrix_dense() function from above.

        densityValues is a dictionary mapping {vertex ==> value} that specifies
        densities. The density is implicitly zero at every vertex not in this
        dictionary.

        When you run this program with 'python Assignment3.py part1 path/to/your/mesh.obj',
        you will get to click on vertices to specify density conditions. See the
        assignment document for more details.
        """
        index_map = mesh.enumerateVertices
        L = buildLaplaceMatrix_sparse(mesh, index_map)
        M = buildMassMatrix_dense(mesh, index_map) #M <= 2D
        totalArea = mesh.totalArea
        
        rho = np.zeros((len(mesh.verts),1),float)
        for key in densityValues:
            #index_val = index_map[key]
            print 'key dual area = ', key.barycentricDualArea
            rho[index_map[key]] = densityValues[key]#*key.dualArea
        
        nRows,nCols = np.shape(M)
        totalRho = sum(M.dot(rho))
        #rhoBar = np.ones((nRows,1),float)*(totalRho/totalArea)
        rhoBar = totalRho/totalArea
        rhs = M.dot(rhoBar-rho)
        #rhs = np.matmul(M,(rho-rhoBar) )
        #rhs = np.dot(M,rho)
        #
        # SwissArmyLaplacian, 
        #   page 179 Cu = Mf is better conditioned
        # assert(Cu == L) ??
        #sol_vec = np.linalg.solve(L, np.dot(M,rho) )
        
        #sparse:
        #sol_vec = dsolve.spsolve(L, np.dot(M,rho) , use_umfpack=True)
        # standard:
        #sol_vec = dsolve.spsolve(L, rhs , use_umfpack=True)
        
        
        #sparse Cholesky solve:
        llt = skchol.cholesky_AAt(L) #factor
        sol_vec = llt(rhs)
        
        #eigen:
        #sol_vec = np.zeros((nRows),float)
        #scipy.sparse.linalg.lobpcg(L,sol_vec,rhs) #@eigensolver
        
        #sol_vec = dsolve.spsolve(L, rho , use_umfpack=True)
        vert_sol = {}
        for vert in mesh.verts:
            key = index_map[vert]
            #print 'TLM sol_vec = ',sol_vec[key]
            vert.solutionVal = sol_vec[key]
            vert_sol[vert] = sol_vec[key]
            if rho[key]:
                vert.densityVal = rho[key]
            else:
                vert.densityVal = 0.
                
        
        return vert_sol


    ###################### END YOUR CODE


    ### More prep functions
    def generateFieldConstant(mesh):
        print("\n=== Using constant field as arbitrary direction field")
        for vert in mesh.verts:
            vert.vector = vert.projectToTangentSpace(Vector3D(1.4, 0.2, 2.4))

    def generateFieldSimple(mesh):
        for face in mesh.faces:
            face.vector = face.center + Vector3D(-face.center[2], face.center[1], face.center[0])
            face.vector = face.projectToTangentSpace(face.vector)

    def gradFromPotential(mesh, potAttr, gradAttr):
        # Simply compute gradient from potential
        for vert in mesh.verts:
            sumVal = Vector3D(0.0,0.0,0.0)
            sumWeight = 0.0
            vertVal = getattr(vert, potAttr)
            for he in vert.adjacentHalfEdges():
                sumVal += he.edge.cotanWeight * (getattr(he.vertex, potAttr) - vertVal) * he.vector
                sumWeight += he.edge.cotanWeight
            setattr(vert, gradAttr, normalize(sumVal))

    def generateInterestingField(mesh,
                                 divscale=1.,
                                 curlscale=1.):
        print("\n=== Generating a hopefully-interesting field which has all three types of components\n")


        # Somewhat cheesy hack: 
        # We want this function to generate the exact same result on repeated runs of the program to make
        # debugging easier. This means ensuring that calls to random.sample() return the exact same result
        # every time. Normally we could just set a seed for the RNG, and this work work if we were sampling
        # from a list. However, mesh.verts is a set, and Python does not guarantee consistency of iteration
        # order between runs of the program (since the default hash uses the memory address, which certainly
        # changes). Rather than doing something drastic like implementing a custom hash function on the 
        # mesh class, we'll just build a separate data structure where vertices are sorted by position,
        # which allows reproducible sampling (as long as positions are distinct).
        sortedVertList = list(mesh.verts)
        sortedVertList.sort(key= lambda x : (x.position[0], x.position[1], x.position[2]))
        random.seed(100)


        # Generate curl-free (ish) component
        curlFreePotentialVerts = random.sample(sortedVertList, max((2,len(mesh.verts)/1000)))
        potential = divscale
        bVals = {}
        for vert in curlFreePotentialVerts:
            bVals[vert] = potential
            potential *= -1.
        smoothPotential = solvePoisson(mesh, bVals)
        mesh.applyVertexValue(smoothPotential, "curlFreePotential")
        gradFromPotential(mesh, "curlFreePotential", "curlFreeVecGen")


        # Generate divergence-free (ish) component
        divFreePotentialVerts = random.sample(sortedVertList, max((2,len(mesh.verts)/1000)))
        potential = curlscale
        bVals = {}
        for vert in divFreePotentialVerts:
            bVals[vert] = potential
            potential *= -1.
        smoothPotential = solvePoisson(mesh, bVals)
        mesh.applyVertexValue(smoothPotential, "divFreePotential")
        gradFromPotential(mesh, "divFreePotential", "divFreeVecGen")
        for vert in mesh.verts:
            normEu = eu.Vector3(*vert.normal)
            vecEu = eu.Vector3(*vert.divFreeVecGen)
            vert.divFreeVecGen = vecEu.rotate_around(normEu, pi / 2.0) # Rotate the field by 90 degrees


        # Combine the components
        for face in mesh.faces:
            face.vector = Vector3D(0.0, 0.0, 0.0)
            for vert in face.adjacentVerts():
                face.vector = 1.0 * vert.curlFreeVecGen + 1.0 * vert.divFreeVecGen
            
            face.vector = face.projectToTangentSpace(face.vector)

        
        # clear out leftover attributes to not confuse people
        for vert in mesh.verts:
            del vert.curlFreeVecGen
            del vert.curlFreePotential
            del vert.divFreeVecGen
            del vert.divFreePotential


    # Verify the orientations were defined. Need to do this early, since they are needed for setup
    def checkOrientationDefined(mesh):
        """Verify that edges have oriented halfedges and halfedges have orientation signs"""
    
        for edge in mesh.edges:
            if not hasattr(edge, 'orientedHalfEdge'):
                print("ERROR: Edges do not have orientedHalfEdge defined. Cannot proceed")
                exit()
        for he in mesh.halfEdges:
            if not hasattr(he, 'orientationSign'):
                print("ERROR: halfedges do not have orientationSign defined. Cannot proceed")
                exit()


    # Verify the correct properties are defined after the assignment is run
    def checkResultTypes(mesh):
        
        for edge in mesh.edges:
            # Check exact
            if not hasattr(edge, 'exactComponent'):
                print("ERROR: Edges do not have edge.exactComponent defined. Cannot proceed")
                exit()
            if not isinstance(edge.exactComponent, float):
                print("ERROR: edge.exactComponent is defined, but has the wrong type. Type is " + str(type(edge.exactComponent)) + " when if should be 'float'")
                exit()
        
            # Check cocoexact
            if not hasattr(edge, 'coexactComponent'):
                print("ERROR: Edges do not have edge.coexactComponent defined. Cannot proceed")
                exit()
            if not isinstance(edge.coexactComponent, float):
                print("ERROR: edge.coexactComponent is defined, but has the wrong type. Type is " + str(type(edge.coexactComponent)) + " when if should be 'float'")
                exit()

            # Check harmonic 
            if not hasattr(edge, 'harmonicComponent'):
                print("ERROR: Edges do not have edge.harmonicComponent defined. Cannot proceed")
                exit()
            if not isinstance(edge.harmonicComponent, float):
                print("ERROR: edge.harmonicComponent is defined, but has the wrong type. Type is " + str(type(edge.harmonicComponent)) + " when if should be 'float'")
                exit()



    # Visualization related
    def covectorToFaceVectorWhitney(mesh, covectorName, vectorName):
        """lookout wedge below! (tlm)
        
        this code is okay because it is able to show the initial 
        vector field correctly.
        """
        for face in mesh.faces:
            pi = face.anyHalfEdge.vertex.position
            pj = face.anyHalfEdge.next.vertex.position
            pk = face.anyHalfEdge.next.next.vertex.position
            eij = pj - pi
            ejk = pk - pj
            eki = pi - pk
            N = cross(eij, -eki)
            A = 0.5 * norm(N)
            N /= 2*A
            wi = getattr(face.anyHalfEdge.edge, covectorName) * face.anyHalfEdge.orientationSign
            wj = getattr(face.anyHalfEdge.next.edge, covectorName) * face.anyHalfEdge.next.orientationSign
            wk = getattr(face.anyHalfEdge.next.next.edge, covectorName) * face.anyHalfEdge.next.next.orientationSign
            #s = (1.0 / (6.0 * A)) * cross(N, wi*(eki-ejk) + wj*(eij-eki) + wk*(ejk-eij))
            s = (1.0 / (6.0 * A)) * cross(N, wi*(ejk-eij) + wj*(eki-ejk) + wk*(eij-eki))

            setattr(face, vectorName, s) 
        return
            


    # Visualization related
    def covectorToFaceVectorWhitneyJS(mesh, covectorName, vectorName):
        """lookout wedge below! (tlm)
        """
        edgeIndex = mesh.enumerateEdges
        for face in mesh.faces:
            h = face.anyHalfEdge
            
            pi = h.vertex.position
            pj = h.next.vertex.position
            pk = h.next.next.vertex.position
            eij = pj - pi
            ejk = pk - pj
            eki = pi - pk
            
            #cij = 
            #if h.edge.anyHalfEdge is not h:
             #   cij *= -1.
            
            
            wij = getattr(face.anyHalfEdge.edge, covectorName) #* face.anyHalfEdge.orientationSign
            wjk = getattr(face.anyHalfEdge.next.edge, covectorName) #* face.anyHalfEdge.next.orientationSign
            wki = getattr(face.anyHalfEdge.next.next.edge, covectorName) #* face.anyHalfEdge.next.next.orientationSign
            if h.edge.anyHalfEdge is not h:
                wij *= -1
            if h.next.edge.anyHalfEdge is not h:
                wjk *= -1
            if h.next.next.edge.anyHalfEdge is not h:
                wki *= -1
            
                
            
            #N = cross(eij, -eki)
            #A = 0.5 * norm(N)
            #N /= 2*A
            A = face.area 
            N = face.normal 
            #
            a = (eki - ejk)*wij
            b = (eij - eki)*wjk
            c = (ejk - eij)*wki
            
            #pystyle
            #a=wij*(ejk-eij) 
            #b=wjk*(eki-ejk) 
            #c=wki*(eij-eki)
            
            
            
            #s = (1.0 / (6.0 * A)) * cross(N, wij*(eki-ejk) + wjk*(eij-eki) + wki*(ejk-eij))
            #s = (1.0 / (6.0 * A)) * cross(N, wij*(ejk-eij) + wjk*(eki-ejk) + wki*(eij-eki))
            s = cross(N, (a+b+c))*(1./(6.*A))
            setattr(face, vectorName, s) 

    def flat(mesh, vectorFieldName, oneFormName):
        """
        Given a vector field defined on faces, compute the corresponding (integrated) 1-form 
        on edges.
        """

        for edge in mesh.edges:

            oe = edge.orientedHalfEdge

            if not oe.isReal:
                val2 = getattr(edge.orientedHalfEdge.twin.face, vectorFieldName)
                meanVal = val2
            elif not oe.twin.isReal:
                val1 = getattr(edge.orientedHalfEdge.face, vectorFieldName)
                meanVal = val1
            else:
                val1 = getattr(edge.orientedHalfEdge.face, vectorFieldName)
                val2 = getattr(edge.orientedHalfEdge.twin.face, vectorFieldName)
                meanVal = 0.5 * (val1 + val2)
    
            setattr(edge, oneFormName, dot(edge.orientedHalfEdge.vector, meanVal))


    ### Actual main method:

    # get ready
    mesh.assignEdgeOrientations()
    checkOrientationDefined(mesh)

    # Generate a vector field on the surface
    if simpleTest:
        generateFieldSimple(mesh)
        #generateFieldConstant(mesh)
    else:
        generateInterestingField(mesh,
                                 divscale  = 1.,
                                 curlscale = 1.)
    
    flat(mesh, 'vector', 'omega')

    # Apply the decomposition from this assignment
    print("\n=== Decomposing field in to components")
    #decomposeField(mesh) 
    #hd = HodgeDecomposition(mesh)
    mesh.HodgeDecomposition()
    mesh.hodgeDecomposition.decomposeField()
    print("=== Done decomposing field ===\n\n")

    # Verify everything necessary is dfined for the output
    checkResultTypes(mesh)
#
    # Convert the covectors to face vectors for visualization
    covectorToFaceVectorWhitney(mesh, 
                                "exactComponent", 
                                "omega_exact_component")
    covectorToFaceVectorWhitney(mesh, 
                                "coexactComponent", 
                                "omega_coexact_component")
    covectorToFaceVectorWhitney(mesh, 
                                "harmonicComponent", 
                                "omega_harmonic_component")
    covectorToFaceVectorWhitney(mesh, 
                                "omega", 
                                "omega_original")
#
#
    # Register a vector toggle to switch between the vectors we just defined
    vectorList = [  {'vectorAttr':'omega_original', 
                     'key':'1', 
                     'colormap':'Spectral', 
                     'vectorDefinedAt':'face'},
                    {'vectorAttr':'omega_exact_component', 
                     'key':'2', 
                     'colormap':'Blues', 
                     'vectorDefinedAt':'face'},
                    {'vectorAttr':'omega_coexact_component', 
                     'key':'3', 
                     'colormap':'Reds', 
                     'vectorDefinedAt':'face'},
                    {'vectorAttr':'omega_harmonic_component', 
                     'key':'4', 
                     'colormap':'Greens', 
                     'vectorDefinedAt':'face'}
                 ]
    meshDisplay.registerVectorToggleCallbacks(vectorList)
    
    print 'Computing Tree Cotree decomposition'
    mesh.TreeCotree()
    print 'Computing Tree Cotree generators'
    mesh.TreeCotree_compute_generators()
    print 'Plot Tree Cotree generators'
    mesh.setup_TreeCotree_plot()
#    meshDisplay.registerVectorToggleCallbacks(
#                    [{'vectorAttr':'g1', 
#                      'key':'5', 
#                      'colormap':'Oranges', 
#                     'vectorDefinedAt':'face'},
#                    {'vectorAttr':'g2', 
#                      'key':'6', 
#                      'colormap':'Oranges', 
#                     'vectorDefinedAt':'face'}])

    # Start the GUI
    if show:
        meshDisplay.startMainLoop()
    
    
    return mesh, meshDisplay


    #bricksland 
    # http://brickisland.net/DDGSpring2016/wp-content/uploads/2016/04/DDG_CMUSpring2016_ExteriorCalculus.pdf
if __name__ == """__main__""": 
    # 
    # sphere_small
    # sphere_large - 7.3 MB !
    # cube
    # blub #twice as big as the torus
    # torus
    # test-torus.ply
    # teapot
    # bunny
    # spot1
    # plane_small
    # plane_large
    # f-16
    #"""
    #mesh, meshDisplay = main(inputfile='../meshes/test-torus.ply',
    mesh, meshDisplay = main(inputfile='../meshes/sphere_small.obj',
                             show=True,
                             StaticGeometry=True,
                             partString='part3',
                             is_simple=False)
    #"""
    """
    mesh, meshDisplay = main(inputfile='../meshes/torus.obj',
                             show=True,
                             StaticGeometry=True,
                             partString='part3',
                             is_simple=False)  
    #"""
    self = mesh
    v = mesh.verts[0]
    f = list(v.adjacentFaces())[0]
    he = list(v.adjacentHalfEdges())[0]
    edge = list(v.adjacentEdges())[0]
    
    edgeIndex = mesh.enumerateEdges
    vertexIndex = mesh.enumerateVertices
    faceIndex = mesh.enumerateFaces
    3
    print 'n verts = ',mesh.nverts # alpha (0 form) 162
    print 'n edges = ',mesh.nedges # omega (1 form) 480
    print 'n faces = ',mesh.nfaces # Beta  (2 form) 320
    print 'n hedgs = ',mesh.nhalfedges
    
    d0 = mesh.buildExteriorDerivative0Form()
    d1 = mesh.buildExteriorDerivative1Form()
    d0T = d0.T
    d1T = d1.T
    #h0 = mesh.buildHodgeStar0Form()
    h1 = mesh.buildHodgeStar1Form()
    h2 = mesh.buildHodgeStar2Form()
    
    ihodge1 = mesh.diagonalInverse(h1)
    ihodge2 = mesh.diagonalInverse(h2)
    
    bb = d1.dot(d0)
    print 'boundary of boundary representation: ',np.shape(bb)
    aa = bb.toarray()
    print 'any non-zero dd ? =>', np.amin(aa),np.amax(aa)
    
    
    #    t0 = time.time()
    #    LHS = np.matmul(d0T.todense(),
    #                 np.matmul(h1,d0.todense()))
    #    tSolve = time.time() - t0
    #    print("...numpy solution completed.")
    #    print("Solution took {:.5f} seconds.".format(tSolve))

    
    #    t0 = time.time()
    #    LHS = d0T.dot(h1)
    #    LHS = csr_matrix(LHS)
    #    LHS = LHS.dot(d0)
    #    tSolve = time.time() - t0
    #    print("...sparse solution completed.")
    #    print("Solution took {:.5f} seconds.".format(tSolve))

    
    v0 = mesh.verts[0]
    v1 = mesh.verts[1]
    v4 = mesh.verts[4]
    v5 = mesh.verts[5]
    
    e27 = mesh.edges[27]
    e28 = mesh.edges[28]
    e2 = mesh.edges[2]
    e0 = mesh.edges[0]
    e1 = mesh.edges[1]
    
    f0 = mesh.faces[0]
    f9 = mesh.faces[9]
    f10 = mesh.faces[10]
    
    f0edgelist = list(f0.adjacentEdges())
    f0vertlist = list(f0.adjacentVerts())
    
    f9edgelist = list(f9.adjacentEdges())
    f9vertlist = list(f9.adjacentVerts())
    
    f10edgelist = list(f10.adjacentEdges())
    f10vertlist = list(f10.adjacentVerts())
    
    """
    omega = np.zeros((mesh.nedges),float)
    for edge in mesh.edges:
        i = edgeIndex[edge]
        omega[i] = edge.omega
        
    
    for edge in mesh.edges:
        i = edgeIndex[edge]
        print edge.exactComponent,edge.coexactComponent,edge.harmonicComponent 
        
        
    #"""
    """
    import sys, os
    import euclid as eu
    import time
    import random
    import numpy as np
    import scipy.sparse
    import scipy.sparse.linalg
    from scipy.sparse.linalg.dsolve import linsolve
    from scipy.sparse.linalg import dsolve
    from scipy.sparse import csr_matrix
    import sksparse.cholmod as skchol
    #"""

    """
    LHS=np.asarray([[1.,2.],[0.2,6.8]])
    RHS = np.asarray([[1.],[2.]])
    #    llt = scipy.linalg.cho_factor(LHS) 
    #    alpha = scipy.linalg.cho_solve(c_and_lower=llt,
    #                                   b=RHS)
    #    llt = scipy.linalg.cholesky_banded(LHS) 
    #    alpha = scipy.linalg.cho_solve_banded(c_and_lower=llt,
    #                                   b=RHS)
    alpha = dsolve.spsolve(LHS, RHS , 
                           use_umfpack=True)
    
    
    LHS=np.asarray([[1.,2.],[0.2,6.8]])
    RHS = np.asarray([[1.],[2.]])
    #LU decomposition:
    DLU = scipy.linalg.lu_factor(LHS)
    Beta = scipy.linalg.lu_solve(lu_and_piv=DLU,
                                b=RHS)
    #"""
    
    
    
    """
    mesh.TreeCotree()
    mesh.TreeCotree_compute_generators()
    mesh.setup_TreeCotree_plot()
    meshDisplay.registerVectorToggleCallbacks(
                    [{'vectorAttr':'g1', 
                      'key':'5', 
                      'colormap':'Oranges', 
                     'vectorDefinedAt':'face'}])
    
    
    # generators are on halfedges
    # vector fields are on edges...
    meshDisplay.setShapeColorFromScalar("solutionVal", 
                                        definedOn='vertex', 
                                        cmapName="seismic")
    meshDisplay.generateAllMeshValues()
    
    mesh.HarmonicBasis()
    mesh.HarmonicBases.compute(mesh.hodgeDecomposition)
    #"""
    
    
    
    #    for edge in mesh.edges:
    #        print (edge.omega - \
    #               ( edge.exactComponent + edge.coexactComponent)) - \
    #                       edge.harmonicComponent
                       
    
    #    for edge in mesh.edges:
    #        print ( (edge.omega - edge.coexactComponent) -
    #               (edge.exactComponent + edge.harmonicComponent),
    #               edge.exactComponent , edge.harmonicComponent)
    #        
    #        
    #    for edge in mesh.edges:
    #        print ( edge.omega, 
    #               edge.exactComponent,
    #               edge.coexactComponent ,
    #               edge.harmonicComponent)
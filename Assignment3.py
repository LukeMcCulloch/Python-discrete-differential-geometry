# Basic application to load a mesh from file and view it in a window

# Python imports
import sys, os
import euclid as eu
import time
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse.linalg import dsolve
from scipy.sparse import csr_matrix

## Imports from this project
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core')) # hack to allow local imports without creaing a module or modifying the path variable
from InputOutput import *
from MeshDisplay import MeshDisplay
#from HalfEdgeMesh import *
from HalfEdgeMesh_ListImplementation import *
from Utilities import *


"""
an important misunderstanding about 
the cotan-Laplace matrix L: 
    it does not represent the Laplace operator. 
    Instead, it represents the 'conformal Laplacian'
    it gives you the laplacian integrated over each dual cell
    
    
TLM says:
    We are computing the Laplace operator on a surface
    We are not solving for some potential-flow satisfying various constraints
    
    The only 'constraint' is the shape of the space. --- A closed manifold.
    No extra conditions.  Just compute 
            d*d(u) = rho
    
"""


"""
import pydec
import numpy as np
from scipy.linalg import lu
A = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
p, l, u = lu(A)


from scipy.sparse import csr_matrix # row format
#"""

def main(inputfile, show=False, 
         StaticGeometry=False, partString='part1'):

    # Get the path for the mesh to load from the program argument
    if(len(sys.argv) == 3):
        partString = sys.argv[1]
        if partString not in ['part1','part2','part3']:
            print("ERROR part specifier not recognized. Should be one of 'part1', 'part2', or 'part3'")
            exit()
        filename = sys.argv[2]
    elif inputfile is not None:
        filename = inputfile
    else:
        print("ERROR: Incorrect call syntax. Proper syntax is 'python Assignment3.py partN path/to/your/mesh.obj'.")
        exit()

    # Read in the mesh
    mesh = HalfEdgeMesh(readMesh(filename),
                        staticGeometry=StaticGeometry)

    # Create a viewer object
    winName = 'DDG Assignment3 ' + partString + '-- ' + os.path.basename(filename)
    
    meshDisplay = MeshDisplay(windowTitle=winName)
    meshDisplay.setMesh(mesh)



    ###################### BEGIN YOUR CODE
    # implement the body of each of these functions
    
    ############################
    # assignment 2 code:
    ############################

    @property
    @cacheGeometry
    def faceArea(self):
        """
        Compute the area of a face. 
        Though not directly requested, this will be
        useful when computing face-area weighted normals below.
        This method gets called on a face, 
        so 'self' is a reference to the
        face at which we will compute the area.
        """
        
        v = list(self.adjacentVerts())
        a = 0.5 * norm(cross(v[1].position - v[0].position, 
                             v[2].position - v[0].position))

        return a 
    
        
    @property
    @cacheGeometry
    def vertexNormal_EquallyWeighted(self):
        """
        Compute a vertex normal using the 'equally weighted' method.
        This method gets called on a vertex, 
        so 'self' is a reference to the
        vertex at which we will compute the normal.
        
        http://brickisland.net/cs177/?p=217
        Perhaps the simplest way to get vertex normals 
        is to just add up the neighboring face normals:
        """

        normalSum = np.array([0.0,0.0,0.0])
        for face in self.adjacentFaces():
            normalSum += face.normal 
        n = normalize(normalSum)
        
        #issue:
        # two different tessellations of the same geometry 
        #   can produce very different vertex normals

        return n
        
        
    @property
    @cacheGeometry
    def vertexNormal_AreaWeighted(self):
        """
        Compute a vertex normal using 
        the 'face area weights' method.
        
        This method gets called on a vertex, 
        so 'self' is a reference to the
        vertex at which we will compute the normal.
        
        The area-weighted normal vector for this vertex"""

        normalSum = np.array([0.0,0.0,0.0])
        for face in self.adjacentFaces():
            normalSum += face.normal * face.area
        n = normalize(normalSum)
        #print 'computed vertexNormal_AreaWeighted n = ',n

        return n

    @property
    @cacheGeometry
    def vertexNormal_AngleWeighted(self):
        """
        element type : vertex
        
        Compute a vertex normal using the 
        'Tip-Angle Weights' method.
        
        This method gets called on a vertex, 
        so 'self' is a reference to the
        vertex at which we will compute the normal.
        
        A simple way to reduce dependence 
        on the tessellation is to weigh face normals 
        by their corresponding tip angles theta, i.e., 
        the interior angles incident on the vertex of interest:
        """
        normalSum = np.array([0.0,0.0,0.0])
        
        for face in self.adjacentFaces():
            
            vl = list(face.adjacentVerts())
            vl.remove(self)
            
            v1 = vl[0].position - self.position
            v2 = vl[1].position - self.position
            
            # norm ->no need for check:
            #  it doesn not matter what the sign is?
            #area = norm(cross(v1, v2))
            ##if area < 0.0000000001*max((norm(v1),norm(v2))):
            #if area <  0.:
            #    area *= -1.
            
            alpha = np.arctan2(norm(cross(v1,v2)),
                                dot(v1,v2))
            #print v1
            #print v2
            #print alpha
            #print ''
            
            
            normalSum += face.normal * alpha
        n = normalize(normalSum)

        return n


    @property
    @cacheGeometry
    def faceNormal(self):
        """
        Compute normal at a face of the mesh. 
        Unlike at vertices, there is one very
        obvious way to do this, since a face 
        uniquely defines a plane.
        This method gets called on a face, 
        so 'self' is a reference to the
        face at which we will compute the normal.
        """

        v = list(self.adjacentVerts())
        n = normalize(cross(v[1].position - v[0].position, 
                            v[2].position - v[0].position))

        return n
    @property
    @cacheGeometry
    def cotan(self):
        """
        element type : halfedge
        
        Compute the cotangent of 
        the angle OPPOSITE this halfedge. 
        This is not directly required, 
        but will be useful 
        when computing the mean curvature
        normals below.
        
        This method gets called 
        on a halfedge, 
        
        so 'self' is a reference to the
        halfedge at which we will compute the cotangent.
        
        https://math.stackexchange.com/questions/2041099/
            angle-between-vectors-given-cross-and-dot-product
            
        see half edge here:
        Users/lukemcculloch/Documents/Coding/Python/
            DifferentialGeometry/course-master/libddg_userguide.pdf
        """
        # Validate that this is on a triangle
        if self.next.next.next is not self:
            raise ValueError("ERROR: halfedge.cotan() is only well-defined on a triangle")

        
        if self.isReal:

            # Relevant vectors
            A = -self.next.vector
            B = self.next.next.vector

            # Nifty vector equivalent of cot(theta)
            val = np.dot(A,B) / norm(cross(A,B))
            return val

        else:
            return 0.0
        
        

    @property
    @cacheGeometry
    def angleDefect(self):
        """
        angleDefect <=> local Gaussian Curvature
        element type : vertex
        
        Compute the angle defect of a vertex, 
        d(v) (see Assignment 1 Exercise 8).
        
        This method gets called on a vertex, 
        so 'self' is a reference to the
        vertex at which we will compute the angle defect.
        """
        """
        el      = list(self.adjacentEdges())
        evpl    = list(self.adjacentEdgeVertexPairs())
        fl      = list(self.adjacentFaces())
        
        vl      = list(self.adjacentVerts())
        
        https://scicomp.stackexchange.com/questions/27689/
                numerically-stable-way-of-computing-angles-between-vectors
        #"""
        hl      = list(self.adjacentHalfEdges())
        lenhl = len(hl)
        hl.append(hl[0])
        
        alpha = 0.
        for i in range(lenhl):
            v1 = hl[i].vector
            v2 = hl[i+1].vector
            alpha += np.arctan2(norm(cross(v1,v2)),
                                dot(v1,v2))
        #dv = 2.*np.pi - alpha

        return 2.*np.pi - alpha
    def totalGaussianCurvature():
        """
        Compute the total Gaussian curvature 
        in the mesh, 
        meaning the sum of Gaussian
        curvature at each vertex.
        
        Note that you can access 
        the mesh with the 'mesh' variable.
        """
        tot = 0.
        for vel in mesh.verts:
            tot += vel.angleDefect
        return tot


    def gaussianCurvatureFromGaussBonnet():
        """
        Compute the total Gaussian curvature 
        that the mesh should have, given that the
        Gauss-Bonnet theorem holds 
        (see Assignment 1 Exercise 9).
        
        Note that you can access 
        the mesh with the 'mesh' variable. 
        The mesh includes members like 
        'mesh.verts' and 'mesh.faces', which are
        sets of the vertices (resp. faces) in the mesh.
        """
        V = len(mesh.verts)
        E = len(mesh.edges)
        F = len(mesh.faces)
        EulerChar = V-E+F
        return 2.*np.pi*EulerChar

    ############################
    # Part 0: Helper functions #
    ############################
    # Implement a few useful functions that you will want in the remainder of
    # the assignment.

    @property
    @cacheGeometry
    def cotanWeight(self):
        """
        Return the cotangent weight for an edge. Since this gets called on
        an edge, 'self' will be a reference to an edge.

        This will be useful in the problems below.

        Don't forget, everything you implemented for the last homework is now
        available as part of the library (normals, areas, etc). (Moving forward,
        Vertex.normal will mean area-weighted normals, unless otherwise specified)
        """
        val = 0.0
        if self.anyHalfEdge.isReal:
            val += self.anyHalfEdge.cotan
        if self.anyHalfEdge.twin.isReal:
            val += self.anyHalfEdge.twin.cotan
        val *= 0.5
        return val



    @property
    @cacheGeometry
    def vertex_Laplace(self):
        """
        element type : vertex
        
        Compute a vertex normal 
        using the 'mean curvature' method.
        
        del del phi = 2NH
        
        -picked up negative sign due to 
           cross products pointing into the page?
        
        -no they are normalized.
        
        -picked up a negative sign due to 
        the cotan(s) being defined 
        for pj, instead of pi.
        
        But how did it change anything?
        """
        
        hl      = list(self.adjacentHalfEdges())
        pi = self.position
        sumj = 0.
        ot = 1./3.
        for hlfedge in hl:
            pj = hlfedge.vertex.position
            ct1 = hlfedge.cotan
            ct2 = hlfedge.twin.cotan
            sumj += (ct1+ct2)*(pj-pi)
        #laplace = .5*sumj
        
        return normalize(-.5*sumj)



    ##
    ##*******************************************************
    ##
    @property
    @cacheGeometry
    def dualArea(self):
        """
        Return the dual area associated with a vertex. 
        Since this gets called on
        a vertex, 'self' will be a 
        reference to a vertex.

        Recall that the dual area can be 
        defined as 1/3 the area of the surrounding
        faces.
        
        http://brickisland.net/DDGFall2017/
        'the barycentric dual area associated 
        with a vertex i is equal to one-third the area
        of all triangles ijk touching i.'
        """
        fl = list(self.adjacentFaces())
        area_star = 0.
        for ff in fl:
            area_star += ff.area/3.

        return area_star

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

    def buildLaplaceMatrix_dense(mesh, index_map=None):
        """
        Build a Laplace operator for the mesh, with a dense representation

        'index' is a dictionary mapping {vertex ==> index}
        TLM renamed to index_map

        Returns the resulting matrix.
        """
        if index_map is None:
            # index_map = mesh.enumerateVertices()
            index_map = enumerateVertices(mesh)
        
        nrows = ncols = len(mesh.verts)
        adjacency = np.zeros((nrows,ncols),int)
        for vv in mesh.verts:
            ith = index_map[vv]
            avlist = list(vv.adjacentVerts())
            for av in avlist:
                jth = index_map[av]
                adjacency[ith,jth] = 1 
                
                
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
                if ith == jth:
                    pass #Laplacian[ith,jth] = edge.cotanWeight
                else:
                    Laplacian[ith,jth] = edge.cotanWeight
            
            Laplacian[ith,ith] = -sum(Laplacian[ith])
            
        return Laplacian


    def buildMassMatrix_dense(mesh, index):
        """
        Build a mass matrix for the mesh.

        Returns the resulting matrix.
        """
        nrows = ncols = len(mesh.verts)
        
        #MassMatrix = np.zeros((nrows),float)
        MassMatrix = np.zeros((nrows,ncols),float)
        for i,vert in enumerate(mesh.verts):
            #MassMatrix[i,i] = 1./vert.dualArea
            MassMatrix[i,i] = vert.dualArea
        
        return MassMatrix


    def solvePoissonProblem_dense(mesh, densityValues):
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
        index_map = enumerateVertices(mesh)
        L = buildLaplaceMatrix_dense(mesh, index_map)
        
        M = buildMassMatrix_dense(mesh,index_map) #M <= 2D
        rho = np.zeros((len(mesh.verts)),float)
        
        for key in densityValues:
            #index_val = index_map[key]
            print 'key dual area = ', key.dualArea
            rho[index_map[key]] = densityValues[key]#*key.dualArea
        
        #
        # SwissArmyLaplacian, 
        #   page 179 Cu = Mf is better conditioned
        sol_vec = np.linalg.solve(L, np.dot(M,rho) )
        
        #sparse attempts:
        #sol_vec = linsolve.spsolve(L, rho)
        #sol_vec = dsolve.spsolve(L, rho, use_umfpack=False)
        #sol_vec = dsolve.spsolve(L, rho, use_umfpack=True)
        
        for vert in mesh.verts:
            key = index_map[vert]
            #print 'TLM sol_vec = ',sol_vec[key]
            vert.solutionVal = sol_vec[key]
            if rho[key]:
                vert.densityVal = rho[key]
            else:
                vert.densityVal = 0.
                

        return 


    ##################################
    # Part 2: Sparse Poisson Problem #
    ##################################
    # Solve a Poisson problem on the mesh. The primary function here
    # is solvePoissonProblem_sparse(), it will get called when you run
    #   python Assignment3.py part2 /path/to/your/mesh.obj
    # and specify density values with the mouse (the press space to solve).
    #
    # This will be very similar to the previous part. Be sure to see the wiki
    # for notes about the nuances of sparse matrix computation. Now, your code
    # should scale well to larger meshes!

    def buildLaplaceMatrix_sparse(mesh, index_map=None):
        """
        Build a laplace operator for the mesh, with a sparse representation.
        This will be nearly identical to the dense method.

        'index' is a dictionary mapping {vertex ==> index}

        Returns the resulting sparse matrix.
        """
        if index_map is None:
            # index_map = mesh.enumerateVertices()
            index_map = enumerateVertices(mesh)
        
        nrows = ncols = len(mesh.verts)
        adjacency = np.zeros((nrows,ncols),int)
        for vv in mesh.verts:
            ith = index_map[vv]
            avlist = list(vv.adjacentVerts())
            for av in avlist:
                jth = index_map[av]
                adjacency[ith,jth] = 1 
                
                
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
                if ith == jth:
                    pass #Laplacian[ith,jth] = edge.cotanWeight
                else:
                    Laplacian[ith,jth] = edge.cotanWeight
            
            Laplacian[ith,ith] = -sum(Laplacian[ith])
            
        return csr_matrix(Laplacian)
    
    

    def buildMassMatrix_sparse(mesh, index):
        """
        Build a sparse mass matrix for the system.

        Returns the resulting sparse matrix.
        """
        nrows = ncols = len(mesh.verts)
        
        MassMatrix = np.zeros((nrows),float)
        #for i,vert in enumerate(mesh.verts):
        #    MassMatrix[i] = vert.dualArea
        
        return MassMatrix



    def solvePoissonProblem_sparse(mesh, densityValues):
        """
        Solve a Poisson problem on the mesh, using sparse matrix operations.
        This will be nearly identical to the dense method.
        The results should be stored on the vertices in a variable named 'solutionVal'.

        densityValues is a dictionary mapping {vertex ==> value} that specifies any
        densities. The density is implicitly zero at every vertex not in this dictionary.

        Note: Be sure to look at the notes on the github wiki about sparse matrix
        computation in Python.

        When you run this program with 'python Assignment3.py part2 path/to/your/mesh.obj',
        you will get to click on vertices to specify density conditions. See the
        assignment document for more details.
        """

        index_map = enumerateVertices(mesh)
        L = buildLaplaceMatrix_sparse(mesh, index_map)
        
        M = buildMassMatrix_dense(mesh,index_map) #M <= 2D
        rho = np.zeros((len(mesh.verts)),float)
        
        for key in densityValues:
            #index_val = index_map[key]
            print 'key dual area = ', key.dualArea
            rho[index_map[key]] = densityValues[key]#*key.dualArea
        
        
        # convert to sparse matrix (CSR method)
        #Lsparse = csr_matrix(L)
        #iL = np.linalg.inv(L)
        #sol_vec = np.dot(iL,rho)
        
        #sol_vec = np.linalg.solve(L, rho)
        #sol_vec = linsolve.spsolve(L, rho)
        
        #sol_vec = linsolve.spsolve(L, np.dot(M,rho) )
        #sol_vec = dsolve.spsolve(L, rho, use_umfpack=False)
        sol_vec = dsolve.spsolve(L, np.dot(M,rho) , use_umfpack=True)
        
        for vert in mesh.verts:
            key = index_map[vert]
            #print 'TLM sol_vec = ',sol_vec[key]
            vert.solutionVal = sol_vec[key]
            if rho[key]:
                vert.densityVal = rho[key]
            else:
                vert.densityVal = 0.
                
        return


    ###############################
    # Part 3: Mean Curvature Flow #
    ###############################
    # Perform mean curvature flow on the mesh. The primary function here
    # is meanCurvatureFlow(), which will get called when you run
    #   python Assignment3.py part3 /path/to/your/mesh.obj
    # You can adjust the step size with the 'z' and 'x' keys, and press space
    # to perform one step of flow.
    #
    # Of course, you will want to use sparse matrices here, so your code
    # scales well to larger meshes.

    def buildMeanCurvatureFlowOperator(mesh, 
                                       index=None, 
                                       h=None):
        """
        Construct the (sparse) mean curvature operator matrix for the mesh.
        It might be helpful to use your buildLaplaceMatrix_sparse() and
        buildMassMatrix_sparse() methods from before.

        Returns the resulting matrix.
        """
        nrows = ncols = len(mesh.verts)
        
        ##MassMatrix = np.zeros((nrows),float)
        #MassMatrix = np.zeros((nrows,ncols),float)
        #for i,vert in enumerate(mesh.verts):
        #    MassMatrix[i] = 1./vert.dualArea
        #    #MassMatrix[i,i] = 1./vert.dualArea

        Laplacian = np.zeros((nrows,ncols),float)
        for vi in mesh.verts:
            ith = index[vi]
            ll = list(vi.adjacentEdgeVertexPairs())
            for edge, vj in ll:
                jth = index[vj]
                #                Laplacian[ith,jth] = np.dot(vj.normal,
                #                                             edge.cotanWeight*(vj.position - 
                #                                                       vi.position)
                #                                             )
                if ith == jth:
                    pass #Laplacian[ith,jth] = edge.cotanWeight
                else:
                    Laplacian[ith,jth] = edge.cotanWeight
            
            Laplacian[ith,ith] = -sum(Laplacian[ith])
            
        return csr_matrix(Laplacian)

    def meanCurvatureFlow_use_numpy_solve(mesh, h):
        """
        Perform mean curvature flow on the mesh. The result of this operation
        is updated positions for the vertices; you should conclude by modifying
        the position variables for the mesh vertices.

        h is the step size for the backwards euler integration.

        When you run this program with 'python Assignment3.py part3 path/to/your/mesh.obj',
        you can press the space bar to perform this operation and z/x to change
        the step size.

        Recall that before you modify the positions of the mesh, you will need
        to set mesh.staticGeometry = False, which disables caching optimizations
        but allows you to modfiy the geometry. After you are done modfiying
        positions, you should set mesh.staticGeometry = True to re-enable these
        optimizations. You should probably have mesh.staticGeometry = True while
        you assemble your operator, or it will be very slow.
        """
        # index_map = mesh.enumerateVertices()
        index_map = enumerateVertices(mesh)
        nrows = ncols = len(mesh.verts)
        
        Id = np.identity(nrows,float)
        M = buildMassMatrix_dense(mesh,index_map) #M <= 2D
        
        MCF = buildMeanCurvatureFlowOperator(mesh,
                                             index=index_map,
                                             h=h)
        
        
        
        #
        # SwissArmyLaplacian, 
        #   page 181 (I-hC)u = u is not symmetric
        #            (M-hC)u = Mu is better conditioned
        #----------------------------------------------
        Mi = np.linalg.inv(M)
        
        L = np.matmul(Mi,MCF)
        #UpdateOperator = np.linalg.inv(Id-h*L)
        #----------------------------------------------
        #UpdateOperator = np.linalg.inv(M-h*MCF)
        
        
        
        LHS = M-h*MCF
        UpdateOperator = np.linalg.inv(LHS)
        #UpdateOperator = np.matmul(UpdateOperator,M)
        
        vertices = np.zeros((nrows,3),float)
        for i,vert in enumerate(mesh.verts):
            vertices[i] = vert.position
        LHS = Id-h*L
        
        UpdateOperator = np.linalg.solve(LHS, vertices)
        vertices = UpdateOperator
        for i,vert in enumerate(mesh.verts):
            #key = index_map[vert]
            vert.position = vertices[i]
            
#            
#        vertices = np.dot(UpdateOperator,vertices)
#        for i,vert in enumerate(mesh.verts):
#            key = index_map[vert]
#            vert.position = vertices[i]
            
        return 

    def meanCurvatureFlow(mesh, h):
        """
        Perform mean curvature flow on the mesh. The result of this operation
        is updated positions for the vertices; you should conclude by modifying
        the position variables for the mesh vertices.

        h is the step size for the backwards euler integration.

        When you run this program with 'python Assignment3.py part3 path/to/your/mesh.obj',
        you can press the space bar to perform this operation and z/x to change
        the step size.

        Recall that before you modify the positions of the mesh, you will need
        to set mesh.staticGeometry = False, which disables caching optimizations
        but allows you to modfiy the geometry. After you are done modfiying
        positions, you should set mesh.staticGeometry = True to re-enable these
        optimizations. You should probably have mesh.staticGeometry = True while
        you assemble your operator, or it will be very slow.
        """
        # index_map = mesh.enumerateVertices()
        index_map = enumerateVertices(mesh)
        nrows = ncols = len(mesh.verts)
        
        #Id = np.identity(nrows,float)
        M = buildMassMatrix_dense(mesh,index_map) #M <= 2D
        Msp = csr_matrix(M)
        
        #pure cotan operator:
        MCF = buildMeanCurvatureFlowOperator(mesh,
                                             index=index_map,
                                             h=h)
        
        
        
        #
        # SwissArmyLaplacian, 
        #   page 181 (I-hC)u = u is not symmetric
        #            (M-hC)u = Mu is better conditioned
        #----------------------------------------------
        #Mi = np.linalg.inv(M)
        #L = np.matmul(Mi,MCF)
        #UpdateOperator = np.linalg.inv(Id-h*L)
        #----------------------------------------------
        #LHS = M-h*MCF
        
        LHS = Msp - MCF.multiply(h)
        
        #UpdateOperator = np.linalg.inv(LHS)
        #UpdateOperator = np.matmul(UpdateOperator,M)
        
        UpdateOperator = dsolve.spsolve(LHS, 
                                        M , 
                                        use_umfpack=True)
        
        vertices = np.zeros((nrows,3),float)
        for i,vert in enumerate(mesh.verts):
            vertices[i] = vert.position
            
        #https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.linalg.cho_solve.html
        #UpdateOperator = scipy.linalg.cho_solve(
        #        scipy.linalg.cho_factor(LHS), 
        #          np.dot(M,vertices)) 
        
        #P, L, U = scipy.linalg.lu(LHS)
        
        # for non symmetric, numpy solve, style:
        #        LHS = Id-h*L
        #        UpdateOperator = np.linalg.solve(LHS, vertices)
        #        vertices = UpdateOperator
        #        for i,vert in enumerate(mesh.verts):
        #            #key = index_map[vert]
        #            vert.position = vertices[i]
            
        #            
        vertices = np.dot(UpdateOperator,vertices)
        for i,vert in enumerate(mesh.verts):
            #key = index_map[vert]
            vert.position = vertices[i]
            
        return 



    ###################### END YOUR CODE
    # from assignment 2:
    Face.normal = faceNormal
    Face.area = faceArea
    Vertex.normal = vertexNormal_AreaWeighted
    Vertex.vertexNormal_EquallyWeighted = vertexNormal_EquallyWeighted
    Vertex.vertexNormal_AreaWeighted = vertexNormal_AreaWeighted
    Vertex.vertexNormal_AngleWeighted = vertexNormal_AngleWeighted
    ##
    Vertex.vertex_Laplace = vertex_Laplace
    #
    #Vertex.vertexNormal_SphereInscribed = vertexNormal_SphereInscribed
    Vertex.angleDefect = angleDefect
    HalfEdge.cotan = cotan
    
    
    
    def toggleDefect():
        print("\nToggling angle defect display")
        if toggleDefect.val:
            toggleDefect.val = False
            meshDisplay.setShapeColorToDefault()
        else:
            toggleDefect.val = True
            meshDisplay.setShapeColorFromScalar("angleDefect", 
                                                cmapName="seismic")
                                                #,vMinMax=[-pi/8,pi/8])
        meshDisplay.generateFaceData()
    toggleDefect.val = False
    meshDisplay.registerKeyCallback('3', 
                                    toggleDefect, 
                                    docstring="Toggle drawing angle defect coloring")
    def computeDiscreteGaussBonnet():
        print("\nComputing total curvature:")
        computed = totalGaussianCurvature()
        predicted = gaussianCurvatureFromGaussBonnet()
        print("   Total computed curvature: " + str(computed))
        print("   Predicted value from Gauss-Bonnet is: " + str(predicted))
        print("   Error is: " + str(abs(computed - predicted)))
    meshDisplay.registerKeyCallback('z', 
                                    computeDiscreteGaussBonnet, 
                                    docstring="Compute total curvature")


    ###################### Assignment 3 stuff
    Edge.cotanWeight = cotanWeight
    Vertex.dualArea = dualArea

    # A pick function for choosing density conditions
    densityValues = dict()
    def pickVertBoundary(vert):
        """
        See MeshDisplay callbacks,
        pickVertexCallback
        for how this works!
        
        self.pickVertexCallback <== pickVertBoundary(vert)
        self.pickVertexCallback(pickObject = your_vertex)
        """
        value = 1.0 if pickVertBoundary.isHigh else -1.0
        print("   Selected vertex at position:" + printVec3(vert.position))
        print("   as a density with value = " + str(value))
        densityValues[vert] = value
        print 'densityValues = ',densityValues
        pickVertBoundary.isHigh = not pickVertBoundary.isHigh
    pickVertBoundary.isHigh = True



    # Run in part1 mode
    if partString == 'part1':

        print("\n\n === Executing assignment 2 part 1")
        print("""
        Please click on vertices of the mesh to specify density conditions.
        Alternating clicks will specify high-value (= 1.0) and low-value (= -1.0)
        density conditions. You may select as many density vertices as you want,
        but >= 2 are necessary to yield an interesting solution. When you are done,
        press the space bar to execute your solver and view the results.
        """)

        meshDisplay.pickVertexCallback = pickVertBoundary
        meshDisplay.drawVertices = True

        def executePart1Callback():
            print("\n=== Solving Poisson problem with your dense solver\n")

            # Print and check the density values
            print("Density values:")
            for key in densityValues:
                print("    " + str(key) + " = " + str(densityValues[key]))
            #if len(densityValues) < 2:
            #    print("Aborting solve, not enough density vertices specified")
            #    return

            # Call the solver
            print("\nSolving problem...")
            t0 = time.time()
            solvePoissonProblem_dense(mesh, densityValues)
            tSolve = time.time() - t0
            print("...solution completed.")
            print("Solution took {:.5f} seconds.".format(tSolve))

            print("Visualizing results...")

            # Error out intelligently if nothing is stored on vert.solutionVal
            for vert in mesh.verts:
                if not hasattr(vert, 'solutionVal'):
                    print("ERROR: At least one vertex does not have the attribute solutionVal defined.")
                    exit()
                if not isinstance(vert.solutionVal, float):
                    print("ERROR: The data stored at vertex.solutionVal is not of type float.")
                    print("   The data has type=" + str(type(vert.solutionVal)))
                    print("   The data looks like vert.solutionVal="+str(vert.solutionVal))
                    exit()

            # Visualize the result
            #            meshDisplay.setShapeColorFromScalar("solutionVal", 
            #                                                definedOn='vertex', 
            #                                                cmapName="seismic", 
            #                                                vMinMax=[-10.0,10.0])
            meshDisplay.setShapeColorFromScalar("solutionVal", 
                                                definedOn='vertex', 
                                                cmapName="seismic")
            meshDisplay.generateAllMeshValues()

        meshDisplay.registerKeyCallback(' ', 
                                        executePart1Callback, 
                                        docstring="Solve the Poisson problem and view the results")
        
        def showdensity():
            # Visualize the result
#            meshDisplay.setShapeColorFromScalar("densityVal", 
#                                                definedOn='vertex', 
#                                                cmapName="seismic", 
#                                                vMinMax=[-1.0,1.0])
            meshDisplay.setShapeColorFromScalar("densityVal", 
                                                definedOn='vertex', 
                                                cmapName="seismic")
            meshDisplay.generateAllMeshValues()

        meshDisplay.registerKeyCallback('b', 
                                        showdensity, 
                                        docstring="Show the density map for the Poisson Problem")
            

        # Start the GUI
        if show:
            meshDisplay.startMainLoop()




    # Run in part2 mode
    elif partString == 'part2':
        print("\n\n === Executing assignment 2 part 2")
        print("""
        Please click on vertices of the mesh to specify density conditions.
        Alternating clicks will specify high-value (= 1.0) and low-value (= -1.0)
        density conditions. You may select as many density vertices as you want,
        but >= 2 are necessary to yield an interesting solution. When you are done,
        press the space bar to execute your solver and view the results.
        """)

        meshDisplay.pickVertexCallback = pickVertBoundary
        meshDisplay.drawVertices = True

        def executePart2Callback():
            print("\n=== Solving Poisson problem with your sparse solver\n")

            # Print and check the density values
            print("Density values:")
            for key in densityValues:
                print("    " + str(key) + " = " + str(densityValues[key]))
            #if len(densityValues) < 2:
            #    print("Aborting solve, not enough density vertices specified")
            #    return

            # Call the solver
            print("\nSolving problem...")
            t0 = time.time()
            solvePoissonProblem_sparse(mesh, densityValues)
            tSolve = time.time() - t0
            print("...solution completed.")
            print("Solution took {:.5f} seconds.".format(tSolve))

            print("Visualizing results...")

            # Error out intelligently if nothing is stored on vert.solutionVal
            for vert in mesh.verts:
                if not hasattr(vert, 'solutionVal'):
                    print("ERROR: At least one vertex does not have the attribute solutionVal defined.")
                    exit()
                if not isinstance(vert.solutionVal, float):
                    print("ERROR: The data stored at vertex.solutionVal is not of type float.")
                    print("   The data has type=" + str(type(vert.solutionVal)))
                    print("   The data looks like vert.solutionVal="+str(vert.solutionVal))
                    exit()

            # Visualize the result
            # meshDisplay.setShapeColorFromScalar("solutionVal", definedOn='vertex', cmapName="seismic", vMinMax=[-1.0,1.0])
            meshDisplay.setShapeColorFromScalar("solutionVal", definedOn='vertex', cmapName="seismic")
            meshDisplay.generateAllMeshValues()

        meshDisplay.registerKeyCallback(' ', executePart2Callback, docstring="Solve the Poisson problem and view the results")

        # Start the GUI
        if show:
            meshDisplay.startMainLoop()



    # Run in part3 mode
    elif partString == 'part3':

        print("\n\n === Executing assignment 2 part 3")
        print("""
        Press the space bar to perform one step of mean curvature
        flow smoothing, using your solver. Pressing the 'z' and 'x'
        keys will decrease and increase the step size (h), respectively.
        """)


        stepSize = [0.01]
        def increaseStepsize():
            stepSize[0] += 0.001
            print("Increasing step size. New size h="+str(stepSize[0]))
        def decreaseStepsize():
            stepSize[0] -= 0.001
            print("Decreasing step size. New size h="+str(stepSize[0]))
        meshDisplay.registerKeyCallback('z', decreaseStepsize, docstring="Increase the value of the step size (h) by 0.1")
        meshDisplay.registerKeyCallback('x', increaseStepsize, docstring="Decrease the value of the step size (h) by 0.1")



        def smoothingStep():
            print("\n=== Performing mean curvature smoothing step\n")
            print("  Step size h="+str(stepSize[0]))

            # Call the solver
            print("  Solving problem...")
            t0 = time.time()
            meanCurvatureFlow(mesh, stepSize[0])
            tSolve = time.time() - t0
            print("  ...solution completed.")
            print("  Solution took {:.5f} seconds.".format(tSolve))

            print("Updating display...")
            meshDisplay.generateAllMeshValues()

        meshDisplay.registerKeyCallback(' ', 
                                        smoothingStep, 
                                        docstring="Perform one step of your mean curvature flow on the mesh")

        # Start the GUI
        if show:
            meshDisplay.startMainLoop()
    return mesh, meshDisplay



if __name__ == "__main__": 

    #mesh, meshDisplay = main(
    #                        inputfile='../meshes/sphere_small.obj',
    #                        show=True)
    # 
    # sphere_small
    # sphere_large - 7.3 MB !
    # torus
    # teapot
    # bunny
    # spot
    # plane_small
    # plane_large
    # f-16
    #"""
    #mesh, meshDisplay = main(inputfile='../meshes/test-torus.ply',
    mesh, meshDisplay = main(inputfile='../meshes/torus.obj',
    #mesh, meshDisplay = main(inputfile='../meshes/bunny.obj',
                             show=True,
                             StaticGeometry=False,
                             partString='part2')
    #"""
    """
    mesh, meshDisplay = main(inputfile='../meshes/torus.obj',
                             show=False,
                             StaticGeometry=True,
                             partString='part1')  
    #"""
    self = mesh
    v = mesh.verts[0]
    f = list(v.adjacentFaces())[0]
    he = list(v.adjacentHalfEdges())[0]
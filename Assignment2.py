# Basic application to load a mesh from file and view it in a window

# Python imports
import sys, os
import euclid as eu
import numpy as np

## Imports from this project
#  hack to allow local imports 
#    without creaing a module or modifying the path variable:
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core')) 
from InputOutput import *
from MeshDisplay import MeshDisplay

#from HalfEdgeMesh import *
from HalfEdgeMesh_ListImplementation import *

from Utilities import *
import pydec as dec


def main(inputfile, show=False, StaticGeometry=False):

    # Get the path for the mesh to load, either from the program argument if
    # one was given, or a dialog otherwise
    if(len(sys.argv) > 1):
        filename = sys.argv[1]
    elif inputfile is not None:
        filename = inputfile
    else:
        string1 = "ERROR: No file name specified. "
        string2 = "Proper syntax is 'python Assignment2.py path/to/your/mesh.obj'."
        print(string1+string2)
        exit()


    # Read in the mesh
    mesh = HalfEdgeMesh(readMesh(filename),
                        staticGeometry=StaticGeometry)

    # Create a viewer object
    winName = 'DDG Assignment2 -- ' + os.path.basename(filename)
    if show:
        meshDisplay = MeshDisplay(windowTitle=winName)
        meshDisplay.setMesh(mesh)

    ###################### BEGIN YOUR CODE
    # implement the body of each of these functions
    
    
    #
    #
    #    def buildLaplaceMatrix_dense(mesh, index):
    #        """
    #        Build a Laplace operator for the mesh, with a dense representation
    #
    #        'index' is a dictionary mapping {vertex ==> index}
    #
    #        Returns the resulting matrix.
    #        """
    #        #index_map = mesh.enumerateVertices()
    #        index_map = enumerateVertices(mesh)
    #        
    #        return Laplacian
    
    
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
        
        SwissArmyLaplacian.pdf, page 147
        Applying 'L' to a column bector u
        implements the cotan formula
        
        M = [square diagonal]
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
        
        return normalize(.5*sumj)

    @property
    @cacheGeometry
    def vertexNormal_MeanCurvature(self):
        """
        element type : vertex
        
        Compute a vertex normal 
        using the 'mean curvature' method.
        
        Be sure to understand 
        the relationship between 
        this method and the
        area gradient method.
        
        aka, http://brickisland.net/cs177/?p=217:
        (the remarkable fact is that the most 
        straightforward discretization of laplacian 
        leads us right back to the cotan formula! I
        n other words, the vertex normals we get from 
        the mean curvature vector are precisely 
        the same as the ones we get from the area gradient.)
        
        p 60 siggraph2013
        del del phi = 2NH
        
        This method gets called 
        on a vertex, so 'self' is a reference to the
        vertex at which we will compute the normal.
        
        
        http://brickisland.net/cs177/?p=309
        For the dual area of a vertex 
        you can simply use one-third 
        the area of the incident faces
        
        
        hl[0].next.next.next is hl[0]
        >>> True
        
        hl[0].twin.twin is hl[0]
        >>> True
        """
        
        hl      = list(self.adjacentHalfEdges())
        #        lenhl = len(hl)
        #        
        #        for hlfedge in self.adjacentHalfEdges:
        #            pass.
        pi = self.position
        sumj = 0.
        ot = 1./3.
        for hlfedge in hl:
            pj = hlfedge.vertex.position
            #ct1 = hlfedge.next.cotan
            ct2 = hlfedge.cotan
            #ct2 = hlfedge.twin.next.cotan
            ct1 = hlfedge.twin.cotan
            #dual_area = -ot*hlfedge.face.area #wtf
            sumj += (ct2+ct1)*(pj-pi)#/dual_area
        laplace = .5*sumj 
        
        """
        Picked up a sign because?
        
        
        -picked up negative sign due to 
           cross products pointing into the page?
        
        -no they are normalized.
        
        -picked up a negative sign due to 
           the cotan(s) being defined 
           for pj, instead of pi.
        
        But how did it change anything?
        """
        return normalize(laplace)
        #return normalize(laplace*(.5/self.angleDefect))

    @property
    @cacheGeometry
    def vertexNormal_SphereInscribed(self):
        """
        element type : vertex
        
        Compute a vertex normal 
        using the 'inscribed sphere' method.
        
        This method gets called on a vertex, 
        so 'self' is a reference to the
        vertex at which we will compute the normal.
        
        normal at a vertex pi
        can be expressed purely in terms of the 
        edge vectors
            ej = pj-pi
        where pj
            are the immediate neighbors
            of pi
        """
        
        vl      = list(self.adjacentVerts())
        lenvl = len(vl)
        vl.append(vl[0])
        
#        Ns = Vector3D(0.0,0.0,0.0)
#        for i in range(lenvl):
#            v1 = vl[i].position
#            v2 = vl[i+1].position
#            e1 = v1 - self.position
#            e2 = v2 - self.position
#            Ns += cross(e1,e2)/((norm(e1)**2)*
#                                 (norm(e2)**2))
            
        
        hl      = list(self.adjacentHalfEdges())
        lenhl = len(hl)
        hl.append(hl[0])
        Ns = Vector3D(0.0,0.0,0.0)
        for i in range(lenhl):
            e1 = hl[i].vector
            e2 = hl[i+1].vector
            #Ns += cross(e1,e2)/(sum(abs(e1)**2)*
            #                        sum(abs(e2)**2))
            Ns += cross(e1,e2)/((norm(e1)**2)*
                                 (norm(e2)**2)
                                 )
            
        return normalize(-Ns)
        #return Vector3D(0.0,0.0,0.0) # placeholder value



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


    ###################### END YOUR CODE


    # Set these newly-defined methods 
    # as the methods to use in the classes
    Face.normal = faceNormal
    Face.area = faceArea
    Vertex.normal = vertexNormal_AreaWeighted
    Vertex.vertexNormal_EquallyWeighted = vertexNormal_EquallyWeighted
    Vertex.vertexNormal_AreaWeighted = vertexNormal_AreaWeighted
    Vertex.vertexNormal_AngleWeighted = vertexNormal_AngleWeighted
    Vertex.vertexNormal_MeanCurvature = vertexNormal_MeanCurvature
    #
    Vertex.vertex_Laplace = vertex_Laplace
    #
    Vertex.vertexNormal_SphereInscribed = vertexNormal_SphereInscribed
    Vertex.angleDefect = angleDefect
    HalfEdge.cotan = cotan
    
    
    
    if show:
        ## Functions which will be called 
        #    by keypresses to visualize these definitions
    
        def toggleFaceVectors():
            print("\nToggling vertex vector display")
            if toggleFaceVectors.val:
                toggleFaceVectors.val = False
                meshDisplay.setVectors(None)
            else:
                toggleFaceVectors.val = True
                meshDisplay.setVectors('normal', vectorDefinedAt='face')
            meshDisplay.generateVectorData()
        toggleFaceVectors.val = False # ridiculous Python scoping hack
        meshDisplay.registerKeyCallback('1', 
                                        toggleFaceVectors, 
                                        docstring="Toggle drawing face normal vectors")
    
    
        def toggleVertexVectors():
            print("\nToggling vertex vector display")
            if toggleVertexVectors.val:
                toggleVertexVectors.val = False
                meshDisplay.setVectors(None)
            else:
                toggleVertexVectors.val = True
                meshDisplay.setVectors('normal', vectorDefinedAt='vertex')
            meshDisplay.generateVectorData()
        toggleVertexVectors.val = False # ridiculous Python scoping hack
        meshDisplay.registerKeyCallback('2', 
                                        toggleVertexVectors, 
                                        docstring="Toggle drawing vertex normal vectors")
    
    
        def toggleDefect():
            print("\nToggling angle defect display")
            if toggleDefect.val:
                toggleDefect.val = False
                meshDisplay.setShapeColorToDefault()
            else:
                toggleDefect.val = True
                meshDisplay.setShapeColorFromScalar("angleDefect", 
                                                    cmapName="seismic")
                                                   # vMinMax=[-pi/8,pi/8])
            meshDisplay.generateFaceData()
        toggleDefect.val = False
        meshDisplay.registerKeyCallback('3', 
                                        toggleDefect, 
                                        docstring="Toggle drawing angle defect coloring")
    
    
        def useEquallyWeightedNormals():
            mesh.staticGeometry = False
            print("\nUsing equally-weighted normals")
            Vertex.normal = vertexNormal_EquallyWeighted
            mesh.staticGeometry = True
            meshDisplay.generateAllMeshValues()
        meshDisplay.registerKeyCallback('4', 
                                        useEquallyWeightedNormals, 
                                        docstring="Use equally-weighted normal computation")
    
        def useAreaWeightedNormals():
            mesh.staticGeometry = False
            print("\nUsing area-weighted normals")
            Vertex.normal = vertexNormal_AreaWeighted
            mesh.staticGeometry = True
            meshDisplay.generateAllMeshValues()
        meshDisplay.registerKeyCallback('5', 
                                        useAreaWeightedNormals, 
                                        docstring="Use area-weighted normal computation")
    
        def useAngleWeightedNormals():
            mesh.staticGeometry = False
            print("\nUsing angle-weighted normals")
            Vertex.normal = vertexNormal_AngleWeighted
            mesh.staticGeometry = True
            meshDisplay.generateAllMeshValues()
        meshDisplay.registerKeyCallback('6', 
                                        useAngleWeightedNormals, docstring="Use angle-weighted normal computation")
    
    
        def useMeanCurvatureNormals():
            mesh.staticGeometry = False
            print("\nUsing mean curvature normals")
            Vertex.normal = vertexNormal_MeanCurvature
            mesh.staticGeometry = True
            meshDisplay.generateAllMeshValues()
        meshDisplay.registerKeyCallback('7', 
                                        useMeanCurvatureNormals, 
                                        docstring="Use mean curvature normal computation")
    
        def useSphereInscribedNormals():
            mesh.staticGeometry = False
            print("\nUsing sphere-inscribed normals")
            Vertex.normal = vertexNormal_SphereInscribed
            mesh.staticGeometry = True
            meshDisplay.generateAllMeshValues()
        meshDisplay.registerKeyCallback('8', 
                                        useSphereInscribedNormals, 
                                        docstring="Use sphere-inscribed normal computation")
    
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
    
        def deformShape():
            print("\nDeforming shape")
            mesh.staticGeometry = False
    
            # Get the center and scale of the shape
            center = meshDisplay.dataCenter
            scale = meshDisplay.scaleFactor
    
            # Rotate according to swirly function
            ax = eu.Vector3(-1.0,.75,0.5)
            for v in mesh.verts:
                vec = v.position - center
                theta = 0.8 * norm(vec) / scale
                newVec = np.array(eu.Vector3(*vec).rotate_around(ax, theta))
                v.position = center + newVec
    
    
            mesh.staticGeometry = True
            meshDisplay.generateAllMeshValues()
    
        meshDisplay.registerKeyCallback('x', 
                                        deformShape, 
                                        docstring="Apply a swirly deformation to the shape")
    
    
    
        ## Register pick functions that output useful information on click
        def pickVert(vert):
            print("   Position:" + printVec3(vert.position))
            print("   Angle defect: {:.5f}".format(vert.angleDefect))
            print("   Normal (equally weighted): " + printVec3(vert.vertexNormal_EquallyWeighted))
            print("   Normal (area weighted):    " + printVec3(vert.vertexNormal_AreaWeighted))
            print("   Normal (angle weighted):   " + printVec3(vert.vertexNormal_AngleWeighted))
            print("   Normal (sphere-inscribed): " + printVec3(vert.vertexNormal_SphereInscribed))
            print("   Normal (mean curvature):   " + printVec3(vert.vertexNormal_MeanCurvature))
        meshDisplay.pickVertexCallback = pickVert
    
        def pickFace(face):
            print("   Face area: {:.5f}".format(face.area))
            print("   Normal: " + printVec3(face.normal))
            print("   Vertex positions: ")
            for (i, vert) in enumerate(face.adjacentVerts()):
                print("     v{}: {}".format((i+1),printVec3(vert.position)))
        meshDisplay.pickFaceCallback = pickFace


    # Start the viewer running
    if show:
        meshDisplay.startMainLoop()
        
    return mesh


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
    # 
    mesh = main(
            inputfile='../meshes/bunny.obj',
            show=True,
            StaticGeometry=True)
    f1 = mesh.faces[0]
    he = mesh.halfEdges[0]
    ed = mesh.edges[0]

    v1 = mesh.verts[0]
    self = v1
    vl      = list(self.adjacentVerts())
    hl      = list(self.adjacentHalfEdges())
    fl      = list(self.adjacentFaces())
    
    
    d0 = mesh.buildExteriorDerivative0Form()
    d1 = mesh.buildExteriorDerivative1Form()
    h0 = mesh.buildHodgeStar0Form()
    h1 = mesh.buildHodgeStar1Form()
    h2 = mesh.buildHodgeStar2Form()
    bb = d1.dot(d0)
    print 'boundary of boundary representation: ',np.shape(bb)
    print 'any non-zero dd ? =>', np.min(bb),np.max(bb)
    
    """
    d0js = mesh.buildExteriorDerivative0FormJS()
    d1js = mesh.buildExteriorDerivative1FormJS()
    print d0js.todense() - d0.todense()
    print d1js.todense() - d1.todense()
    
    bb = d1js.dot(d0js)
    print 'boundary of boundary representation: ',np.shape(bb)
    print 'any non-zero dd ? =>', np.min(bb),np.max(bb)
    #"""
    """
    

    v1.vertexNormal_MeanCurvature
    v1.vertex_Laplace
    #Out[11]: array([ 1.73796675e-07,  1.15007117e-01, -1.38777878e-17])
    #Out[4]:  array([ 1.51118192e-06,  1.00000000e+00, -1.20668948e-16])
    
    v1.vertexNormal_SphereInscribed
    #Out[36]: array([ 5.63171872e-07,  1.00000000e+00, -6.08380079e-18])
    #Out[3]:  array([-5.63171872e-07, -1.00000000e+00,  6.08380079e-18])
    #Out[6]:  array([-5.63171872e-07, -1.00000000e+00, -0.00000000e+00])
    

    v1.vertexNormal_AreaWeighted
    #Out[12]: array([-1.59270406e-06, -1.00000000e+00,  0.00000000e+00])
    
    v1.vertexNormal_EquallyWeighted
    #Out[13]: array([-9.12346898e-07, -1.00000000e+00,  0.00000000e+00])
    
    v1.vertexNormal_AngleWeighted
    Out[8]: array([-1.30601294e-06, -1.00000000e+00,  0.00000000e+00])
    """
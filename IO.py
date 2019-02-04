#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 14:29:24 2018

@author: luke
"""

# Basic application to load a mesh from file and view it in a window

# Python imports
import sys, os

## Imports from this project
# hack to allow local imports without creaing a module or modifying the path variable
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core')) 
from InputOutput import *
from MeshDisplay import MeshDisplay

#from HalfEdgeMesh import *
from HalfEdgeMesh_ListImplementation import *


def main(inputfile,show):

    # Get the path for the mesh to load, either from the program argument if
    # one was given, or a dialog otherwise
    if(len(sys.argv) > 1):
        filename = sys.argv[1]
    elif inputfile is not None:
        filename = inputfile
    else:
        print("ERROR: No file name specified. Proper syntax is 'python testview.py path/to/your/mesh.obj'.")
        exit()

    # Read in the mesh
    mesh = HalfEdgeMesh(readMesh(filename),
                        staticGeometry=False)

    # Create a viewer window
    winName = 'meshview -- ' + os.path.basename(filename)
    meshDisplay = MeshDisplay(windowTitle=winName)
    meshDisplay.setMesh(mesh)
    if show:
        meshDisplay.startMainLoop()
    return mesh, meshDisplay

if __name__ == "__main__": 
    
    
    mesh, meshDisplay = main(
                            inputfile='../meshes/bunny.obj',
                            show=True)
    
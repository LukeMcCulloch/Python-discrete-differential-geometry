# Python discrete differential geometry

Discrete Differential Geometry Processing in Python

Notes:  The included "assignments" demonstrate some initial applications of discrete differential geometry and the exterior calculus to geometric processing tasks.  I basically set about doing the homework from Keenan Crane's discrete differential geometry course(s).  They also serve as an introduction to computational simulation of simple PDE's on a manifold.
This stuff resides in various places at Cal tech and Carnegie Mellon.  See:
[here](http://brickisland.net/cs177fa12/?p=272)
and 
[pdf discrete differential geometry book](https://www.cs.cmu.edu/~kmcrane/Projects/DDG/)
for starters.  

To run these little python demos, you will need methes, e.g. the Stanford bunny. 
For testing my code, it's easiest to pull my mesh repo and situate them together at the same level in a directory.
"My" meshes are [here](https://github.com/LukeMcCulloch/meshes.git)

[TODO: document in depth here.](https://python-discrete-differential-geometry.readthedocs.io/)


# Installation

Requirements beyond NumPy, SciPy, Matplotlib:

pip install euclid
pip install plyfile
pip install pyOpenGL

pyOpenGL GLUT issue:
    https://stackoverflow.com/questions/26700719/pyopengl-glutinit-nullfunctionerror

    Linux Users can just install glut using the following command:

    $ sudo apt-get install freeglut3-dev


Most critical, you need suitsparse for fast inversion of sparse matrices 
in order to take advantage of some of the solver forumulations:

get an up to date suitsparse [here](https://pythonhosted.org/scikit-sparse/overview.html)
    
    Installing scikit-sparse requires:

    Python
    NumPy
    SciPy
    Cython
    CHOLMOD

On Debian/Ubuntu systems, the following command should suffice:

    $ apt-get install python-scipy libsuitesparse-dev


    Then just:

    $ pip install --user scikit-sparse

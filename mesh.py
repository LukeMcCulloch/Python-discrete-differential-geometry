#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 22:09:36 2018

@author: luke

https://TLMbot@bitbucket.org/TLMbot/pydg
"""

class HalfEdge(object):
    def __init__(self, 
                 next_,
                 flip,
                 vertex,
                 edge,
                 face):
        self.next = next_
        self.flip = flip
        self.vertex = vertex
        self.edge = edge
        self.face = face
# -*- coding: utf-8 -*-
from vectorizers import *
# Tests for vectors

def test_build_bin_vect():   
    s = '(S (NP (DT the) (NN cat)) (VP (VBD ate) (NP (DT a) (NN cookie))))'
    t = Tree.fromstring(s)  
    d  = build_bin_vect(t)    
    assert({'NN': 1, 'VBD': 1, 'VP': 1, 'S': 1, 'NP': 1, 'DT': 1} ==d)
    print 'Test of collecting relations went ok \n'
    

def test_build_count_vect():
    s = '(S (NP (DT the) (NN cat)) (VP (VBD ate) (NP (DT a) (NN cookie))))'
    t = Tree.fromstring(s)
    d  = build_count_vect(t)       
    assert({'NN': 2, 'VBD': 1, 'VP': 1, 'S': 1, 'NP': 2, 'DT': 2}==d)
    print 'Test of counting went ok \n'

def test_build_norm_vect():
    s = '(S (NP (DT the) (NN cat)) (VP (VBD ate) (NP (DT a) (NN cookie))))'
    t = Tree.fromstring(s)  
    d  = build_norm_vect(t)  
    assert(sum(d.values()) == 1)
    print 'Test of normalization went ok with \n: ',d,'\n'
#
def test_build_height_vect():
    s = '(S (NP (DT the) (NN cat)) (VP (VBD ate) (NP (DT a) (NN cookie))))'
    t = Tree.fromstring(s)  
    print "Test of mean height :\n", build_height_vect(t)

# Testing the precess of building the different vectors  
test_build_bin_vect()  
test_build_count_vect()
test_build_norm_vect()
test_build_height_vect()
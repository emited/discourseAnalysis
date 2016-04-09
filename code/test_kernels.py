from kernels import *
from vectorizers import *
# -*- coding: utf-8 -*-

# Tests for kernels
def test_kernel_collections():
    s1 = '(S (NP (DT the) (NN cat)) (VP (VBD ate) (NP (DT a) (NN0 cookie))))'
    t1 = Tree.fromstring(s1)  
    
    s2 = '(S2 (NP (DT the) (NN2 cat)) (VP (VBD ate) (NP2 (DT a) (NN cookie))))'
    t2 = Tree.fromstring(s2)

    v1,v2 = build_bin_vect(t1),build_bin_vect(t2)    
    
    d = vector_kernel(v1,v2)
    print "test_kernel_collections : ",d
    
def test_kernel_count():
    s1 = '(S (NP (DT the) (NN cat)) (VP (VBD ate) (NP (DT a) (NN0 cookie))))'
    t1 = Tree.fromstring(s1)  
    
    s2 = '(S2 (NP (DT the) (NN2 cat)) (VP (VBD ate) (NP2 (DT a) (NN cookie))))'
    t2 = Tree.fromstring(s2)
    
    v1,v2 = build_count_vect(t1),build_count_vect(t2)
    
    d = vector_kernel(v1,v2) 
    print "test_kernel_count :",d

def test_kernel_normalized_vectors():
    s1 = '(S (NP (DT the) (NN cat)) (VP (VBD ate) (NP (DT a) (NN0 cookie))))'
    t1 = Tree.fromstring(s1)  
    
    s2 = '(S2 (NP (DT the) (NN2 cat)) (VP (VBD ate) (NP2 (DT a) (NN cookie))))'
    t2 = Tree.fromstring(s2)
    
    v1,v2 = build_norm_vect(t1),build_norm_vect(t2)
   
    d = vector_kernel(v1,v2) 
    print "Test_kernel_normalized_vectors : ",d

def test_kernel_on_positions():
    s1 = '(S (NP (DT the) (NN cat)) (VP (VBD ate) (NP (DT a) (NN0 cookie))))'
    t1 = Tree.fromstring(s1)  
    
    s2 = '(S2 (NP (DT the) (NN2 cat)) (VP (VBD ate) (NP2 (DT a) (NN cookie))))'
    t2 = Tree.fromstring(s2)
    
    v1,v2 = build_mean_height_vect(t1),build_mean_height_vect(t2)
    p =  vector_kernel(v1,v2) 
    print "test_kernel_count :",p

# Testing the precess of building the different vectors  

test_kernel_collections()
test_kernel_count()
test_kernel_normalized_vectors()
test_kernel_on_positions()

# To load the following test on real data ,copying all the treekernel (.py and _test.py)
# in main directory with analyse.py
# Then uncomment and test

#=> Uncomment to test the following one
from analyse import *

def test_on_data_base():
    t = return_trees_from_merge('./data')

    v1 = build_bin_vect(t[0][0])
    v2 = build_bin_vect(t[1][0])
    
    v3 = build_count_vect(t[0][0])
    v4 = build_count_vect(t[1][0])
    
    v5 = build_norm_vect(t[0][0])
    v6 = build_norm_vect(t[1][0])
    
    v7 = build_mean_height_vect(t[0][0])
    v8 = build_mean_height_vect(t[1][0])
    
    print "Dictionnaries' len : ",len(v1),len(v2)
    print "Computing distances using "
    print " - bin collection : ",vector_kernel(v1,v2)
    print " - counting : ",vector_kernel(v3,v4)
    print " - normalized counting : ",vector_kernel(v5,v6)
    #print " - original TreeKernel : ",TreeKernel(t[0][0],t[1][0])
    print "RQ :"    
    print "=>Counting occurences seems to be expressive enough"
    print "=>TreeKernel is a bit too high value and slow since both texts have ~350 words"
    print " - mean position: ",vector_kernel(v7,v8) 
#=> Uncomment to test the following one
test_on_data_base()


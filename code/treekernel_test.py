# -*- coding: utf-8 -*-
from treekernel import *

# Tests for vectors

def test_build_relations_vector():   
    s = '(S (NP (DT the) (NN cat)) (VP (VBD ate) (NP (DT a) (NN cookie))))'
    t = Tree.fromstring(s)  
    d  = build_relations_vector(t)
    
    assert(['S', 'NP', 'DT', 'NN', 'VP', 'VBD'] ==d)
    print 'Test of collecting relations went ok \n'
    

def test_build_relations_count_vector():
    s = '(S (NP (DT the) (NN cat)) (VP (VBD ate) (NP (DT a) (NN cookie))))'
    t = Tree.fromstring(s)  
    d  = build_relations_count_vector(t)       
    assert({'NN': 2, 'VBD': 1, 'VP': 1, 'S': 1, 'NP': 2, 'DT': 2}==d)
    print 'Test of counting went ok \n'

def test_build_relations_normalized_vector():
    s = '(S (NP (DT the) (NN cat)) (VP (VBD ate) (NP (DT a) (NN cookie))))'
    t = Tree.fromstring(s)  
    d  = build_relations_normalized_vector(t)  
    assert(sum(d.values()) == 1)
    print 'Test of normalization went ok with \n: ',d,'\n'

# Tests for kernels
def test_kernel_collections():
    s = '(S (NP (DT the) (NN cat)) (VP (VBD ate) (NP (DT a) (NN0 cookie))))'
    t = Tree.fromstring(s)  
    
    s2 = '(S2 (NP (DT the) (NN2 cat)) (VP (VBD ate) (NP2 (DT a) (NN cookie))))'
    t2 = Tree.fromstring(s2)
    
    d = kernel_on_relations_collection(t,t2)
    print "test_kernel_collections : ",d
    
def test_kernel_count():
    s = '(S (NP (DT the) (NN cat)) (VP (VBD ate) (NP (DT a) (NN0 cookie))))'
    t = Tree.fromstring(s)  
    
    s2 = '(S2 (NP (DT the) (NN2 cat)) (VP (VBD ate) (NP2 (DT a) (NN cookie))))'
    t2 = Tree.fromstring(s2)
    
    d = kernel_on_relations_count(t,t2)
    print "test_kernel_count :",d

def test_kernel_normalized_vectors():
    s = '(S (NP (DT the) (NN cat)) (VP (VBD ate) (NP (DT a) (NN0 cookie))))'
    t = Tree.fromstring(s)  
    
    s2 = '(S2 (NP (DT the) (NN2 cat)) (VP (VBD ate) (NP2 (DT a) (NN cookie))))'
    t2 = Tree.fromstring(s2)
    
    d = kernel_on_normalized_counting(t,t2)
    print "Test_kernel_normalized_vectors : ",d



# Testing the precess of building the different vectors  
test_build_relations_vector()  
test_build_relations_count_vector()
test_build_relations_normalized_vector()
test_kernel_collections()
test_kernel_count()
test_kernel_normalized_vectors()

# To load the following test on real data ,copying all the treekernel (.py and _test.py)
# in main directory with analyse.py
# Then uncomment and test

#=> Uncomment to test the following one
#from analyse import *

def test_on_data_base():
    t = return_trees_from_merge('./data')

    v1 = build_relations_vector(t[0][0])
    v2 = build_relations_vector(t[1][0])
    
    print "Dictionnaries' len : ",len(v1),len(v2)
    print "distances using collection, counting, normalized counting :"
    print kernel_on_relations_collection(t[0][0],t[1][0])
    print kernel_on_relations_count(t[0][0],t[1][0])
    print kernel_on_normalized_counting(t[0][0],t[1][0])
#=> Uncomment to test the following one
#test_on_data_base()

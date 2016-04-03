from nltk.tree import Tree
import re
import numpy as np

def clean_tree(tree):
    """Returns a tree without any digits"""
    strtree=str(tree)
    match=re.findall(r'[^0-9]+',str(strtree))
    strtree=''.join(match)
    return Tree.fromstring(strtree)


def TreeKernel(T1,T2,const=1,rho=1):
    '''returns the number of common subset tree if rho=1 and common 
        subtrees if rho=0.
        const balances the contribution of subtrees: small values
        decay the contribution of lower nodes in large subtrees.'''
    K=0
    for t1 in T1.subtrees():
        for t2 in T2.subtrees():
            K+=delta(t1,t2,const,rho)
    return K
    
def same_root(T1,T2):
    '''returns true only if the label of the root nodes are the same.'''
    return T1.label()==T2.label()


def pre(T):
    '''returns true only if root of T is a preterminal node.'''
    return T.height()<=2


def delta(T1,T2,const=1,rho=1):
    '''returns the number of common subset trees if rho=1 and common 
        subtrees if rho=0 containing their root.
        const balances the contribution of subtrees: small values
        decay the contribution of lower nodes in large subtrees.'''
    if not same_root(T1,T2):
        return 0
    if(pre(T1) and pre(T2) and T1==T2):
        return const
    if(not(pre(T1)) and not(pre(T2)) and same_root(T1,T2)):
        return const*(rho+delta(T1[0],T2[0]))*(rho+delta(T1[1],T2[1]))
    return 1

def build_relations_vector(T):
    dico = []
    for s in T.subtrees(lambda T: T.label() != "EDU"):
        if s.label() not in dico:
            dico.append(s.label())
    '''returns the list of RST-relations contained in the tree as a vector'''
    return dico
    
def build_relations_count_vector(T):
    dico = {}
    for s in T.subtrees(lambda T: T.label() != "EDU"):
        l = s.label()
        if l not in dico.keys():           
            dico[l] = 1
        else:
            dico[l] += 1
    '''returns a dictionnary as a vector of relations with associated frequency'''
    return dico

    
def build_relations_normalized_vector(T):
    dico = build_relations_count_vector(T)
    n = sum(dico.values())
    for k in dico.keys():
        dico[k]/=float(n)
    '''returns a dictionnary as a vector of appearing relations with associated frequency;
    normalized by total number of occurence'''            
    return dico

def kernel_on_relations_collection(T1,T2):
    v1 = build_relations_vector(T1)
    v2 = build_relations_vector(T2)
    # fusion des dicos  : emsemble des relations sur deux textes
    merge = set(v1+v2)
    d=0    
     
    for k in merge:
        if (k not in v1) or (k not in v2):
            d+=1
    ''' returns a measure of the distance(how far) between the two representations of trees
    based on relations collection'''
    return np.sqrt(d) # Sqrt pour mettre pouvoir comparer aux autres kernels !

def kernel_on_relations_count(T1,T2):
    v1 = build_relations_count_vector(T1)
    v2 = build_relations_count_vector(T2)
    
    merge = dict(v1.items()+v2.items())
     
    d=0    
    for k in merge:
        d+=(v1.get(k,0) - v2.get(k,0))**2
    ''' returns a measure of the distance(how far) between the two representations of trees
    based on relations counting'''
    return np.sqrt(d)
    
def kernel_on_normalized_counting(T1,T2):
    v1 = build_relations_normalized_vector(T1)
    v2 = build_relations_normalized_vector(T2)
    
    merge = dict(v1.items()+v2.items())
   
    d=0    
    for k in merge:
        d+=(v1.get(k,0) - v2.get(k,0))**2
    ''' returns a measure of the distance(how far) between the two representations of trees
    based on relations normalized frequency'''
    return np.sqrt(d)

#### En tenant compte des positions des relations

def build_relations_count_and_positions_vector(T):
    count = {} # Compteur d'occurence pour chaque relation
    pos = {} # Enregistement des positions de chaque occurence, avant d'en extraire la moyenne
    for s in T.subtrees(lambda T: T.label() != "EDU"):
        l = s.label()
        if l not in count.keys():           
            count[l] = 1
            pos[l]=[s.height()]
        else:
            count[l] += 1
            pos[l].append(s.height())
            
    for k in pos.keys():
        pos[k]= np.floor(np.mean(pos[k])+1) # partie entiere sup de la moyenne des hauteur
        
    '''returns 2 dictionnaries as a vector of relations :frequencies and mean positions of each relation'''
    return (count,pos)
    
def kernel_on_count_and_positions(T1,T2):
    
    v1_count,v1_pos = build_relations_count_and_positions_vector(T1)
    v2_count,v2_pos = build_relations_count_and_positions_vector(T2)
    
    merged_labels = dict(v1_count.items()+v2_count.items())
    
    
    d_count=0
    d_pos=0    
    
    for k in merged_labels:
        d_count+=(v1_count.get(k,0) - v2_count.get(k,0))**2
        d_pos+=(v1_pos.get(k,0) - v2_pos.get(k,0))**2
        
    ''' returns a measure of the distance(how far) between the two representations of trees
    based on relations normalized frequency'''
    return np.sqrt(d_count),np.sqrt(d_pos)    
    

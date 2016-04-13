from nltk.tree import Tree
import numpy as np

def build_bin_vect(T):
    dico = {}
    for s in T.subtrees(lambda T: T.label() != "EDU"):
        l = s.label()
        if l not in dico.keys():           
            dico[l] = 1
    '''returns the list of RST-relations contained in the tree as a vector'''
    return dico

def build_bin_vects(T_list):
    return [build_bin_vect(T) for T in T_list]
    
def build_count_vect(T):
    dico = {}
    for s in T.subtrees(lambda T: T.label() != "EDU"):
        l = s.label()
        if l not in dico.keys():           
            dico[l] = 1
        else:
            dico[l] += 1
    '''returns a dictionnary as a vector of relations with associated frequency'''
    return dico

def build_count_vects(T_list):
    return [build_count_vect(T) for T in T_list]
    
def build_norm_vect(T):
    dico = build_count_vect(T)
    n = sum(dico.values())
    for k in dico.keys():
        dico[k]/=float(n)
    '''returns a dictionnary as a vector of appearing relations with associated frequency;
    normalized by total number of occurence'''            
    return dico

def build_norm_vects(T_list):
    return [build_norm_vect(T) for T in T_list]
#### En tenant compte des positions des relations

def build_height_vect(T):
    pos = {} # Enregistement des positions de chaque occurence, avant d'en extraire la moyenne
    for s in T.subtrees(lambda T: T.label() != "EDU"):
        l = s.label()
        if l not in pos.keys():
            pos[l]=s.height()
        else:
            pos[l]+=s.height()
    '''returns dictionnary as a vector of relations :mean height of each relation'''
    return pos

def build_height_vects(T_list):
    return [build_height_vect(T) for T in T_list]

def build_tfid_vects(T_list):
    dicos = build_norm_vects(T_list)
    N = float(len(T_list))
    #term document count
    term_count = {}
    for d in dicos:
        for k in d.keys():
            if k not in term_count.keys():           
                term_count[k] = 1.
            else:
                term_count[k] += 1.
    for i in range(len(dicos)):
        for k in dicos[i].keys():
            dicos[i][k] = dicos[i][k]*(N/term_count[k])
    return dicos

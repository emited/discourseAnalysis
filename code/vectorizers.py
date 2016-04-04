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

    
def build_norm_vect(T):
    dico = build_count_vect(T)
    n = sum(dico.values())
    for k in dico.keys():
        dico[k]/=float(n)
    '''returns a dictionnary as a vector of appearing relations with associated frequency;
    normalized by total number of occurence'''            
    return dico
#### En tenant compte des positions des relations

def build_mean_height_vect(T):
    pos = {} # Enregistement des positions de chaque occurence, avant d'en extraire la moyenne
    for s in T.subtrees(lambda T: T.label() != "EDU"):
        l = s.label()
        if l not in pos.keys():           
            pos[l]=[s.height()]
        else:
            pos[l].append(s.height())
            
    for k in pos.keys():
        pos[k]= np.mean(pos[k]) #moyenne des hauteur
        
    '''returns dictionnary as a vector of relations :mean height of each relation'''
    return pos
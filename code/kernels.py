from nltk.tree import Tree
import re
import numpy as np

def vector_kernel(v1,v2):

    merge = dict(v1.items()+v2.items())
     
    d=0    
    for k in merge:
        d+=(v1.get(k,0) - v2.get(k,0))**2
    ''' returns a measure of the distance(how far) between the two representations of trees
    based on relations counting'''
    return np.sqrt(d)

def rbf_kernel(v1,v2,sigma=1.):
    dist = vector_kernel(v1,v2)
    return np.exp(-dist**2/(sigma*2))


def clean_tree(tree):
    """Returns a tree without any digits"""
    strtree=str(tree)
    match=re.findall(r'[^0-9]+',str(strtree))
    strtree=''.join(match)
    return Tree.fromstring(strtree)
    
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



def tree_kernel(T1,T2,const=1,rho=1):
    '''returns the number of common subset tree if rho=1 and common 
        subtrees if rho=0.
        const balances the contribution of subtrees: small values
        decay the contribution of lower nodes in large subtrees.'''
    K=0
    for t1 in T1_p.subtrees():
        for t2 in T2_p.subtrees():
            K+=delta(t1,t2,const,rho)
    return K

def compute_gram(X,Y,kernel):
	"""computes a gram matrix K with matrices X and Y 
	such as K[i,j] = kernel(X[i],Y[j]).
	"""
	K = np.zeros((X.shape[0],Y.shape[0]))
	for i,x in enumerate(X):
		for j,y in enumerate(Y):
			K[i, j] = kernel(x,y)
	return K
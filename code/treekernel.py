from nltk.tree import Tree

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


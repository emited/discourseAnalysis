from nltk.tree import Tree
import numpy as np

def g_aux(labels,depth,s):
    if(depth==0):
        return ''
    if(np.random.rand(1)[0]<0.6):
        label1,label2=np.random.choice(labels,2)
        str1='( '+label1+' '+g_aux(labels,depth-1,s)+' )'
        str2='( '+label2+' '+g_aux(labels,depth-1,s)+' )'
        return str1+str2
    #if(np.random.rand(1)[0]<0.5):
    #    label=np.random.choice(labels)
    #    return '('+label+' '+g_aux(labels,depth-1,s)+' )'
    if(np.random.rand(1)[0]<0.6):
        return g_aux(labels,depth-1,s)
    else:
        return ''
    
def gen_rand(labels=['A','B'],maxdepth=4):
    '''generates random binary trees, with nodes with associated labels.
    '''
    label=np.random.choice(labels)
    return Tree.fromstring('( '+label+' '+g_aux(labels,maxdepth,'')+' )')

def gen_rand_list(ntrees=10,labels=['A','B'],maxdepth=4):
    '''generates a list of random binary trees, with min_nodes as a minimal amount of nodes per tree
    '''
    k=0
    list_tree=[]
    min_nodes=3
    while(k<ntrees):
        t=gen_rand(labels,maxdepth)        
        if(len(str(t))>=4*min_nodes): #test if tree has more that min_nodes
            list_tree.append(t)
            k+=1
    return list_tree

def prune_aux(tree,cut,i):
    if(type(tree)==str and cut>i):
        return '('+tree+')'
    try:
        if(cut>i):
            return '('+tree.label()+'\n\t'+prune_aux(tree[0],cut,i+1)+prune_aux(tree[1],cut,i+1)+')'
        else:
            return ''
    except:
        return '('+tree.label()+'\n\t'+prune_aux(tree[0],cut,i+1)+')'

def prune(tree,max_height=10):
    '''returns a tree with branches cut when their depth is superior to max_height.'''
    s = prune_aux(tree,max_height,0)
    return Tree.fromstring(s)
import nltk.Tree
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
    
def gen_rand_tree(labels=['A','B'],maxdepth=4):
    label=np.random.choice(labels)
    return Tree.fromstring('( '+label+' '+g_aux(labels,maxdepth,'')+' )')

def gen_rand_tree_list(ntrees=10,labels=['A','B'],maxdepth=4):
    k=0
    list_tree=[]
    min_nodes=3
    while(k<ntrees):
        t=gen_rand_tree(labels,maxdepth)        
        if(len(str(t))>=4*min_nodes): #test if tree has more that min_nodes
            list_tree.append(t)
            k+=1
    return list_tree

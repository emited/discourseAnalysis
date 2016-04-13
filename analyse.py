from DPLP.code.model import ParsingModel
from DPLP.code.tree import RSTTree
from DPLP.code.docreader import DocReader
import numpy as np
from os import listdir
from os.path import join as joinpath
import code.kernels as kernels
import code.vectorizers as vectorizers
from sklearn import feature_extraction
from sklearn.metrics import pairwise
import pandas as pd
import cPickle as pickle
from nltk.tree import Tree
import code.tree as ctree
# Fichier a lancer depuis DPLP 


# Supprimer params inutiles
def return_trees_from_merge(path, report=False, 
               bcvocab=None, withdp=False, fdpvocab=None, fprojmat=None):

    """ Test the parsing performance
    :type path: string
    :param path: path to the evaluation data
    :type report: boolean
    :param report: whether to report (calculate) the f1 score
    """
    # ----------------------------------------
    # Load the parsing model
    
    print 'Load parsing model ...'
    pm = ParsingModel(withdp=withdp,
        fdpvocab=fdpvocab, fprojmat=fprojmat)
    pm.loadmodel("./DPLP/model/parsing-model.pickle.gz")

    # ----------------------------------------
    # Read all files from the given path
    doclist = [joinpath(path, fname) for fname in listdir(path) if fname.endswith('.merge')]
    trees_list =[]

    for fmerge in doclist:
        # recuperation du nom : ici id et classe
        tree_id = fmerge #.split('.')[0]
        # ----------------------------------------
        # Read *.merge file
        dr = DocReader()
        doc = dr.read(fmerge)
        # ----------------------------------------
        # Parsing
        pred_rst = pm.sr_parse(doc, bcvocab)
        strtree = pred_rst.parse()

        trees_list.append((Tree.fromstring(strtree),tree_id))
    
    # retoure liste des (tree, tree_id)     
    return trees_list

# CSV individuels pour le moment 
def write_tree_in_csv(list_tree_tree_id):
    
    for (tree,tree_id) in list_tree_tree_id:
        f = open(tree_id+'.csv','w')
        f.write(str(tree)) # python will convert \n to os.linesep
        f.write('\n')        
        f.close() # you can omit in most cases as the destructor will call it

def read_trees_from_csv(path):
        # Read all files from the given path
    doclist = [joinpath(path, fname) for fname in listdir(path) if fname.endswith('.csv')]
    trees_list =[]
    
    for f in doclist:
        doc = open(f, 'r')
        content = doc.read()
        tree = Tree.fromstring(content)
        trees_list.append(tree)        
    return trees_list
    
 # test d'egalite entre tree lu et tree d'origine
def test_ecriture_lecture():

    print "Beginning test"
    
    t = return_trees_from_merge('./data')
    write_tree_in_csv(t)
    l = read_trees_from_csv('./data')
    for (ti,li) in zip(t,l):
        assert(ti[0].__eq__(li))
        print "Un arbre teste"
    print "Test done for all trees : it's alright"

def build_all_2():
    print 'For each class, we build all the trees and save them in CSVs'
    path_to_save = '~/Documents/s2/tal/discourseAnalysis/data/'
    nar_trees = return_trees_from_merge('~/Documents/s2/tal/discourseAnalysis/data/narrative')
    write_tree_in_csv(nar_trees)    
    
    arg_trees = return_trees_from_merge('~/Documents/s2/tal/discourseAnalysis/data/argumentative')
    write_tree_in_csv(arg_trees) 
     
    inf_trees = return_trees_from_merge('~/Documents/s2/tal/discourseAnalysis/data/informative')
    write_tree_in_csv(inf_trees) 
    
    des_trees = []

    # Attention, contient couples de (trees + tree_ID) ou tree_ID est le nom du fichier.
    all_trees = nar_trees + arg_trees + inf_trees + des_trees
    int2cl = {0:'narrative', 1:'argumentative', 2:'informative',3:'descriptive'}

    T = [t[0] for t in all_trees]
    pickle.dump(T,open(path_to_save+'trees.pkl','wb'))

    y_nar = [0 for t in nar_trees]
    y_arg = [1 for t in arg_trees]
    y_inf = [2 for t in inf_trees]
    y_des = [3 for t in des_trees]
    y = np.array( y_nar + y_arg + y_inf + y_des )
    pickle.dump(y,open(path_to_save+'labels.pkl','wb'))
    
    index = ['bin','count','norm','height','tfid']

    print 'Dicts'
    D_bin = vectorizers.build_bin_vects(T)
    D_count = vectorizers.build_count_vects(T)
    D_norm = vectorizers.build_norm_vects(T)
    D_height = vectorizers.build_height_vects(T)
    D_tfid = vectorizers.build_tfid_vects(T)
    
    D_all = {'bin':D_bin ,'count': D_count,'norm': D_norm,'height': D_height,'tfid': D_tfid}
    pickle.dump(D_all,open(path_to_save+'dicts.pkl','wb'))
    

    print 'Vects'
    vectorizer = feature_extraction.DictVectorizer(sparse=False)
    V_bin = vectorizer.fit_transform(D_bin)
    V_count = vectorizer.fit_transform(D_count)
    V_norm = vectorizer.fit_transform(D_norm)
    V_height = vectorizer.fit_transform(D_height)
    V_tfid = vectorizer.fit_transform(D_tfid)

    V_all = {'bin':V_bin ,'count': V_count,'norm': V_norm,'height': V_height,'tfid': V_tfid}
    pickle.dump(V_all,open(path_to_save+'vects.pkl','wb'))
    
    #Y = vectorizer.inverse_transform(V_bin)



    print 'Kernels'

    ## tree kernels
    max_depth = 15
    T_p = [ctree.prune(t,max_depth) for t in T]
    K_tree = kernels.compute_gram(T_p,T_p,kernels.tree_kernel)
    pickle.dump(K_tree,open(path_to_save+'tree_kernel.pkl'))

    print 'vector kernels'
    print 'linear'
    K_bin_lin = pairwise.linear_kernel(V_bin)
    K_count_lin = pairwise.linear_kernel(V_count)
    K_norm_lin = pairwise.linear_kernel(V_norm)
    K_height_lin = pairwise.linear_kernel(V_height)
    K_tfid_lin = pairwise.linear_kernel(V_tfid)
    K_all_lin = {'bin':K_bin_lin, 'count':K_count_lin, 'norm':K_norm_lin, 'height':K_height_lin, 'tfid':K_tfid_lin}
    print 'rbf'
    K_bin_rbf = pairwise.rbf_kernel(V_bin)
    K_count_rbf = pairwise.rbf_kernel(V_count)
    K_norm_rbf = pairwise.rbf_kernel(V_norm)
    K_height_rbf = pairwise.rbf_kernel(V_height)
    K_tfid_rbf = pairwise.rbf_kernel(V_tfid)
    K_all_rbf = {'bin':K_bin_rbf, 'count':K_count_rbf, 'norm':K_norm_rbf, 'height':K_height_rbf, 'tfid':K_tfid_rbf}
    print 'cosine sim'
    K_bin_cos_sim = pairwise.cosine_similarity(V_bin)
    K_count_cos_sim = pairwise.cosine_similarity(V_count)
    K_norm_cos_sim = pairwise.cosine_similarity(V_norm)
    K_height_cos_sim = pairwise.cosine_similarity(V_height)
    K_tfid_cos_sim = pairwise.cosine_similarity(V_tfid)
    K_all_cos_sim = {'bin':K_bin_cos_sim, 'count':K_count_cos_sim, 'norm':K_norm_cos_sim, 'height':K_height_cos_sim, 'tfid':K_tfid_cos_sim}
    print 'euclidean distance'
    K_bin_eucl_dist = pairwise.pairwise_distances(V_bin,metric='euclidean')
    K_count_eucl_dist = pairwise.pairwise_distances(V_count,metric='euclidean')
    K_norm_eucl_dist = pairwise.pairwise_distances(V_norm,metric='euclidean')
    K_height_eucl_dist = pairwise.pairwise_distances(V_height,metric='euclidean')
    K_tfid_eucl_dist = pairwise.pairwise_distances(V_tfid,metric='euclidean')
    K_all_eucl_dist = {'bin':K_bin_eucl_dist, 'count':K_count_eucl_dist, 'norm':K_norm_eucl_dist, 'height':K_height_eucl_dist, 'tfid':K_tfid_eucl_dist}
    print 'minkowski distance'
    K_bin_mink_dist = pairwise.pairwise_distances(V_bin,metric='minkowski')
    K_count_mink_dist = pairwise.pairwise_distances(V_count,metric='minkowski')
    K_norm_mink_dist = pairwise.pairwise_distances(V_norm,metric='minkowski')
    K_height_mink_dist = pairwise.pairwise_distances(V_height,metric='minkowski')
    K_tfid_mink_dist = pairwise.pairwise_distances(V_tfid,metric='minkowski')
    K_all_mink_dist = {'bin':K_bin_mink_dist, 'count':K_count_mink_dist, 'norm':K_norm_mink_dist, 'height':K_height_mink_dist, 'tfid':K_tfid_mink_dist}


    K_all = {'lin':K_all_lin, 'rbf':K_all_rbf, 'cos_sim':K_all_cos_sim,'eucl_dist':K_all_eucl_dist,'mink_dist':K_all_mink_dist}
    pickle.dump(K_all,open(path_to_save+'vect_kernels.pkl','wb'))
    print "done"

def build_all():
    # For each class, we build all the trees and save them in CSVs
    nar_trees = return_trees_from_merge('~/Documents/s2/tal/discourseAnalysis/data/narrative')
    write_tree_in_csv(nar_trees)    
    
    arg_trees = return_trees_from_merge('~/Documents/s2/tal/discourseAnalysis/data/argumentative/')
    write_tree_in_csv(arg_trees) 
     
    inf_trees = return_trees_from_merge('~/Documents/s2/tal/discourseAnalysis/data/informative/')
    write_tree_in_csv(inf_trees) 
    
    des_trees = []
    #des_trees = return_trees_from_merge('~/Documents/s2/tal/discourseAnalysis/data/informative/')
    #write_tree_in_csv(des_trees) 
    
    
    # Attention, contient couples de (trees + tree_ID) ou tree_ID est le nom du fichier.
    all_trees = nar_trees + arg_trees + inf_trees + des_trees
    int2cl = {0:'narrative', 1:'argumentative', 2:'informative',3:'descriptive'}

    path_to_save = '~/Documents/s2/tal/discourseAnalysis/data/'
    y_nar = [0 for t in nar_trees]
    y_arg = [1 for t in arg_trees]
    y_inf = [2 for t in inf_trees]
    y_des = [3 for t in des_trees]
    y = np.array( y_nar + y_arg + y_inf + y_des )
    pickle.dump(y,open(path_to_save+'labels_test.pkl','wb'))

    T = [t[0] for t in all_trees]
    pickle.dump(T,open(path_to_save+'trees_test.pkl','wb'))
    
    index = ['bin','count','norm','height','tfid']

    #Dicts
    D_bin = vectorizers.build_bin_vects(T)
    D_count = vectorizers.build_count_vects(T)
    D_norm = vectorizers.build_norm_vects(T)
    D_height = vectorizers.build_height_vects(T)
    D_tfid = vectorizers.build_tfid_vects(T)
    
    D_df = pd.DataFrame([D_bin,D_count,D_norm,D_height,D_tfid],index=index)
    D_df = D_df.transpose()
    D_df.to_pickle(path_to_save+'dicts_test.pkl')
    

    #Vects
    vectorizer = feature_extraction.DictVectorizer(sparse=False)
    V_bin = vectorizer.fit_transform(D_bin)
    V_count = vectorizer.fit_transform(D_count)
    V_norm = vectorizer.fit_transform(D_norm)
    V_height = vectorizer.fit_transform(D_height)
    V_tfid = vectorizer.fit_transform(D_tfid)

    V_all = np.zeros((len(index),V_bin.shape[0],V_bin.shape[1]))
    V_all = np.array([V_bin,V_count,V_norm,V_height,V_tfid])
    V_df = []
    for i in range(V_all.shape[1]):
        d = {}
        for j,v in enumerate(V_all[:,i]):
            d[index[j]]=v
        V_df.append(d)
    V_df = pd.DataFrame(V_df)
    V_df.to_pickle(path_to_save+'vects_test.pkl')
    
    #euclidean distance
    K_bin_eucl_dist = pairwise.pairwise_distances(V_bin,metric='euclidean')
    K_count_eucl_dist = pairwise.pairwise_distances(V_count,metric='euclidean')
    K_norm_eucl_dist = pairwise.pairwise_distances(V_norm,metric='euclidean')
    K_height_eucl_dist = pairwise.pairwise_distances(V_height,metric='euclidean')
    K_tfid_eucl_dist = pairwise.pairwise_distances(V_tfid,metric='euclidean')
    K_all_eucl_dist = [K_bin_eucl_dist, K_count_eucl_dist, K_norm_eucl_dist, K_height_eucl_dist, K_tfid_eucl_dist]
    
    K_all = {'eucl_dist':K_all_eucl_dist}
    pickle.dump(K_all,open(path_to_save+'kernels_test.pkl','wb'))

build_all_2()



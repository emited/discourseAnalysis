from DPLP.code.model import ParsingModel
from DPLP.code.tree import RSTTree
from DPLP.code.docreader import DocReader
from os import listdir
from os.path import join as joinpath

# Fichier a lancer depuis DPLP 
from nltk.tree import Tree

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
        tree_id = fmerge.split('.')[0]
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

test_ecriture_lecture()

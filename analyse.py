from DPLP.code.model import ParsingModel
from DPLP.code.tree import RSTTree
from DPLP.code.docreader import DocReader
from os import listdir
from os.path import join as joinpath


from nltk.tree import Tree

def return_trees(path='data', report=False, 
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

        # ----------------------------------------
        # Read *.merge file
        dr = DocReader()
        doc = dr.read(fmerge)
        # ----------------------------------------
        # Parsing
        pred_rst = pm.sr_parse(doc, bcvocab)
        strtree = pred_rst.parse()
        trees_list.append(Tree.fromstring(strtree))
        
    return trees_list

print return_trees()

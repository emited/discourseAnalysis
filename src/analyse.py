from nltk.tree import Tree


f = open('../data/7oldsamr.txt.brackets','r')

s = ""
for line in f.readlines():
    print line
    s += line
f.close()



from nltk.tree import Tree
tree = Tree.fromstring('( NN-textualorganization ( EDU 1 )  ( SN-purpose ( NS-elaboration ( EDU 2 )  ( NS-temporal ( EDU 3 )  ( EDU 4 )  )  )  ( NS-elaboration ( NS-elaboration ( EDU 5 )  ( EDU 6 )  )  ( EDU 7 )  )  )  )')
tree.draw()
#tree = Tree.fromstring(s)
#tree.draw()


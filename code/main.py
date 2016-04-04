import vectorizers
import kernels
from nltk.tree import Tree

with open('../data/3gables.csv','r') as f:
	s1 = f.read()

with open('../data/100west.csv','r') as f:
	s2 = f.read()


t1 = Tree.fromstring(s1)
t2 = Tree.fromstring(s2)

v1 = vectorizers.build_norm_vect(t1)
v2 = vectorizers.build_norm_vect(t2)

print kernels.rbf_kernel(v1,v2)
print kernels.vector_kernel(v1,v2)

print kernels.tree_kernel(t1,t2)
#print(v2)
import vectorizers
import kernels
from nltk.tree import Tree
import numpy as np
from sklearn import svm
from sklearn import cross_validation
from sklearn import feature_extraction

with open('../data/3gables.csv','r') as f:
	s1 = f.read()

with open('../data/100west.csv','r') as f:
	s2 = f.read()


int2cl = {0:'descriptive', 1:'argumentative', 2:'narrative',3:'explicative'}

t1 = Tree.fromstring(s1)
t2 = Tree.fromstring(s2)
t_list = [t1,t2,t2,t1,t1,t2]

D = np.array([vectorizers.build_norm_vect(t) for t in t_list])
y = np.array([0,1,1,0,0,1])


#print kernels.rbf_kernel(v1,v2)
#print kernels.vector_kernel(v1,v2)


def compute_kernel(X,Y,kernel=kernels.rbf_kernel):
	"""computes a gram matrix K with matrices X and Y 
	such as K[i,j] = kernel(X[i],Y[j]).
	"""
	K = np.zeros((X.shape[0],Y.shape[0]))
	for i,x in enumerate(X):
		for j,y in enumerate(Y):
			K[i, j] = kernel(x,y)
	return K

#K = compute_kernel(X,X)


clf1 = svm.SVC(kernel="precomputed")
#clf1.fit(K,y)
#print(clf1.predict(K))



v = feature_extraction.DictVectorizer(sparse=False)
X = v.fit_transform(D)
Y = v.inverse_transform(X)
#print X
#print ''
#print Y

clf2 = svm.LinearSVC()
clf2.fit(X,y)
print(clf2.predict(X))
print [int2cl[x] for x in clf2.predict(X)]

scores = cross_validation.cross_val_score(clf2,X,y,cv=2)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


clf3 = svm.SVC(kernel='rbf')
clf3.fit(X,y)
print(clf3.predict(X))

K4 = compute_kernel(t_list,t_list,kernels.tree_kernel)
clf4 = svm.SVC(kernel='precomputed')
clf4.fit(X,y)
print(clf4.predict(X))
#print kernels.tree_kernel(t1,t2)
#print(v2)


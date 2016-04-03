import numpy as np


# Abstract classifier
class Classifier(object):
    def fit(self,x,y):
        raise NotImplementedError("fit not implemented")
    def predict(self,x):
        raise NotImplementedError("predict not implemented")
    def score(self,x,y):
        return (self.predict(x)==y).mean()


# KNN
class KNN(Classifier):
    def fit(self,kernel,x,y,k):
        self.kernel=kernel
        self.x=x
        self.y=y
        self.k=k
        
    def predict(self,z):
        z_labels=np.zeros((len(z)))
        for index,j in enumerate(z):
            dist=np.array([self.kernel(i,j) for i in self.x])
            arg_dist=np.argsort(dist)[::-1][:self.k]
            #print arg_dist
            vote=self.y[arg_dist]
            z_labels[index]=np.argmax(np.bincount(vote))
        return z_labels


#Testing
def cross_validation(model,kernel,x,y,k):
    n=len(x)
    y=np.array(y)
    for index in range(k):
        scores=np.zeros((k))
        ik=int(float(index)*n/k)
        ikp1=int(float(index+1)*n/k)
        x_train=x[:ik]+x[ikp1:]
        y_train=np.concatenate((y[:ik],y[ikp1:]))
        x_test=x[ik:ikp1]
        model.fit(kernel,x_train,y_train,k)        
        y_test=y[ik:ikp1]
        scores[index]=model.score(x_test,y_test)
        print 'round '+str(index)+': '+str(scores[index])+'%.'
    return scores[index].mean()   


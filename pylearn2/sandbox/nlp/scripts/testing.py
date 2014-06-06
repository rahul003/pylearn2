import theano
import pylearn2
import numpy as np
from pylearn2.utils import sharedX
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

def project(w, x):
        """
        Takes a sequence of integers and projects (embeds) these labels
        into a continuous space by concatenating the correspending
        rows in the projection matrix W i.e. [2, 5] -> [W[2] ... W[5]]

        Parameters
        ----------
        x : theano.tensor, int dtype
            A vector of labels (or a matrix where each row is a sample in
            a batch) which will be projected
        """

        #assert 'int' in x.dtype
        #print x.ndim

        if x.ndim == 2:
            shape = (x.shape[0], x.shape[1] * w.shape[1])
            return w[x.flatten()].reshape(shape)
        elif x.ndim == 1:
            return w[x].flatten()
        else:
            assert ValueError("project needs 1- or 2-dimensional input")

v = 10
k = 5
n = 15
x = np.random.randint(0,v,size=(n,6))
#x.shape = (15,6)
y = np.random.randint(0,v,size=(n,1))
#y.shape = (15,1)
w = np.random.rand(v,k)
#w.shape = 10,5
b = np.random.rand(v,1)
#b.shape = 10,1
c = np.random.randint(0,100,(k,6))
C = sharedX(c)
#c shape = 5,6
#we get a column of 5 dimensions for each context word



rproj = w[x.flatten()]
rproj.shape
#90,5
shape = (x.shape[0], x.shape[1] * w.shape[1])
rproj = rproj.reshape(shape)
#15 examples of 6 words each and 5 dim for each word
#so 15 rows. where each row has 6*5 dim
sb = rproj.reshape(rproj.shape[0],k,6)
sb.shape
#made it 15,5,6 now

#c shape is 1,5,6
#so that for each example we still use the same C columns
qh = (C.dimshuffle('x',0,1)*sharedX(sb)).sum(axis=2)

ally = np.arange(v).reshape(v,1)

qw = project(w,y)
allqw = project(w,ally)

swh = (qw*qh).sum(axis=1) + b[y].flatten()
sallwh = theano.tensor.dot(qh,allqw.T)+b[ally].flatten()

esallwh = T.exp(sallwh)
eswh = T.exp(swh)
esallwh = esallwh.sum(axis=1)

print esallwh.eval().shape
print eswh.eval().shape

prob = eswh/esallwh

theano_rng = RandomStreams(seed = np.random.randint(2 ** 15))

#qhat dim  is 15,5

#we dont need this i think
#qh = qhat.dimshuffle('x',0,1)

#qh shape is 1,15,5
#meaning 

#qw = sharedX(projy.reshape(y.shape[0],y.shape[1],k)).dimshuffle(1,0,2)
#qw = sharedX(projy)
#rval = (qhat*qw).sum(axis=1) 

#b[y].flatten()
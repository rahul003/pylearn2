import theano
import pylearn2
import numpy as np
from pylearn2.utils import sharedX

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

#qhat = theano.tensor.tensordot(C,sharedX(sb),axes=[[0,1],[1,2]])

#c shape is 1,5,6
#so that for each example we still use the same C columns
qhat = (C.dimshuffle('x',0,1)*sharedX(sb)).sum(axis=2)

#qhat dim  is 15,5

#we dont need this i think
#qh = qhat.dimshuffle('x',0,1)

#qh shape is 1,15,5
#meaning 
projy = w[y.flatten()]
#projy is of shape 15,5

#qw = sharedX(projy.reshape(y.shape[0],y.shape[1],k)).dimshuffle(1,0,2)
qw = sharedX(projy)
rval = (qhat*qw).sum(axis=1) 

b[y].flatten()
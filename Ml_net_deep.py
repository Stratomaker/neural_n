# coding: utf8
import numpy as np

with open('matrix_DTO.txt', 'r') as source:
    testOL = [map(int, line.split(',')) for line in source]
    print(testOL)

with open('matrix_DTI.txt', 'r') as source:
    testIL = [map(int, line.split(',')) for line in source]
    print(testIL)

#sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

#input dataset

np.random.seed(1)

syn0 = 2*np.random.random((4,5))-1
syn1 = 2*np.random.random((5,4))-1
#print(syn0)

for iter in xrange(100000):

    #forward propagation
    l0 = testIL
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))

    l2_error = testOL - l2

    if (iter%10000) == 0:
        print "Error afer " + str(iter) + ' ' + str(np.mean(np.abs(l2_error)))

    l2_delta = l2_error*nonlin(l2,deriv=True)  

    #back propagation
    l1_error = l2_delta.dot(syn1.T)

    #sigmoid
    l1_delta = l1_error * nonlin(l1,True)

    #update weights
    syn1 += l1.T.dot(l2_delta)
    syn0 += np.dot(l0.T,l1_delta)

print "Output After Training:"
print l2

'''l0 = realIL
l1 = nonlin(np.dot(l0,syn0))
l2 = nonlin(np.dot(l1,syn1))'''
print l2
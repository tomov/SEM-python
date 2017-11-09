
# coding: utf-8

# In[63]:
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from numpy.fft import fft, ifft


# plate's formula for calculating appropriate dimension D 
# based on how much stuff we're storing (K) and how many possibilities there are (N)
# and the error rate err
#
def plate(N, K, err):
    return int(np.round(3.16*(K-0.25)*(np.log(N)-3*np.log(err))))

D = plate(10, 5, 0.01)

# generate new random vector(s) corresponding to symbol(s)
#
def embed(N, D):
    # N = # of vectors to generate
    # D = dimension of vector (= n in Plate's paper)
    #
    return mvn.rvs(mean = np.zeros(D), cov = np.eye(D) * 1/D, size=N)


# circular convolution c = a * b
#
def conv(a, b):
    return np.real(ifft(fft(a) * fft(b)))

# involution of a^* -- a_i = a_-i, modulo D (dimension of a)
#
def involution():
    return np.real(ifft(np.conj(fft(c))))

# circular correlation c = a # b = a^* * b
# approximately inverts circular convolution
# so that b ~= a # (a * b)
#
def corr(a, b):
    return np.real(ifft(np.conj(fft(a)) * fft(b)))

# notice it's flipped in Sam's code
# x_ = decode(encode(x, a), a)
# 


# Sam's example

dog = embed(1, D)
agent = embed(1, D)
cat = embed(1, D)
patient = embed(1, D)
chase = embed(1, D)
verb = embed(1, D)

def encode(a, b):
    return conv(b, a) # swap them to confuse everybody

# MOMCHIL note: notice that I'm not dividing by length(a)
# Sam does that because of his "spike" in the embeddigns which increases the variance
#
def decode(a, b):
    return corr(b, a) # swapped again

sentence = encode(dog,agent) + encode(chase,verb) + encode(cat,patient)
dog_decoded = decode(sentence,agent)

print dog
print dog_decoded


#plt.scatter(dog, dog_decoded)
#
#f = np.polyfit(dog, dog_decoded, 1)
#p = np.poly1d(f)
#plt.plot([min(dog), max(dog)], [p(min(dog)), p(max(dog))], color='red')
#plt.xlabel('dog')
#plt.ylabel('dog~')
#plt.show()


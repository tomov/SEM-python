import numpy as np
from scipy.stats import norm, multivariate_normal
from event_models import LDS

class SEM(object):
   
    # initialize the SEM object
    #
    def __init__(self, opts):
        self.lmda = opts['lambda']
        self.beta = opts['beta']
        self.eta = opts['eta']
        self.alpha = opts['alpha']

        self.reset()

    # reset the object in preparation of a new scene sequence
    #
    def reset(self):
        self.e = [] # MAP event history \hat{e}_i for i=1..n
        self.model = []; # event model for each event type k = 1..K, in lieu of theta's (thetas are in there); also contains expected next scene function f(x_s; e, theta)

        self.prior = []; # prior history P(e_i = k | e_1:i-1 = MAP e_1:i-1) for i=1..n
        self.lik = []; # likelihood history P(s_i = x_i | s_i-1 = x_i-1, e_i-1 = e_i = k) for i=1..n, k=1..K    MOMCHIL: note the difference from the equation
        self.post = []; # posterior history P(e_i = k | s_1:i = x_1:i) for i=1..n, k=1..K
        self.x = []; # scene history x_i for i=1..n
        self.predicted_x = []; # prediction history x_s' = E(lik) = f(x_s; e, theta) for i=1..n

    # process a single scene x_s
    # for online processing
    #
    def scene(self, x):
        K = 0
        if self.e:
            K = np.max(self.e) + 1 # current # of distinct event types

        print '\n'
        print 'x = ', x
        print 'K = ', K

        # CRP prior from Eq 4 
        # prior[k] = P(e_n = k | e_1:n-1)
        #
        print 'e hat = ', self.e
        if not self.e: # it's the very first scene
            prior = [1] # only 1 possible event type
        else:
            prior = [None] * (K + 1)
            C = np.bincount(self.e) # C_k = number of times each event type k occurred
            print 'C = ', C
            for k in range(K + 1):
                if k < K: # k is 0-indexed
                    prior[k] = C[k] + self.lmda * (self.e[-1] == k) # already seen event type (k <= K)
                else:
                    assert k == K
                    prior[k] = self.alpha # new event type (k == K + 1)
        prior = np.array(prior, dtype=float) / sum(prior) # normalize prior

        print 'prior = ', prior

        while len(self.model) < K + 1: # make sure the new event types have associated models
            self.model.append(LDS(len(x), self.eta))

        # Likelihood from Eq 2 and Eq 7
        # lik[k] = P(s_n = x_s | s_n-1, e_n = e_n-1 = k)       MOMCHIL: note the difference from the equation
        #
        predicted_x = [None] * (K + 1) # predicted scene x_s' = f(x_s, e = k, theta), for each event type k
        lik = [None] * (K + 1)
        for k in range(K + 1):
            if not self.x: # or self.e[-1] != k:
                # it's the very first scene OR the previous event type is different
                # MOMCHIL: note the difference from Sam's code: it doesn't make sense unless we're resegmenting
                # MOMCHIL: scratch that; we basically are admitting that the previous scene was segmented incorrectl, however we're accepting this as a loss and just moving along
                predicted_x[k] = self.model[k].initial_x()
            else:
                predicted_x[k] = self.model[k].next_x(self.x[-1])
            print "x_s'[", k, "] = ", predicted_x[k]
            lik[k] = multivariate_normal.pdf(x, predicted_x[k], self.beta * np.eye(len(x))) # Eq 2 TODO replace with log likelihood from Eq 7
        lik = np.array(lik, dtype=float)
        # lik = lik / np.sum(lik)

        print 'lik = ', lik

        # Posterior from Eq 5
        # post[k] = P(e_n = k | s_1:n)
        # see Sam's code for derivation
        #
        post = lik * prior;
        post = post / np.sum(post) # normalize posterior

        print 'post = ', post

        # MAP of event type
        self.e.append(np.argmax(post))

        # update event model of MAP event type
        if self.x: # don't update on first scene -- no past!
            self.model[self.e[-1]].update(self.x[-1], x)
            print self.model[self.e[-1]].b
            print self.model[self.e[-1]].W

        # update histories
        self.lik.append(lik)
        self.prior.append(prior)
        self.post.append(post)
        self.x.append(x)
        self.predicted_x.append(predicted_x)

        return self.e[-1], post

    def segment(self, X):
        self.reset()
        for x in X:
            print x
            self.scene(x)
        return self.e, self.post

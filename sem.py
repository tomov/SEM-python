import numpy as np
from scipy.stats import norm, multivariate_normal
from event_models import LDS

class SEM(object):
   
    # initialize the SEM object
    #
    def __init__(self, opts):
        self.lambda = opts['lambda']
        self.initial_x = opts['initial_scene']
        self.beta = opts['beta']
        self.eta = opts['eta']

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
        self.next_x = []; # prediction history x_s' = E(lik) = f(x_s; e, theta) for i=1..n

    # process a single scene x_s
    # for online processing
    #
    def scene(self, x):
        K = 0
        if self.e
            K = np.argmax(self.e) # current # of distinct event types

        # CRP prior from Eq 4 
        # prior[k] = P(e_n = k | e_1:n-1)
        #
        if ~self.e: # it's the very first scene
            prior = [1] # only 1 possible event type
            self.model = LDS(len(x), self.eta)
        else:
            prior = [None] * (K + 1)
            self.C = np.bincount(self.e) # C_k = number of times each event type k occurred
            for k in range(K + 1):
                if k < K: # k is 0-indexed
                    prior[k] = C[k] + self.lambda * (e[-1] == k) # already seen event type (k <= K)
                else:
                    assert k == K
                    prior[k] = self.alpha # new event type (k == K + 1)
                    self.model[k] = LDS(len(x), self.eta)
        prior = np.array(prior, dtype=float) / sum(prior) # normalize prior

        # Likelihood from Eq 2 and Eq 7
        # lik[k] = P(s_n = x_s | s_n-1, e_n = e_n-1 = k)       MOMCHIL: note the difference from the equation
        #
        next_x = [] # next scene x_s' = f(x_s, e = k, theta), for each event type k
        for k in range(K + 1):
            if ~self.x or self.e[-1] ~= k # it's the very first scene OR the previous event type is different    MOMCHIL: note the difference from Sam's code: it doesn't make sense unless we're resegmenting
                next_x[k] = self.model[k].initial_x()
            elif:
                next_x[k] = self.model[k].next_x(self.x[-1])
            lik[k] = multivarite_normal.pdf(x, next_x[k], self.beta * np.eye(len(x))) # Eq 2 TODO replace with log likelihood from Eq 7
        lik = np.array(lik, dtype=float)

        # Posterior from Eq 5
        # post[k] = P(e_n = k | s_1:n)
        # see Sam's code for derivation
        #
        post = lik * prior;
        post = post / np.sum(post) # normalize posterior

        self.model[k].update(self.x[-1], x) # update event model

        # update event type MAPs
        self.e.append(np.argmax(post))

        # update histories
        self.lik.append(lik)
        self.prior.append(prior)
        self.post.append(post)
        self.x.append(x)
        self.next_x.append(next_x)

    def segment(self, X):
        self.reset()
        for x in X:
            self.scene(x)
        return self.e

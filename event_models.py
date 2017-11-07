import numpy as np

# Linear dynamical system
#
class LDS(object):

    # initialize event model
    #
    def __init__(self, D):
        self.D = D # dimension of each scene xs
        self.b = np.zeros((D, 1))
        self.W = np.eye(D)

    # predict next scene x_s' based on previous scene x_s
    #
    def next_x(self, x):
        assert len(x) == self.D
        x = np.array(x)
        x.resize((self.D, 1))
        next_x = self.b + np.dot(x.transpose(), self.W)
        print next_x.shape
        assert next_x.shape == (self.D, 1)
        return list(x)

    def initial_x(self):
        return [0] * self.D

    def update(self, x, next_x):
        next_x = np.array(next_x)
        next_x.resize((self.D, 1))
        predicted_x = np.array(self.next_x(x))
        predicted_x.resize((self.D, 1))
        x = np.array(x)
        x.resize((self.D, 1))
        db = self.eta * np.dot(next_x - predicted_x, x.transpose())
        dW = self.eta * (next_x - predicted_x)
        self.b = self.b + db
        self.W = self.W + dW

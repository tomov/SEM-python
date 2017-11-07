import numpy as np

# Linear dynamical system
#
class LDS(object):

    # initialize event model
    #
    def __init__(self, D, eta):
        self.D = D # dimension of each scene xs
        self.b = np.zeros((D, 1))
        self.W = np.eye(D)
        self.eta = eta

    # predict next scene x_s' based on previous scene x_s
    #
    def next_x(self, x):
        assert len(x) == self.D
        x = np.array(x, dtype=float)
        x.resize((self.D, 1))
        next_x = self.b.transpose() + np.dot(x.transpose(), self.W)
        assert next_x.shape == (1, self.D)
        return list(next_x.flat)

    def initial_x(self):
        return [0] * self.D

    def update(self, x, next_x):
        next_x = np.array(next_x, dtype=float)
        next_x.resize((self.D, 1))
        predicted_x = np.array(self.next_x(x), dtype=float)
        predicted_x.resize((self.D, 1))
        x = np.array(x, dtype=float)
        x.resize((self.D, 1))
        db = self.eta * (next_x - predicted_x)
        dW = self.eta * np.dot(next_x - predicted_x, x.transpose())
        self.b = self.b + db
        self.W = self.W + dW

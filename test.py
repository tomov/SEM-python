from sem import SEM

opts = {'lambda': 10, 'alpha': 0.1, 'beta': 0.00001, 'eta': 0.01, 'tau': 0.1}
s = SEM(opts)

X = [[1, 1]] * 2 + [[-1, -1]] * 2

e, post = s.segment(X)

print e
print post
#s.scene([0, 0])
#s.scene([0, 0])
#s.scene([0, 0])
#s.scene([0, 0])
#s.scene([1, 1])
#s.scene([1, 1])
#s.scene([1, 1])

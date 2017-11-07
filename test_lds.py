from event_models import LDS

m = LDS(2, 0.1)

print m.initial_x()
print m.next_x([0, 0])

for i in range(100): # the more you train it, the better it gets 
    m.update([1, 1], [2, 3])

print m.b
print m.W

print m.next_x([1, 1])

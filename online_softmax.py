import math
import numpy as np

def normal_softmax(data):
    max = -math.inf
    for i in data:
        if i > max:
            max = i
    
    local_normalization = 0
    for i in data:
        local_normalization += math.exp(i-max)

    softmax = []
    for i in data:
        softmax.append(math.exp(i-max)/local_normalization)

    return softmax


def online_softmax(data):
    current_max = -math.inf 
    prev_max = -math.inf 
    local_normalization = 0

    for i in data:
        if i > current_max:
            prev_max = current_max
            current_max = i
            local_normalization = local_normalization * math.exp(prev_max - current_max)
        local_normalization += math.exp(i - current_max)

    online_softmax = []
    for i in data:
        online_softmax.append(math.exp(i-current_max)/local_normalization)

    return online_softmax

a = np.array([3, 5, 2, 1])
np_s = list(np.exp(a) / np.sum(np.exp(a)))
np_o = np.array(normal_softmax(a))
np_ols = np.array(online_softmax(a))

print(sum(np_s))
print(sum(np_o))
print(sum(np_ols))
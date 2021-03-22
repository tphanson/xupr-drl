import numpy as np
import matplotlib.pyplot as plt

def load():
    try:
        return list(np.load('models/eval.npy'))
    except IOError:
        return None


data = load()
plt.plot(data)
plt.ylabel('Average Return')
plt.show()

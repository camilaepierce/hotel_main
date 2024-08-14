import numpy as np
array = np.array([np.array([0, 1, 2, 3, 4]),
np.array([5, 6, 7, 8, 9]),
np.array([10, 11, 12, 13, 14])])
flat = array.flatten(order = "F")
print(array)
print(flat)
new = np.reshape(flat, (5, 3))
print(new)

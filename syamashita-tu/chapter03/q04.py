import numpy as np

a = np.arange(20).reshape(5, 4)

print(a)
print()
print(a[2, 3])
print()
print(a[:, 2])
print()
print(a[1::2, :])
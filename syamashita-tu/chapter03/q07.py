import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-2, 2, 100)
y1 = 2 ** x 
y2 = 2 ** -x 

plt.plot(x, y1, label="y = 2^x")
plt.plot(x, y2, label="y = 2^-x")

plt.xlabel("x") 
plt.ylabel("y")
plt.title("y = 2^x, y = 2^-x")
plt.grid()
plt.legend()
plt.show()  
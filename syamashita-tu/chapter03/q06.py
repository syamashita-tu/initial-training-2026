import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 100)
y1 = x ** 3 -6 * (x** 2) + 9 * x + -3
y2 = (x+2) ** 3 - 6 * ((x+2) ** 2) + 9 * (x+2) - 3 +2

plt.plot(x, y1, label="y1")
plt.plot(x, y2, label="y2")
plt.xlabel("x") 
plt.ylabel("y")
plt.title("y = x^3 - 6x^2 + 9x - 3")
plt.grid()
plt.legend()
plt.show()  
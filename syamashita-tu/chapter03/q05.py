import numpy as np

table = np.zeros((361, 4))
title = np.array(["\theta", "\sin{\theta}", "\cos{\theta}", "\tan{\theta}"])


for i in range (0, 361):
    theta = i
    sin_theta = np.sin(np.radians(theta))
    cos_theta = np.cos(np.radians(theta))
    tan_theta = np.tan(np.radians(theta))
    table[i] = [theta, sin_theta, cos_theta, tan_theta]
    
print(title)
print(table)
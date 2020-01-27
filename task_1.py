from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
#drawing a plot based on specific function for each coordinate

fig = plt.figure()
ax = plt.axes(projection='3d')

m = 50
increment = np.arange(m)

x = np.sin(increment)
y = increment
z = np.empty(m)

for i in range (m-1):
    z[i] = 5


ax.plot3D(x, y, z, 'gray')
plt.title('blabla')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
#drawing a plot based on specific function for each coordinate

fig = plt.figure()
ax = plt.axes(projection='3d')

m = 4      #number of points
increment = np.arange(m)

x = np.empty(m)
y = np.empty(m)
z = np.empty(m)

for i in range (m-1):
    x[i] = 0
    y[i] = 0
    z[i] = 0 



ax.plot3D(x, y, z, 'gray')
plt.title('blabla')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
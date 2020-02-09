from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import math  

#drawing a plot based on specific function for each coordinate

fig = plt.figure()
ax = plt.axes(projection='3d')

m = 4      #number of points
increment = np.arange(m)

x = np.empty(m)
y = np.empty(m)
z = np.empty(m)
u = np.empty(m)
v = np.empty(m)
w = np.empty(m)
# for i in range (m):
#     x[i] = input("x_{i} coord")
#     y[i] = input("y_{i} coord")
#     z[i] = input("z_{i} coord")

x = [1, 10,10,1]
y = [1, 1, 10,10]
z = [10,10,1, 1]

for i in range(m-1):
    u[i]=-x[i]+x[i+1]
    v[i]=-y[i]+y[i+1]
    w[i]=-z[i]+z[i+1]

u[m-1]=-x[m-1]+x[0]
v[m-1]=-y[m-1]+y[0]
w[m-1]=-z[m-1]+z[0]

ax.quiver(x, y, z, u, v, w, arrow_length_ratio =0.1)

#ax.plot3D(x, y, z, 'gray')
plt.title('Vector Square')
plt.xlabel('x', color = "blue")
plt.ylabel('y', color = "blue")
ax.set_zlabel('z', color = "blue")
plt.show()
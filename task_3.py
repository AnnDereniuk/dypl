from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

#main task trial
#pre configs
fig = plt.figure()
ax = plt.axes(projection='3d')

#number of points
m = 2 

#G
GAMMA=1.0
#dots A,B
x = np.empty(m)
y = np.empty(m)
z = np.empty(m)

dr_x = np.empty(m)
dr_y = np.empty(m)
dr_z = np.empty(m)

#dot V
grid_x =np.empty(1)
grid_y =np.empty(1)
grid_z =np.empty(1)

r_x = np.empty(m)
r_y = np.empty(m)
r_z = np.empty(m)

x = [3,3]
y = [1,5]
z = [0,4]
grid_x = [7]
grid_y = [2]
grid_z = [1]

#setting vector coords for AB
for i in range(m-1):
    dr_x[i]=x[i+1]-x[i]
    dr_y[i]=y[i+1]-y[i]
    dr_z[i]=z[i+1]-z[i]

#setting vector coords for AV:
r_x = grid_x[0] -x[0]
r_y = grid_y[0] -y[0]
r_z = grid_z[0] -z[0]

#finding length of AV:
r_length = (r_x**2 + r_y**2 + r_z**2)**(1/2)
r_squared = r_length**3

du = (dr_y[0]*r_z - dr_z[0]*r_y)/r_squared
dv = (dr_z[0]*r_x-dr_x[0]*r_z)/r_squared
dw = (dr_x[0]*r_y-dr_y[0]*r_x)/r_squared

velocity_x = (GAMMA/4*np.pi)*du
velocity_y = (GAMMA/4*np.pi)*dv
velocity_z = (GAMMA/4*np.pi)*dw

plt.plot(grid_x, grid_y,grid_z, 'ro')
plt.plot(x, y, z, 'ro')

ax.quiver(x[0], y[0], z[0], dr_x, dr_y, dr_z, length =1,arrow_length_ratio =0.1)
ax.quiver(x[0], y[0], z[0], r_x, r_y, r_z, length =1, arrow_length_ratio =0.1,linestyle = '--', color = "black")

ax.quiver(grid_x[0], grid_y[0],grid_z[0],
velocity_x-grid_x[0],velocity_y-grid_y[0],velocity_z-grid_z[0],
length = 0.2, arrow_length_ratio =0.1, color = "red")

plt.title('Task 4')
plt.xlabel('x', color = "blue")
plt.ylabel('y', color = "blue")
ax.set_zlabel('z', color = "blue")
plt.show()
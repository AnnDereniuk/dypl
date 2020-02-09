from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from some_functions import get_velocity_coefs_with_GAMMA, get_vector_coords, vector_s_squared_length

#number of points
m=4

#number of grid dots
k=1

#dots
x = np.empty(m)
y = np.empty(m)
z = np.empty(m)

#----------------------------------
x = [1, 10,10,1]
y = [1, 1, 10,10]
z = [10,10,1, 1]

#vectors' coords; AB, BC, CD, DA
dr_x = np.empty(m)
dr_y = np.empty(m)
dr_z = np.empty(m)
for i in range(m-1):
    dr_x[i]=get_vector_coords(x[i],x[i+1])
    dr_y[i]=get_vector_coords(y[i],y[i+1])
    dr_z[i]=get_vector_coords(z[i],z[i+1])
dr_x[m-1]=get_vector_coords(x[m-1],x[0])
dr_y[m-1]=get_vector_coords(y[m-1],y[0])
dr_z[m-1]=get_vector_coords(z[m-1],z[0])

#grid dots
grid_x =np.empty(k)
grid_y =np.empty(k)
grid_z =np.empty(k)
grid_x = [7]
grid_y = [2]
grid_z = [1]

#----------------------------------
r_x = np.empty(k)
r_y = np.empty(k)
r_z = np.empty(k)
r_squared = np.empty(k)

#----------------------------------
du = np.empty(k)
dv = np.empty(k)
dw = np.empty(k)

#----------------------------------
velocity_x = np.empty(k)
velocity_y = np.empty(k)
velocity_z = np.empty(k)

#----------------------------------
for i in range(len(r_x)):
    #vector coords for grid dots
    r_x[i] = get_vector_coords(x[i],grid_x[i])
    r_y[i] = get_vector_coords(y[i],grid_y[i])
    r_z[i] = get_vector_coords(z[i],grid_z[i])
    
    #finding length of AV:
    r_squared[i] = vector_s_squared_length(r_x[i], r_y[i], r_z[i])
    
    #finding components du, dv, dw
    du[i] = (dr_y[0]*r_z[i] - dr_z[0]*r_y[i])/r_squared[i]
    dv[i] = (dr_z[0]*r_x[i] - dr_x[0]*r_z[i])/r_squared[i]
    dw[i] = (dr_x[0]*r_y[i] - dr_y[0]*r_x[i])/r_squared[i]
    
    #final velocity values
    velocity_x[i] = get_velocity_coefs_with_GAMMA(du[i])
    velocity_y[i] = get_velocity_coefs_with_GAMMA(dv[i])
    velocity_z[i] = get_velocity_coefs_with_GAMMA(dw[i])


fig = plt.figure()
ax = plt.axes(projection='3d')

plt.plot(grid_x, grid_y,grid_z, 'ro')
plt.plot(x, y, z, 'ro')

ax.quiver(x, y, z, dr_x, dr_y, dr_z, length =1,arrow_length_ratio =0.1)
ax.quiver(x[0], y[0], z[0], r_x, r_y, r_z, length =1, arrow_length_ratio =0.1)
ax.quiver(grid_x[0], grid_y[0],grid_z[0],

velocity_x-grid_x[0],velocity_y-grid_y[0],velocity_z-grid_z[0],
length = 0.5, arrow_length_ratio =0.1, color = "red")

#configs

plt.title('Velocity')
plt.xlabel('X', color = "blue")
plt.ylabel('Y', color = "blue")
ax.set_zlabel('Z', color = "blue")
plt.show()
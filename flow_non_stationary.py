from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from some_functions import *
GAMMA_ZERO = 0.
VELOCITY_ON_INF = 10.
#main task trial
#pre configs
np.set_printoptions(precision=13)
fig = plt.figure(figsize=(6,6))
ax = plt.axes()

#grid
dots_quantity = 40
g = dots_quantity/2
grid_vertical = np.linspace(-g, g+2, dots_quantity)
grid_horizontal = np.linspace(-g, g-2, dots_quantity)
grid_x,grid_y = np.meshgrid(grid_vertical, grid_horizontal)

# print("grid х,у:")
# print(grid_x)
# print(grid_y)

m = 10   #half-number of vortex dots

x_center = 3
y_center = 6

x_left = chooseLeftPointX(x_center)
y_left = chooseLeftPointY(y_center)

x_right = chooseRightPointX(x_center)

#building figure
x1 = np.linspace(x_left,x_center,m-1, endpoint=FALSE)
y1 = np.linspace(y_left,y_center,m, endpoint=FALSE)
x2 = np.linspace(x_center,x_right,m+1)
y2 = np.linspace(y_center,y_left,m)

#array of discrete points
x = np.append(x1,x2) 
y = np.append(y1,y2)

print("х,у вихорів:")
print(x)
print(y)

k=2*m-1
#finding colocation marks:
colocation_x=np.empty(k)
colocation_y=np.empty(k)
for i in range(colocation_x.size):
    colocation_x[i] = (x[i] + x[i+1])/2.
    colocation_y[i] = (y[i] + y[i+1])/2.

print("х,у колокацій:")
print(colocation_x)
print(colocation_y)

normal_x=np.empty(k)
normal_y=np.empty(k)
for i in range(normal_x.size):
    normal_y[i]=(x[i+1]-x[i])/get_vector_length_2d(get_vector_coords(x[i],x[i+1]), get_vector_coords(y[i],y[i+1]))
    normal_x[i]=-(y[i+1]-y[i])/get_vector_length_2d(get_vector_coords(x[i],x[i+1]), get_vector_coords(y[i],y[i+1]))
print("х, у нормалей:")
print(normal_x)
print(normal_y)

left_matrix_part_x = left_matrix_part_y = left_matrix= np.empty(shape=(2*m,2*m))
right_matrix_part_x = right_matrix_part_y = right_matrix = np.empty(2*m)


for j in range(right_matrix.size-1):
    for i in range(right_matrix.size):
        left_matrix_part_x[j][i] = get_velocity_j(x[i], colocation_x[j], get_R(colocation_x[j], colocation_y[j], x[i], x[i+1], y[i], y[i+1]))*normal_y[j]

        left_matrix_part_y[j][i] = get_velocity_j(colocation_y[j], y[i], get_R(colocation_x[j], colocation_y[j], x[i], x[i+1], y[i], y[i+1]))*normal_x[j]

        left_matrix[j][i] = left_matrix_part_x[j][i] + left_matrix_part_y[j][i]

        right_matrix_part_x[j] = -VELOCITY_ON_INF*normal_y[j]
        right_matrix_part_y[j] = -VELOCITY_ON_INF*normal_x[j]
            
        right_matrix[j] = right_matrix_part_x[j] + right_matrix_part_y[j]
        left_matrix[k][i] = 1.
right_matrix[k] = GAMMA_ZERO
print("Ліва частина СЛАР:")
print(left_matrix)
print("Права частина СЛАР:")
print(right_matrix)

gamma_arr = np.empty(k)
gamma_arr = np.linalg.solve(left_matrix, right_matrix)

print("Гамма:")
print(gamma_arr)

velocity_x = np.array([])
velocity_y = np.array([])  
grid_x=np.asarray(grid_x).reshape(-1)
grid_y=np.asarray(grid_y).reshape(-1)
# print(grid_x)


#TODO R: change way of getting R
for i in range(0,dots_quantity):
    for j in range(0,dots_quantity):
        res_x=VELOCITY_ON_INF
        res_y=VELOCITY_ON_INF
        for v in range(gamma_arr.size):
            res_x= res_x+gamma_arr[v]*get_velocity_j(x[v], grid_horizontal[i], get_R(grid_horizontal[i], grid_vertical[j], x[v], x[v+1], y[v], y[v+1]))
            res_y= res_y+gamma_arr[v]*get_velocity_j(grid_vertical[j], y[v], get_R(grid_horizontal[i], grid_vertical[j], x[v], x[v+1], y[v], y[v+1]))
        velocity_x=np.append(velocity_x, res_x)
        velocity_y=np.append(velocity_y, res_y)

plt.plot(y, x, 'ro', markersize=2)                                      #vortices
plt.plot(y, x, 'gray')
plt.plot(colocation_y, colocation_x, 'bo', markersize=2)                #colocation dots
ax.quiver(colocation_y, colocation_x, normal_y, normal_x)               #normals
plt.plot(grid_x,grid_y, 'ko', markersize=1)                             #grid dots
ax.quiver(grid_x,grid_y, velocity_x,velocity_y)          #velocity
plt.title('plot')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
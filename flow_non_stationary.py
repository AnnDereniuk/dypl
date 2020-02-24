from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from some_functions import *
GAMMA_ZERO = 0.
VELOCITY_ON_INF = 0.7
#main task trial
#pre configs
np.set_printoptions(precision=13)
fig = plt.figure(figsize=(6,6))
ax = plt.axes()

#grid
dots_quantity = 50
g = dots_quantity/2
grid_vertical = np.linspace(-g, g+2, dots_quantity)
grid_horizontal = np.linspace(-g, g-2, dots_quantity)
grid_x,grid_y = np.meshgrid(grid_vertical, grid_horizontal)

print("grif х,у:")
print(grid_x)
print(grid_y)

m = 10   #half-number of vortex dots

x_center = 3
y_center = 6

x_left = chooseLeftPointX(x_center)
y_left = chooseLeftPointY(y_center)

x_right = chooseRightPointX(x_center)

#building figure
x1 = np.linspace(x_left,x_center,m, endpoint=FALSE)
y1 = np.linspace(y_left,y_center,m, endpoint=FALSE)
x2 = np.linspace(x_center,x_right,m)    #todo
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
for i in range(0,k):
    colocation_x[i] = (x[i] + x[i+1])/2.
    colocation_y[i] = (y[i] + y[i+1])/2.

print("х,у колокацій:")
print(colocation_x)
print(colocation_y)

normal_x=np.empty(k)
normal_y=np.empty(k)
for i in range(0,k):
    normal_y[i]=(x[i+1]-x[i])/get_vector_length_2d(get_vector_coords(x[i],x[i+1]), get_vector_coords(y[i],y[i+1]))
    normal_x[i]=-(y[i+1]-y[i])/get_vector_length_2d(get_vector_coords(x[i],x[i+1]), get_vector_coords(y[i],y[i+1]))
print("х, у нормалей:")
print(normal_x)
print(normal_y)

left_matrix_part_x = left_matrix_part_y = left_matrix= np.empty(shape=(k,k))
right_matrix_part_x = right_matrix_part_y = right_matrix = np.empty(k)


for j in range(0, k):
    for i in range (0, k):
            left_matrix_part_x[j][i] = get_velocity_j(x[j], colocation_x[i], get_R(colocation_x[i], colocation_y[i],x[j], x[j+1], y[j],y[j+1]))*normal_x[i]

            left_matrix_part_y[j][i] = get_velocity_j(colocation_y[i], y[j], get_R(colocation_x[i], colocation_y[i],x[j], x[j+1], y[j],y[j+1]))*normal_y[i]

            left_matrix[j][i] = left_matrix_part_x[j][i]+left_matrix_part_y[j][i]

            right_matrix_part_x[j] = -VELOCITY_ON_INF*normal_x[i]
            right_matrix_part_y[j] = -VELOCITY_ON_INF*normal_y[i]
            
            right_matrix[j] = right_matrix_part_x[j]+right_matrix_part_y[j]
            left_matrix[k-1][i] = 1.
right_matrix[k-1] = GAMMA_ZERO
# print("Ліва частина СЛАР:")
# print(left_matrix)
# print("Права частина СЛАР:")
# print(right_matrix)

gamma_arr = np.empty(2*m)
gamma_arr=np.linalg.solve(left_matrix, right_matrix)

print("Гамма:")
print(gamma_arr)

# velocity_x=np.empty(dots_quantity)
# velocity_y=np.empty(dots_quantity)
# for i in range(0,dots_quantity):
#     for j in range(0,dots_quantity):
#         for v in range(0,k):
#             velocity_x[i]+=gamma_arr[v]*get_velocity_j(x[v], grid_x[i][j], get_R(grid_x[i][j], grid_y[i][j], x[v], x[v+1], y[v], y[v+1]))
#             velocity_y[i]+=gamma_arr[v]*get_velocity_j(grid_y[i][j], y[v], get_R(grid_y[i][j], grid_y[i][j],x[v], x[v+1], y[v], y[v+1]))
#             velocity_x[i]+=VELOCITY_ON_INF
#             velocity_y[i]+=VELOCITY_ON_INF 


velocity_x=np.array([])
velocity_y=np.array([])
grid_x=np.asarray(grid_x).reshape(-1)
grid_y=np.asarray(grid_y).reshape(-1)
print(grid_x)

# for i in range(0,dots_quantity**2):
#     for v in range(0,k):
#         velocity_x[i]+=gamma_arr[v]*get_velocity_j(x[v], grid_x[i], get_R(grid_x[i], grid_y[i], x[v], x[v+1], y[v], y[v+1]))
#         velocity_y[i]+=gamma_arr[v]*get_velocity_j(grid_y[i], y[v], get_R(grid_y[i], grid_y[i],x[v], x[v+1], y[v], y[v+1]))
#     velocity_x[i]+=VELOCITY_ON_INF
#     velocity_y[i]+=VELOCITY_ON_INF 

for i in range(0,dots_quantity):
    for j in range(0,dots_quantity):
        res_x=0.
        res_y=0.
        for v in range(0,k):
            res_x+=gamma_arr[v]*get_velocity_j(x[v], grid_horizontal[i], get_R(grid_horizontal[i], grid_vertical[j], x[v], x[v+1], y[v], y[v+1]))
            res_y+=gamma_arr[v]*get_velocity_j(grid_vertical[j], y[v], get_R(grid_horizontal[i], grid_vertical[j], x[v], x[v+1], y[v], y[v+1]))
        res_x+=VELOCITY_ON_INF
        res_y+=VELOCITY_ON_INF
        velocity_x=np.append(velocity_x, res_x)
        velocity_y=np.append(velocity_y, res_y)

plt.plot(x, y, 'ro', markersize=2)                                      #vortices
plt.plot(x, y, 'gray')
plt.plot(colocation_x, colocation_y, 'bo', markersize=2)                #colocation dots
# ax.quiver(colocation_x, colocation_y, normal_x, normal_y)               #normals
plt.plot(grid_x,grid_y, 'ko', markersize=1)                             #grid dots
ax.quiver(grid_x,grid_y, velocity_x,velocity_y, norm=TRUE)                         #velocity

plt.title('plot')
plt.xlabel('x')
plt.ylabel('y')
plt.show()



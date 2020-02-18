from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from some_functions import *

#main task trial
#pre configs
fig = plt.figure(figsize=(6,6))
ax = plt.axes()

#grid
grid_vertical = np.linspace(0,6,40)
grid_horizontal = np.linspace(0,6,40)
grid_x,grid_y =np.meshgrid(grid_vertical, grid_horizontal)

m = 10   #half-number of vortex dots
velocity_on_inf=1

x_center = 3
y_center = 3

x_left = chooseLeftPointX(x_center)
y_left = chooseLeftPointY(y_center)

x_right = chooseRightPointX(x_center)

#building figure
x1 = np.linspace(x_left,x_center,m)
y1 = np.linspace(y_left,y_center,m)
x2 = np.linspace(x_center,x_right,m)
y2 = np.flip(y1)

# if (x1[m-1]==x2[0]):
#    x1 = np.delete(x1,x1[m-1])
#    y1 = np.delete(y1,y1[m-1])

#array of discrete points
x = np.append(x1,x2) 
y = np.append(y1,y2)

#finding colocation marks:
colocation_x=np.empty(2*m-1)
colocation_y=np.empty(2*m-1)
for i in range(0,2*m-2):
    colocation_x[i] = (x[i] + x[i+1])/2
    colocation_y[i] = (y[i] + y[i+1])/2

normal_x=np.empty(2*m-1)
normal_y=np.empty(2*m-1)
for i in range(0,2*m-2):
    normal_y[i]=(colocation_x[i+1]-colocation_x[i])/get_vector_length_2d(get_vector_coords(colocation_x[i+1],colocation_x[i]),
     get_vector_coords(colocation_y[i+1],colocation_y[i]))
    normal_x[i]=-(colocation_y[i+1]-colocation_y[i])/get_vector_length_2d(get_vector_coords(colocation_x[i+1],colocation_x[i]), 
    get_vector_coords(colocation_y[i+1],colocation_y[i]))
print("х, у нормалей:")
print(normal_x)
print(normal_y)


left_matrix_part_x = left_matrix_part_y = left_matrix= np.empty(shape=(2*m,2*m))
right_matrix_part_x = right_matrix_part_y = right_matrix = np.empty(2*m)

for j in range(0, 2*m-2):
    for i in range (0, 2*m-2):
            left_matrix_part_x[j][i]=get_velocity_j(x[j], colocation_x[i],get_R(get_vector_coords(x[j], colocation_x[i]),get_vector_coords(y[j], colocation_y[i])))*normal_x[i]
            left_matrix_part_y[j][i]=get_velocity_j(y[j], colocation_y[i],get_R(get_vector_coords(x[j], colocation_x[i]),get_vector_coords(y[j], colocation_y[i])))*normal_y[i]
            left_matrix[j][i]=np.dot(left_matrix_part_x[j][i],left_matrix_part_y[j][i]) 
            right_matrix_part_x[j]=-velocity_on_inf*normal_x[j]
            right_matrix_part_y[j]=-velocity_on_inf*normal_y[j]
            right_matrix[j]=np.dot(right_matrix_part_x[j],right_matrix_part_y[j])
left_matrix[2*m-1][i]=1
right_matrix[2*m-1]=0
print("Ліва частина СЛАР:")
print(left_matrix)
print("Права частина СЛАР:")
print(right_matrix)

gamma_arr = np.empty(2*m)
gamma_arr=np.linalg.solve(left_matrix, right_matrix)

print("Гамма:")
print(gamma_arr)

plt.plot(x, y, 'ro', markersize=2)
plt.plot(x, y, 'gray')
plt.plot(colocation_x, colocation_y, 'bo', markersize=2)
ax.quiver(colocation_x, colocation_y, normal_x, normal_y)


plt.plot(grid_x,grid_y, 'ko', markersize=1)

plt.title('blabla')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
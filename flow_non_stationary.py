from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from some_functions import *

#main task trial
#pre configs
fig = plt.figure(figsize=(6,6))
ax = plt.axes()

#grid
grid_vertical = np.linspace(0,5,30)
grid_horizontal = np.linspace(0,5,30)
grid_x,grid_y =np.meshgrid(grid_vertical, grid_horizontal)

m = 10   #half-number of vortex dots

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

#array of discrete points
x = np.append(x1,x2) 
y = np.append(y1,y2)

#finding colocation marks:
colocation_x=np.empty(2*m-1)
colocation_y=np.empty(2*m-1)
for i in range(0,2*m-1):
    colocation_x[i] = (x[i] + x[i+1])/2
    colocation_y[i] = (y[i] + y[i+1])/2








plt.plot(x, y, 'ro', markersize=2)
plt.plot(x, y, 'gray')
plt.plot(colocation_x, colocation_y, 'bo', markersize=2)

# plt.plot(grid_x,grid_y, 'ko', markersize=1)

plt.title('blabla')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
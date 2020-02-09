from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

# xvalues = np.linspace(0, 10, 10)
# yvalues = np.linspace(0, 10, 10)
# zvalues =[0,0,0,0,0,0,0,0,0,0]
# xx, yy, zz = np.meshgrid(xvalues, yvalues, zvalues)
fig = plt.figure(figsize=(10,15))
ax = fig.gca(projection='3d')

x = [1, 3, 1, 3]
y = [0.5,  0.5, 5, 5]
z = [0, 1, 0, 3]
xm=[2]
ym=[1]
zm=[1]
intensity =1
# np.pi


# u = [1,1,1,1]
# v=[1,1,1,1]
# g=[1,1,1,1]
# for i in range(0, len(x), 2):
#     plt.plot(x[i:i+2], y[i:i+2], 'ro-')
#     plot.show()

plt.plot(x, y, z, 'ro')
plt.plot(xm,ym,zm, 'ro')

def connectpoints(x, y, z, p1, p2):
    x1, x2 = x[p1], x[p2]
    y1, y2 = y[p1], y[p2]
    z1, z2 = z, z
    plt.plot([x1, x2], [y1, y2], [z1, z2], 'k-')
    # ax.quiver(xx,yy,[x1, x2], [y1, y2], [z1, z2])
    return [x1, x2], [y1, y2], [z1, z2]


# for i in np.arange(0,len(x),2):
#     connectpoints(x,y,i,i+1)

a=connectpoints(x, y, 0, 0, 1)
b=connectpoints(x, y, 0, 2, 3)
c=connectpoints(x, y, 0, 2, 0)
d=connectpoints(x, y, 0, 3, 1)
# ax.quiver(x,y,z,u,v,g)
# ax.plot()
# plt.axis('equal')
plt.xlabel('x', color = "blue")
plt.ylabel('y', color = "blue")
ax.set_zlabel('z', color = "blue")
plt.show()
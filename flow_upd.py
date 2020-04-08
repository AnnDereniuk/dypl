# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import animation
from IPython.display import HTML
from some_functions import *

# %pylab inline

num = 20
V_inf = np.array([1., 0.], dtype=np.float32)
p_inf = 1
gamma_0 = 1.

N1 = 10000
N2 = 500
N = 2*N1+N2
i = -1
x_list, y_list = list(), list()
discrete_points = np.empty(shape=(N, 2), dtype=np.float32)

m = 10   #half-number of vortex dots

x_center = 0
y_center = 1

x_left = chooseLeftPointX(x_center)
y_left = chooseLeftPointY(y_center)

x_right = chooseRightPointX(x_center)

#building figure
x1 = np.linspace(x_left,x_center,m)
y1 = np.linspace(y_left,y_center,m)
x2 = np.linspace(x_center,x_right,m)
y2 = np.linspace(y_center,y_left,m)

#array of discrete points
x = np.append(x1,x2) 
y = np.append(y1, y2)
x_list = x
y_list = y


discrete_points = (x,y)
# for y in np.linspace(- 0.25, - 0.5 + 0.25/N2, N2):
#   i = i+1
#   x_list.append(0.5)
#   y_list.append(y)
#   discrete_points[i] = (0.5,y)
  
# for x in np.linspace(0.5, -0.5, N1):
#   i = i + 1 
#   x_list.append(x)
#   y = - 1/2*np.sqrt(1-(x-0.5)*(x-0.5))
#   y_list.append(y)
#   discrete_points[i] = (x,y)
  
# for x in np.linspace(-0.5 + 1/N1, 0.5, N1):
#   i = i + 1
#   x_list.append(x)
#   y = 1/2*np.sqrt(1-(x-0.5)*(x-0.5))
#   y_list.append(y)
#   discrete_points[i] = (x,y)




print(discrete_points)
plt.axis("equal")
plt.xticks(np.arange(-1, 1.1, 0.5))
plt.yticks(np.arange(-1, 1.1, 0.5))
plt.plot(x_list, y_list)

def split_curve(x_list, y_list, start_point, cnt) -> list:
    res = list()
    res.append(start_point)
    
    left = 1e-5
    right = np.sqrt(max((x - start_point[0])**2 + (y - start_point[1])**2
                        for x, y in zip(x_list, y_list))) + 1e-5
    while right - left > 1e-5:
        step = (left + right) / 2
        c = 1
        curr_x, curr_y = start_point
        for x, y in zip(x_list, y_list):
            r = (x - curr_x)**2 + (y - curr_y)**2
            if r >= step**2:
                c += 1
                curr_x, curr_y = x, y
        if c < cnt:
            right = step
        else:
            left = step

    step = left
    curr_x, curr_y = start_point
    for x, y in zip(x_list, y_list):
        r = (x - curr_x)**2 + (y - curr_y)**2
        if r >= step**2:
            curr_x, curr_y = x, y
            res.append((x, y))


    arr = np.empty(shape=(len(res), 2), dtype=np.float32)
    arr[:, 0] = np.array([p[0] for p in res])
    arr[:, 1] = np.array([p[1] for p in res])

    return arr
 
discrete_points = split_curve(x_list, y_list, (0.5, - 0.25), num)

plt.axis("equal")
plt.xticks(np.arange(-1, 1.1, 0.5))
plt.yticks(np.arange(-1, 1.1, 0.5))
plt.plot(discrete_points[:, 0], discrete_points[:, 1], 'ro',markersize=2)

collocation_points = list()
collocation_points = (discrete_points[:-1] + discrete_points[1:]) / 2

plt.axis("equal")
plt.xticks(np.arange(-1, 1.1, 0.5))
plt.yticks(np.arange(-1, 1.1, 0.5))
plt.plot(discrete_points[:, 0], discrete_points[:, 1], 'ro',markersize=2)
plt.plot(collocation_points[:, 0], collocation_points[:, 1], 'bo', markersize=2)

tau = discrete_points[1:] - discrete_points[:-1]
tau /= np.linalg.norm(tau, axis=1, keepdims=True)

normals = np.empty_like(tau)
normals[:, 0] = -tau[:, 1]
normals[:, 1] = tau[:, 0]

plt.axis("equal")
plt.xticks(np.arange(-1.5, 2.1, 0.5))
plt.yticks(np.arange(-2, 2.1, 0.5))
plt.plot(discrete_points[:, 0], discrete_points[:, 1], 'ro', markersize=2)
plt.plot(collocation_points[:, 0], collocation_points[:, 1], 'bo', markersize=2)
_norm_vecs = np.empty(shape=(2 * normals.shape[0], 2), dtype=normals.dtype)
_paired_ind = (np.arange(0, _norm_vecs.shape[0]) % 2 == 0)
_norm_vecs[_paired_ind] = collocation_points
_norm_vecs[np.bitwise_not(_paired_ind)] = collocation_points + normals
for i in range(0, len(_norm_vecs), 2):
    # plt.plot([_norm_vecs[i][0], _norm_vecs[i + 1][0]], [_norm_vecs[i][1],_norm_vecs[i + 1][1]], 'g')
    # plt.plot([_norm_vecs[i + 1][0]], [_norm_vecs[i + 1][1]], 'g')
    ax = plt.axes()
    ax.quiver(collocation_points[:, 0], collocation_points[:, 1],normals[:, 0], normals[:, 1])
delta = np.min(np.linalg.norm(discrete_points[1:] - discrete_points[:-1], axis=1)) / 2

def calc_R(point, discrete_points, delta):
    #print("2:", point.shape)
    if len(point.shape) == 3:
        vv = np.ones(shape=(len(point), len(point), discrete_points.shape[0], 2), dtype=discrete_points.dtype)
    else:
        vv = np.ones(shape=(1, 1, discrete_points.shape[0], 2), dtype=discrete_points.dtype)
    vv[:, :, :, 0] *= delta
    point = np.expand_dims(point, axis=-2)
    #print("3:", point.shape)
    vv[:, :, :, 1] = np.linalg.norm(discrete_points - point, axis=-1)
    R_arr = np.max(vv, axis=-1, keepdims=True)
    return R_arr

def calc_V0j(colloc_point, points_0):
    """
    :param colloc_point: (xk, yk)
    :param points_0: vectors of discrete points, t=0
    """
#     print("6:", colloc_point.shape, points_0.shape)
    R_arr = calc_R(colloc_point, points_0, delta)
#     print("4:", R_arr.shape)
    R_arr = np.squeeze(R_arr)
    R_arr = np.expand_dims(R_arr, axis=-1)
    V0 = (points_0 - colloc_point) / (2 * np.pi * R_arr * R_arr)
#     print("5:", V0.shape)
    V0 = V0[:, ::-1]
    V0[:, 1] *= -1
    return V0

A = np.empty(shape=(discrete_points.shape[0], discrete_points.shape[0]), dtype=discrete_points.dtype)

for k in range(len(collocation_points)):
    colloc_point = collocation_points[k]
    V0 = calc_V0j(colloc_point, discrete_points)
#     print("7:", V0.shape)
    A[k, :] = V0.dot(normals[k])

A[-1, :] = 1

b = np.empty(shape=(discrete_points.shape[0],), dtype=discrete_points.dtype)
b[:-1] = -normals.dot(V_inf)
b[-1] = gamma_0

gamma_coef = np.linalg.solve(A, b)

def calc_Vj(points_xy, points_xy_0, t):
    """
    :param points_xy: (i, j) -> (x, y)
    """
    delta = np.min(np.linalg.norm(points_xy_0[1:] - points_xy_0[:-1], axis=1)) / 2
    R_arr = calc_R(points_xy, points_xy_0, delta)
    #R_arr = np.squeeze(R_arr)
    # print(R_arr.shape)
    points_xy = np.expand_dims(points_xy, axis=-2)
    V0 = (points_xy_0 - points_xy) / (2 * np.pi * R_arr * R_arr)
#     print("1:", V0.shape)
    V0 = V0[:, :, :, ::-1]
    V0[:, :, :, 1] *= -1
    return V0

def calc_V(points_xy, t, gamma_coef, points_xy_0):
    vj = calc_Vj(points_xy, points_xy_0, t)
    # print(vj.shape, gamma_coef.shape, V_inf.shape)
    return V_inf + gamma_coef.dot(vj)

X_1, X_2 = -2, 2.1
Y_1, Y_2 = -2, 2.1

# matrix of (x, y) coordinates of (j, i) point
points_coords = np.dstack(np.meshgrid(np.arange(X_1, X_2, delta/2), np.arange(Y_1, Y_2, delta/2)))

V_field = calc_V(points_coords, 0, gamma_coef, discrete_points)
# print(V_field.shape)
U, V = V_field[:, :, 0], V_field[:, :, 1]
# print(U.shape, V.shape)

def calc_phi_simple(points_xy, t, gamma_coef, points_xy_0):
    dxdy = np.expand_dims(points_coords, axis=-2) - points_xy_0
#     print("8:", dxdy.shape)
    
    pi_part = np.zeros(shape=list(dxdy.shape)[:-1], dtype=dxdy.dtype)
    pi_part[np.expand_dims(points_xy[:, :, 0], axis=-1) > points_xy_0[:, 0]] -= np.pi
#     print("9:", pi_part.shape)
    
    big_sum = np.squeeze((np.arctan(dxdy[:, :, :, 1] / dxdy[:, :, :, 0]) + pi_part).dot(np.expand_dims(gamma_coef, axis=-1)) / (2 * np.pi))
    # print("simple big sum shape:", big_sum.shape)
    inf_part = np.squeeze(points_xy.dot(np.expand_dims(V_inf, axis=-1)))
    # print("simple inf part shape:", inf_part.shape)
    return inf_part + big_sum

def calc_psi_simple(points_xy, t, gamma_coef, points_xy_0):
    R_arr = calc_R(points_xy, points_xy_0, delta)
    # print("R_arr shape:", R_arr.shape)
    R_log_arr = np.log(np.squeeze(R_arr))
    big_sum = np.squeeze(R_log_arr.dot(np.expand_dims(gamma_coef, axis=-1)) / (2 * np.pi))
    # print("psi simple big sum shape:", big_sum.shape)
    inf_part = np.squeeze(points_xy.dot(np.expand_dims(V_inf[::-1] * [-1, 1], axis=-1)))
    # print("simple inf part shape:", inf_part.shape)
    return inf_part - big_sum

def calc_phi_dipol(points_xy, t, gamma_coef, points_xy_0):
    dxdy0 = points_xy_0[1:] - points_xy_0[:-1]
    dxdy = np.expand_dims(points_coords, axis=-2) - points_xy_0[:-1]
    R_arr = calc_R(points_xy, points_xy_0, delta)[:, :, :-1]
    R_arr = np.squeeze(R_arr)
    # print("R_arr shape:", R_arr.shape)
    z = (dxdy0[:, 1] * dxdy[:, :, :, 0] - dxdy0[:, 0] * dxdy[:, :, :, 1]) / (R_arr * R_arr)
    # print("z shape:", z.shape)
    gamma_sums = gamma_coef.dot(np.triu(np.ones(shape=(gamma_coef.shape[0], gamma_coef.shape[0]), dtype=np.float32)))
    # print("gamma sums shape:", gamma_sums.shape)
    delta_points = points_xy - points_xy_0[-1]
    big_sum = np.squeeze(z.dot(np.expand_dims(gamma_sums[:-1], axis=-1)) / (2 * np.pi))
    pi_part = np.zeros_like(big_sum)
    pi_part[points_xy[:, :, 0] > points_xy_0[-1, 0]] -= np.pi
    c0 = gamma_0 * (np.arctan(delta_points[:, :, 1] / delta_points[:, :, 0]) + pi_part) / (2 * np.pi)
    # print("delta points shape:", delta_points.shape)
    # print("big sum shape:", big_sum.shape)
    # print("c0 shape:", c0.shape)
    inf_part = np.squeeze(points_xy.dot(np.expand_dims(V_inf, axis=-1)))
    # print("inf part shape:", inf_part.shape)
    return inf_part + big_sum + c0

def calc_psi_dipol(points_xy, t, gamma_coef, points_xy_0):
    dxdy0 = points_xy_0[1:] - points_xy_0[:-1]
    dxdy = np.expand_dims(points_coords, axis=-2) - points_xy_0[:-1]
    R_arr = calc_R(points_xy, points_xy_0, delta)[:, :, :-1]
    R_arr = np.squeeze(R_arr)
    # print("R_arr shape:", R_arr.shape)
    z =  np.sum(dxdy0 * dxdy, axis=3) / (R_arr * R_arr)  # (dxdy0[:, 0] * dxdy[:, :, :, 0] - dxdy0[:, 1] * dxdy[:, :, :, 1]) / (R_arr * R_arr)
    # print("z shape:", z.shape)
    gamma_sums = gamma_coef.dot(np.triu(np.ones(shape=(gamma_coef.shape[0], gamma_coef.shape[0]), dtype=np.float32)))
    print("gamma sums shape:", gamma_sums.shape)
    # delta_points = points_xy - points_xy_0[-1]
    big_sum = np.squeeze(z.dot(np.expand_dims(gamma_sums[:-1], axis=-1)) / (2 * np.pi))
    print("11:", R_arr.shape)
    c0 = gamma_0 * np.sum(np.log(R_arr), axis=-1) / (2 * np.pi)
    # print("delta points shape:", delta_points.shape)
    # print("big sum shape:", big_sum.shape)
    # print("c0 shape:", c0.shape)
    inf_part = np.squeeze(points_xy.dot(np.expand_dims(V_inf[::-1] * [-1, 1], axis=-1)))
    # print("inf part shape:", inf_part.shape)
    return inf_part - big_sum - c0

def calc_Cp(v):
    return 1 - (np.linalg.norm(v, axis=-1) / np.linalg.norm(V_inf))**2

def calc_p(Cp, rho=1.2041):
    return rho * np.linalg.norm(V_inf)**2 * Cp + p_inf

Cp = calc_Cp(V_field)
# print(Cp.shape, Cp.min(), Cp.max())
p = calc_p(Cp)
# print(p.shape, p.min(), p.max())

phi_simple_map = calc_phi_simple(points_coords, 0, gamma_coef, discrete_points)
# print(phi_simple_map.shape)
phi_map = calc_phi_dipol(points_coords, 0, gamma_coef, discrete_points)
# print(phi_map.shape)

V_field.shape

T = np.linspace(0, 1, 1)
X = np.arange(X_1, X_2, delta/2)
Y = np.arange(Y_1, Y_2, delta/2)

fig = plt.figure(figsize=(20,70), num='This is the title')
U = np.zeros(shape=(T.shape[0], X.shape[0], Y.shape[0]), dtype=np.float32)
V = np.zeros(shape=(T.shape[0], X.shape[0], Y.shape[0]), dtype=np.float32)
phi_map = np.zeros(shape=(T.shape[0], X.shape[0], Y.shape[0]), dtype=np.float32)
phi_simple_map = np.zeros(shape=(T.shape[0], X.shape[0], Y.shape[0]), dtype=np.float32)
psi_map = np.zeros(shape=(T.shape[0], X.shape[0], Y.shape[0]), dtype=np.float32)
psi_simple_map = np.zeros(shape=(T.shape[0], X.shape[0], Y.shape[0]), dtype=np.float32)
Cp_map = np.zeros(shape=(T.shape[0], X.shape[0], Y.shape[0]), dtype=np.float32)
p_map = np.zeros(shape=(T.shape[0], X.shape[0], Y.shape[0]), dtype=np.float32)
for i in range(T.shape[0]):
    V_field = calc_V(points_coords, T[i], gamma_coef, discrete_points)
    U[i], V[i] = V_field[:, :, 0], V_field[:, :, 1]
    phi_map[i] = calc_phi_dipol(points_coords, T[i], gamma_coef, discrete_points)
    phi_simple_map[i] = calc_phi_simple(points_coords, T[i], gamma_coef, discrete_points)
    psi_map[i] = calc_psi_dipol(points_coords, T[i], gamma_coef, discrete_points)
    psi_simple_map[i] = calc_psi_simple(points_coords, T[i], gamma_coef, discrete_points)
    Cp_map[i] = calc_Cp(V_field)
    p_map[i] = calc_p(Cp_map[i])

phi_maps = {
    "PHI: OPTION 1": phi_simple_map,
    "PHI: OPTION 2 (DIPOL)": phi_map
}

for plot_num, (title_s, phi_map) in enumerate(phi_maps.items()):
    lo = phi_map.min()
    hi = phi_map.max()
    for i in range(T.shape[0]):
        t = T[i]
        ax = plt.subplot(1, 2, plot_num + 1)
        plot_num += 1
        # plt.axis('off')
        q = ax.quiver(X[::3], Y[::3], U[i, ::3, ::3], V[i, ::3, ::3])

        ax.axis("equal")
        #ax.xticks(np.arange(X_1, X_2, 0.5))
        #ax.yticks(np.arange(Y_1, Y_2, 0.5))
        ax.set_xlabel([X_1, X_2])
        ax.set_ylabel([Y_1, Y_2])

        ax.plot(discrete_points[:, 0], discrete_points[:, 1], 'ro', markersize=2)
        ax.plot(collocation_points[:, 0], collocation_points[:, 1], 'bo', markersize=2)

        im = ax.imshow(phi_map[i, ::-1, :], cmap=cm.coolwarm, extent=[X_1, X_2, Y_1, Y_2])  # drawing the function

        # adding the Contour lines with labels
        # cset = plt.contour(phi_map[i], np.linspace(lo, hi, 20), linewidths=1, cmap=cm.Set3, extent=[X_1, X_2, Y_1, Y_2])
        # plt.clabel(cset, inline=True, fmt='%1.1f', fontsize=10)
        plt.colorbar(im)  # adding the colobar on the right
        # latex fashion title
        plt.title('{}, t = {:.4f}'.format(title_s, t))

plt.show()






T = np.linspace(0, 1, 1)
X = np.arange(X_1, X_2, delta/2)
Y = np.arange(Y_1, Y_2, delta/2)

fig = plt.figure(figsize=(25, 90))
U = np.zeros(shape=(T.shape[0], X.shape[0], Y.shape[0]), dtype=np.float32)
V = np.zeros(shape=(T.shape[0], X.shape[0], Y.shape[0]), dtype=np.float32)
phi_map = np.zeros(shape=(T.shape[0], X.shape[0], Y.shape[0]), dtype=np.float32)
phi_simple_map = np.zeros(shape=(T.shape[0], X.shape[0], Y.shape[0]), dtype=np.float32)
psi_map = np.zeros(shape=(T.shape[0], X.shape[0], Y.shape[0]), dtype=np.float32)
psi_simple_map = np.zeros(shape=(T.shape[0], X.shape[0], Y.shape[0]), dtype=np.float32)
Cp_map = np.zeros(shape=(T.shape[0], X.shape[0], Y.shape[0]), dtype=np.float32)
p_map = np.zeros(shape=(T.shape[0], X.shape[0], Y.shape[0]), dtype=np.float32)
for i in range(T.shape[0]):
    V_field = calc_V(points_coords, T[i], gamma_coef, discrete_points)
    U[i], V[i] = V_field[:, :, 0], V_field[:, :, 1]
    phi_map[i] = calc_phi_dipol(points_coords, T[i], gamma_coef, discrete_points)
    phi_simple_map[i] = calc_phi_simple(points_coords, T[i], gamma_coef, discrete_points)
    psi_map[i] = calc_psi_dipol(points_coords, T[i], gamma_coef, discrete_points)
    psi_simple_map[i] = calc_psi_simple(points_coords, T[i], gamma_coef, discrete_points)
    Cp_map[i] = calc_Cp(V_field)
    p_map[i] = calc_p(Cp_map[i])
    
phi_maps = {
    "PSI: OPTION 1": psi_simple_map,
    "PSI: OPTION 2 (DIPOL)": psi_map
}

for plot_num, (title_s, phi_map) in enumerate(phi_maps.items()):
    lo = phi_map.min()
    hi = phi_map.max()
    for i in range(T.shape[0]):
        t = T[i]
        ax = plt.subplot(1, 2, plot_num + 1)
        plot_num += 1
        # plt.axis('off')
        q = ax.quiver(X[::3], Y[::3], U[i, ::3, ::3], V[i, ::3, ::3])

        ax.axis("equal")
        #ax.xticks(np.arange(X_1, X_2, 0.5))
        #ax.yticks(np.arange(Y_1, Y_2, 0.5))
        ax.set_xlabel([X_1, X_2])
        ax.set_ylabel([Y_1, Y_2])

        ax.plot(discrete_points[:, 0], discrete_points[:, 1], 'ro', markersize=2)
        ax.plot(collocation_points[:, 0], collocation_points[:, 1], 'bo', markersize=2)

        im = ax.imshow(phi_map[i, ::-1, :], cmap=cm.coolwarm, extent=[X_1, X_2, Y_1, Y_2])  # drawing the function

        # adding the Contour lines with labels
        cset = plt.contour(phi_map[i], np.linspace(lo, hi, 20), linewidths=1, cmap=cm.plasma, extent=[X_1, X_2, Y_1, Y_2])
        plt.clabel(cset, inline=True, fmt='%1.1f', fontsize=10)
        plt.colorbar(im)  # adding the colobar on the right
        # latex fashion title
        plt.title('{}, t = {:.4f}'.format(title_s, t))

T = np.linspace(0, 1, 1)
X = np.arange(X_1, X_2, delta/2)
Y = np.arange(Y_1, Y_2, delta/2)

fig = plt.figure(figsize=(25, 90))
U = np.zeros(shape=(T.shape[0], X.shape[0], Y.shape[0]), dtype=np.float32)
V = np.zeros(shape=(T.shape[0], X.shape[0], Y.shape[0]), dtype=np.float32)
phi_map = np.zeros(shape=(T.shape[0], X.shape[0], Y.shape[0]), dtype=np.float32)
phi_simple_map = np.zeros(shape=(T.shape[0], X.shape[0], Y.shape[0]), dtype=np.float32)
psi_map = np.zeros(shape=(T.shape[0], X.shape[0], Y.shape[0]), dtype=np.float32)
psi_simple_map = np.zeros(shape=(T.shape[0], X.shape[0], Y.shape[0]), dtype=np.float32)
Cp_map = np.zeros(shape=(T.shape[0], X.shape[0], Y.shape[0]), dtype=np.float32)
p_map = np.zeros(shape=(T.shape[0], X.shape[0], Y.shape[0]), dtype=np.float32)
for i in range(T.shape[0]):
    V_field = calc_V(points_coords, T[i], gamma_coef, discrete_points)
    U[i], V[i] = V_field[:, :, 0], V_field[:, :, 1]
    phi_map[i] = calc_phi_dipol(points_coords, T[i], gamma_coef, discrete_points)
    phi_simple_map[i] = calc_phi_simple(points_coords, T[i], gamma_coef, discrete_points)
    psi_map[i] = calc_psi_dipol(points_coords, T[i], gamma_coef, discrete_points)
    psi_simple_map[i] = calc_psi_simple(points_coords, T[i], gamma_coef, discrete_points)
    Cp_map[i] = calc_Cp(V_field)
    p_map[i] = calc_p(Cp_map[i])
    
phi_maps = {
    "C_p": Cp_map,
    "P:": p_map
}

for plot_num, (title_s, phi_map) in enumerate(phi_maps.items()):
    lo = phi_map.min()
    hi = phi_map.max()
    for i in range(T.shape[0]):
        t = T[i]
        ax = plt.subplot(1, 2, plot_num + 1)
        plot_num += 1
        # plt.axis('off')
        q = ax.quiver(X[::3], Y[::3], U[i, ::3, ::3], V[i, ::3, ::3])

        ax.axis("equal")
        #ax.xticks(np.arange(X_1, X_2, 0.5))
        #ax.yticks(np.arange(Y_1, Y_2, 0.5))
        ax.set_xlabel([X_1, X_2])
        ax.set_ylabel([Y_1, Y_2])

        ax.plot(discrete_points[:, 0], discrete_points[:, 1], 'ro', markersize=2)
        ax.plot(collocation_points[:, 0], collocation_points[:, 1], 'bo', markersize=2)

        im = ax.imshow(phi_map[i, ::-1, :], cmap=cm.coolwarm, extent=[X_1, X_2, Y_1, Y_2])  # drawing the function

        # adding the Contour lines with labels
        cset = plt.contour(phi_map[i], np.linspace(lo, hi, 20), linewidths=1, cmap=cm.plasma, extent=[X_1, X_2, Y_1, Y_2])
        plt.clabel(cset, inline=True, fmt='%1.1f', fontsize=10)
        plt.colorbar(im)  # adding the colobar on the right
        # latex fashion title
        plt.title('{}, t = {:.4f}'.format(title_s, t))

# fig = plt.figure(figsize=(10, 10))

# mm = 1-(np.linalg.norm(V_field, axis=-1) / np.linalg.norm(V_inf))**2
# print(mm.shape)
# lo = mm.min()
# hi = mm.max()
# im = plt.imshow(-mm[::-1, :], cmap=cm.coolwarm, extent=[X_1, X_2, Y_1, Y_2])  # drawing the function

# # adding the Contour lines with labels
# cset = plt.contour(mm, np.linspace(lo, hi, 20), linewidths=1, cmap=cm.Set3, extent=[X_1, X_2, Y_1, Y_2])
# plt.clabel(cset, inline=True, fmt='%1.1f', fontsize=10)
# plt.colorbar(im)  # adding the colobar on the right
plt.show()




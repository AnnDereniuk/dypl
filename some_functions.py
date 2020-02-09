import numpy as np
GAMMA=1.

#get G/4pi version
def get_velocity_coefs_with_GAMMA(x):
    return (GAMMA/4*3.14)*x

#get vector coords
def get_vector_coords(start, end):
    return  end-start

#get vector length
def vector_length(x_coord, y_coord, z_coord):
    return (x_coord**2+y_coord**2+z_coord**2)**(1/2)

#get squared length
def vector_s_squared_length(x_coord, y_coord, z_coord):
    return vector_length(x_coord, y_coord, z_coord)**3










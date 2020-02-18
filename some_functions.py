import numpy as np
from tkinter import *
GAMMA=1.

#2D:
h = 1   #heidth of the figure 
w = 1   #half-width of the figure

#chosing a 2D point
def choosePoint():
    root = Tk()
    topFrame = Frame(root)
    topFrame.pack()
    bottomFrame=Frame(root)
    bottomFrame.pack(side=BOTTOM)
    
    header = Label (topFrame, text="Please, enter values for x and y and press Enter", bg="green")
    header.pack()
    
    #labels
    labelB = Label(bottomFrame, text="x = ")
    labelB.grid(row=0, column =0)
    labelB = Label(bottomFrame, text="y = ")
    labelB.grid(row=1, column =0)
    
    #entry fields
    entryB = Entry(bottomFrame)
    entryB.grid(row=0, column =1)
    entryB = Entry(bottomFrame)
    entryB.grid(row=1, column =1)
    
    root.mainloop()

def chooseLeftPointX(x):
    return x - w
def chooseLeftPointY(x):
    return x - h

def chooseRightPointX(x):
    return x + w

def chooseRightPointY(x):
    return x - h

def get_step(x1,x2,m):
    return (x2-x1)/m

def get_R(x_coord, y_coord):
    return (x_coord**2+y_coord**2)

def get_vector_length_2d(x_coord, y_coord):
    return np.sqrt(x_coord**2+y_coord**2)

#get G/4pi version
def get_velocity_coefs_with_GAMMA(x):
    return (GAMMA/4.*np.pi)*x

#get vector coords
def get_vector_coords(start, end):
    return  end-start

#get velocity for j index
def get_velocity_j(x, x_j, R):          #R=(x-x_j)^2 +(y-Y_j)^2
    return (1./(2.*np.pi))*((x_j - x)/R)

#get vector length
def vector_length(x_coord, y_coord, z_coord):
    return np.sqrt(x_coord**2+y_coord**2+z_coord**2)

#get squared length
def vector_s_squared_length(x_coord, y_coord, z_coord):
    return vector_length(x_coord, y_coord, z_coord)**3










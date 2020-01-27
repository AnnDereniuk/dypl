import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load in X,Y,Z values from a sub-sample of 10 points for testing
# XY Values are scaled to a reasonable point of origin
z_vals = np.array([3.08,4.46,0.27,2.40,0.48,0.21,0.31,3.28,4.09,1.75])
x_vals = np.array([22.88,20.00,20.36,24.11,40.48,29.08,36.02,29.14,32.20,18.96])
y_vals = np.array([31.31,25.04,31.86,41.81,38.23,31.57,42.65,18.09,35.78,31.78])

# Updated code below
# Variables needed for 2D,3D histograms
xmax, ymax, zmax = int(x_vals.max())+1, int(y_vals.max())+1, int(z_vals.max())+1
xmin, ymin, zmin = int(x_vals.min()), int(y_vals.min()), int(z_vals.min())
xrange, yrange, zrange = xmax-xmin, ymax-ymin, zmax-zmin
xedges = np.linspace(xmin, xmax, (xrange + 1), dtype=int)
yedges = np.linspace(ymin, ymax, (yrange + 1), dtype=int)
zedges = np.linspace(zmin, zmax, (zrange + 1), dtype=int)

# Make the 2D histogram
h2d, xedges, yedges = np.histogram2d(x_vals, y_vals, bins=(xedges, yedges))
assert np.count_nonzero(h2d) == len(x_vals), "Unclassified points in the array"
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
plt.imshow(h2d.transpose(), extent=extent,  interpolation='none', origin='low')
# Transpose and origin must be used to make the array line up when using imshow, unsure why
# Plot settings, not sure yet on matplotlib update/override objects
plt.grid(b=True, which='both')
plt.xticks(xedges)
plt.yticks(yedges)
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.plot(x_vals, y_vals, 'ro')
plt.show()

# 3-dimensional histogram with 1 x 1 x 1 m bins. Produces point counts in each 1m3 cell.
xyzstack = np.stack([x_vals,y_vals,z_vals], axis=1)
h3d, Hedges = np.histogramdd(xyzstack, bins=(xedges, yedges, zedges))
assert np.count_nonzero(h3d) == len(x_vals), "Unclassified points in the array"
h3d.shape  # Shape of the array should be same as the edge dimensions
testzbin = np.sum(np.logical_and(z_vals >= 1, z_vals < 2))  # Slice to test with
np.sum(h3d[:,:,1]) == testzbin  # Test num points in second bins
np.sum(h3d, axis=2)  # Sum of all vertical points above each x,y 'pixel'
# only in this example the h2d and np.sum(h3d,axis=2) arrays will match as no z bins have >1 points

# Remaining issue - how to get a r x c count of empty z bins.
# i.e. for each 'pixel'  how many z bins contained no points?
# Possible solution is to reshape to use logical operators
count2d = h3d.reshape(xrange * yrange, zrange)  # Maintain dimensions per num 3D cells defined
zerobins = (count2d == 0).sum(1)
zerobins.shape
# Get back to x,y grid with counts - ready for output as image with counts=pixel digital number
bincount_pixels = zerobins.reshape(xrange,yrange)
# Appears to work, perhaps there is a way without reshapeing?

# # Setup
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Load in X,Y,Z values from a sub-sample of 10 points for testing
# # XY Values are scaled to a reasonable point of origin
# z_vals = np.array([3.08,4.46,0.27,2.40,0.48,0.21,0.31,3.28,4.09,1.75])
# x_vals = np.array([22.88,20.00,20.36,24.11,40.48,29.08,36.02,29.14,32.20,18.96])
# y_vals = np.array([31.31,25.04,31.86,41.81,38.23,31.57,42.65,18.09,35.78,31.78])

# # This plot is instructive to visualize the problem
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x_vals, y_vals, z_vals, c='b', marker='o')
# plt.show()
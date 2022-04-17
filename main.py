import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

"""
points = np.array([[0, 0], [0, 1],[2,0], [1, 0],[2,1], [1, 1]])
tri = Delaunay(points)

print(points[tri.simplices])
plt.figure()
plt.triplot(points[:,0], points[:,1], tri.simplices)
plt.plot(points[:,0], points[:,1], 'o')
plt.show()
"""
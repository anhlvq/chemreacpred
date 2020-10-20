from numpy import genfromtxt
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import kmedoids

# number of clusters
numclusters = 50
numpatients = 56286

outcenters = "out_center_50.dat"
outlabels = "out_label_50.dat"

# load the data
data = genfromtxt('vectorfix.dat', delimiter=' ')

# sampling data
#data = data[0:20000,:]

# print data to test
print(data)

# distance matrix
D = pairwise_distances(data, metric='euclidean')

print("Start clustering...")

# split into 2 clusters
M, C = kmedoids.kMedoids(D, numclusters)

# print medoids
print('medoids:')
print(M)
np.savetxt(outcenters, M, fmt='%d', delimiter=' ', newline='\n')

# print clusters
print('')
L = np.zeros([numpatients, 2])
print('clustering result:')
for label in C:
    for point_idx in C[label]:
        print('label {0}:　{1}'.format(label, point_idx))
        L[point_idx][0] = point_idx;
        L[point_idx][1] = label
np.savetxt(outlabels, L, fmt='%d', delimiter=' ', newline='\n')




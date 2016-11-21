# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 16:11:40 2016

@author: Chris
"""

from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
from simsearch import SimSearch
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-built corpus.
print('Loading the saved SimSearch and corpus...')
ssearch = SimSearch.load(save_dir='./mhc_corpus/')

# Get the dataset to be clustered.
# Note - The index is store with all of the vectors *already normalized*.
X = ssearch.index.index

# If you needed to normalize the vectors:
# norms = np.linalg.norm(X)
# norms = norms.reshape(-1, 1)
# X = X / norms

# These lists will store the actual values to plot.
plotx = []
ploty1 = []
ploty2 = []

print('Calculating avg(x^2)')
tss = sum(pdist(X)**2)/X.shape[0]

for k in range(10, 200, 10):
    
    plotx.append(k)    
    
    # Create the KMeans object        
    kmeans_model = KMeans(n_clusters=k, init='k-means++', max_iter=100)
    
    # Run clustering.
    print 'K-Means with', k, 'clusters...'
    kmeans_model.fit(X)
    
    # Get the resulting centroids
    centroids = kmeans_model.cluster_centers_
    
    # Calculate the distances matrix between all data points and the final
    # centroids.
    # TODO - I changed this to squared euclidean, let's see if it does any better?
    D = cdist(X, centroids, 'sqeuclidean')
    
    # For each data point, find the minimum distance value (the distance to the
    # closest centroid)
    # This could also be done using the '.labels_' object from kmeans...    
    dist = np.min(D, axis=1)
    
    # Calculate the average distance
    # TODO - This is really the average member-centroid L2 distance...
    avgWithinSS = sum(dist) / X.shape[0]
    
    # Append it to the plot.
    ploty1.append(avgWithinSS)    
    
    # Total with-in sum of square
    wcss = sum(dist**2)

    bss = tss - wcss / tss * 100
    
    ploty2.append(bss)

###############################################################################
# Find the point with the highest second derivative.
dx = []
for i in range(0, len(plotx) - 1):
    dxi = (ploty1[i + 1] - ploty1[i]) / (plotx[i + 1] - plotx[i])
    dx.append(dxi)

ddx = []
for i in range(0, len(dx) - 1):
    ddxi = (dx[i + 1] - dx[i]) / (plotx[i + 1] - plotx[i])
    # Append the absolute value of the second derivative.
    ddx.append(abs(ddxi))

# Get the index of the point with the highest second derivative.
# This is given by the index of the highest second derivative value, plus 1,
# see the example below:
# Points   0   1   2   3   4
#     dx     0   1   2   3
#    ddx       0   1   2
    
kIdx = np.argmax(ddx) + 1


###############################################################################
# Plot the elbow curve.
#

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(plotx, ploty1, 'b*-')
ax.plot(plotx[kIdx], ploty1[kIdx], marker='o', markersize=12, 
markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares')
plt.title('Elbow for KMeans clustering')

###############################################################################
# Plot the explained variance elbow.
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(plotx, ploty2, 'b*-')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Percentage of variance explained')
plt.title('Elbow for KMeans clustering')
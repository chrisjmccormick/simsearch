# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 21:11:49 2016

@author: Chris
"""
from simsearch import SimSearch

from corpusbuilder import CorpusBuilder

#def depression(ksearch, ssearch):
    # 'Depression' is not in the corpus
    #ksearch.getIDForWord('depression')
    
    # Depressed is in the corpus.
    #ksearch.getIDForWord('depressed')
    
    # Look for entries with 'depressed'.
    #results = ksearch.keywordSearch(includes=['depressed'])
    
    # Print out the first document. 
    # It's the wrong meaning of depressed, "none may be either too much
    # elevated or too much depressed."
    #ksearch.printDocSourcePretty(results[0], max_lines=100)

    # Print out the second document. 
    # Again, wrong meaning: "now as much depressed as he had been elevated"
    #ksearch.printDocSourcePretty(results[1], max_lines=100)
    
    # Try again, but exclude elevated. This returned a result which 
    # gave me ideas for synonyms: dejected, discouraged.
    # results = ksearch.keywordSearch(includes=['depressed'], excludes=['elevated']) 

#print('Loading the saved SimSearch and corpus...')
#(ksearch, ssearch) = SimSearch.load(save_dir='./mhc_corpus/')

sub_patterns = [('“', ' '),
                ('”', ' ')]

# Match blank lines as the separator between "documents".    
doc_start_pattern = r'^\s*$'

# Create the CorpusBuilder.
cb = CorpusBuilder(stop_words_file='stop_words.txt', sub_patterns=sub_patterns, 
                   doc_start_pattern=doc_start_pattern, 
                   doc_start_is_separator=True)

print 'Parsing Pride and Prejudice...'    
# Parse all of the text files in the directory.
cb.addDirectory('./books/')

print 'Done.'

print 'Building corpus...'

cb.buildCorpus()

# Initialize a KeySearch object from the corpus.
ksearch = cb.toKeySearch()

# Print the top 30 most common words.
ksearch.printTopNWords(topn=30)

print '\nVocabulary contains', ksearch.getVocabSize(), 'unique words.'

print 'Corpus contains', len(ksearch.corpus_tfidf), '"documents" represented by tf-idf vectors.'

# Initialize a SimSearch object from the KeySearch.
ssearch = SimSearch(ksearch)

# Train LSI with 100 topics.
print '\nTraining LSI...'
ssearch.trainLSI(num_topics=100)

print '\nSaving to disk...'
if not exists('./pnp_corpus/'):
    makedirs('./pnp_corpus/')

ssearch.save(save_dir='./pnp_corpus/')

print 'Done!'

from sklearn.cluster import KMeans

# Number of clusters to find...
k = 10    

print('Clustering all documents in corpus...')
# Initialize a k-means clustering object.
km = KMeans(n_clusters=k, init='k-means++', max_iter=100)

# Cluster the LSI vectors.     
km.fit(ssearch.index.index)

# For each of the clusters...    
for cluster_id in range(0, k):
    
    # Get the list of all doc IDs belonging to this cluster.
    cluster_doc_ids = []
    for doc_id in range(0, len(km.labels_)):            
        if km.labels_[doc_id] == cluster_id:
            cluster_doc_ids.append(doc_id)

    # Get the top words in this cluster
    top_words = ssearch.getTopWordsInCluster(cluster_doc_ids)

    print('  Cluster %d: %s' % (cluster_id, " ".join(top_words)))

###############################################################################
# Elbow method on Pride & Prejudice

from scipy.spatial.distance import cdist, pdist
import numpy as np
import matplotlib.pyplot as plt



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

for k in range(1, 40, 2):
    
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

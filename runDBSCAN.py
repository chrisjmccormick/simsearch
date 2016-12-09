# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 11:19:04 2016

@author: Chris
"""

from simsearch import SimSearch
from sklearn.cluster import DBSCAN
import sklearn
import time



from sklearn.neighbors import NearestNeighbors


def findEps(ssearch):
    """
    Find a good epsilon value to use.
    """
    ###########################################################################
    # Calculate nearest neighbors
    ###########################################################################
    
    # Create a nearest neighbors model--we need 2 nearest neighbors since the 
    # nearest neighbor to a point is going to be itself.
    nbrs_model = NearestNeighbors(n_neighbors=2, algorithm='brute', metric='cosine').fit(ssearch.index.index)
    
    t0 = time.time()
    
    # Find nearest neighbors.
    distances, indices = nbrs_model.kneighbors(ssearch.index.index)
    
    elapsed = time.time() - t0
    
    print 'Took %.2f seconds' % elapsed
    
    distances = [d[1] for d in distances]
    indeces = [ind[1] for ind in indices]
    
    ###########################################################################
    # Histogram the nearest neighbor distances.
    ###########################################################################
    
    import matplotlib.pyplot as plt
    
    counts, bins, patches = plt.hist(distances, bins=16)
    plt.title("Nearest neighbor distances")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    
    print '\n%d bins:' % len(counts)
    
    countAcc = 0
    num_points = len(ssearch.index.index)
    
    for i in range(0, len(counts)):
        countAcc += counts[i]
        
        # Calculate the percentage of values which fall below the upper limit 
        # of this bin.
        prcnt = float(countAcc) / float(num_points) * 100.0    
        
        print '  %.2f%% < %.2f' % (prcnt, bins[i + 1])


def findMinPts(ssearch, eps):
    """
    Find a good value for MinPts.
    """
    
    ###########################################################################
    # Count neighbors within threshold
    ###########################################################################
    
    print 'Calculating pair-wise distances...'
    # Calculate pair-wise cosine distance for all documents.
    t0 = time.time()
    
    DD = sklearn.metrics.pairwise.cosine_distances(ssearch.index.index)
    
    elapsed = time.time() - t0
    
    print '    Took %.2f seconds' % elapsed
    
    print 'Counting number of neighbors...'
    
    t0 = time.time()
    
    # Create a list to hold the number of neighbors for each point.
    numNeighbors = [0]*len(DD)
    
    for i in range(0, len(DD)):
        dists = DD[i]
        
        count = 0
        for j in range(0, len(DD)):
            if (dists[j] < eps):
                count += 1
    
        numNeighbors[i] = count            
    
    elapsed = time.time() - t0
    
    print '    Took %.2f seconds' % elapsed
    
    ###############################################################################
    # Histogram the nearest neighbor distances.
    ###############################################################################
    
    import matplotlib.pyplot as plt
    
    counts, bins, patches = plt.hist(numNeighbors, bins=60)
    plt.title("Number of neighbors")
    plt.xlabel("Number of neighbors")
    plt.ylabel("Frequency")
    
    print '\n%d bins:' % (len(bins) - 1)
    binsStr = ''
    for b in bins:
        binsStr += '  %0.2f' % b
    
    print binsStr


def runClustering(ssearch, eps, min_samples):
    """
    Run DBSCAN with the determined eps and MinPts values.
    """
    print('Clustering all documents with DBSCAN, eps=%0.2f min_samples=%d' % (eps, min_samples))
    
    # Initialize DBSCAN with parameters.
    # I forgot to use cosine at first!
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine', algorithm='brute')
    
    # Time this step.
    t0 = time.time()
    
    # Cluster the LSI vectors.     
    db.fit(ssearch.index.index)
    
    # Calculate the elapsed time (in seconds)
    elapsed = (time.time() - t0)
    print("  done in %.3fsec" % elapsed)
    
    # Get the set of unique IDs.
    cluster_ids = set(db.labels_)
    
    # Show the number of clusters (don't include noise label)
    print('Number of clusters (excluding "noise"): %d' % (len(cluster_ids) - 1))  
     
    # For each of the clusters...    
    for cluster_id in cluster_ids:
            
            # Get the list of all doc IDs belonging to this cluster.
            cluster_doc_ids = []
            for doc_id in range(0, len(db.labels_)):            
                if db.labels_[doc_id] == cluster_id:
                    cluster_doc_ids.append(doc_id)
    
            # Get the top words in this cluster
            top_words = ssearch.getTopWordsInCluster(cluster_doc_ids)
    
            print('  Cluster %d: (%d docs) %s' % (cluster_id, len(cluster_doc_ids), " ".join(top_words)))
        

def main():   
    """
    Entry point for the script.
    """
    
    ###########################################################################
    # Load the corpus
    ###########################################################################
    
    # Load the pre-built corpus.
    print('Loading the saved SimSearch and corpus...')
    (ksearch, ssearch) = SimSearch.load(save_dir='./mhc_corpus/')
    
    print '    %d documents.' % len(ssearch.index.index)

    # Step 1: Run a technique to find a good 'eps' value.
    #findEps(ssearch)
    #eps = 0.5
    eps = 0.44

    # Step 2: Run a technique to find a good 'MinPts' value.    
    # TODO - This took ~17 min. on my desktop!
    #findMinPts(ssearch, eps)
    #min_samples = 8
    min_samples = 4

    # Step 3: Run DBSCAN
    runClustering(ssearch, eps, min_samples)

main()







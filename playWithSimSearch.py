# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 16:29:31 2016

@author: Chris McCormick
"""

from simsearch import SimSearch
from sklearn.cluster import KMeans

def runExample1(ksearch, ssearch):
    """
    ======== Example 1 ========
    Find documents similar to 'document' number 73, which is mhc1.txt lines 
    1617 - 1647. This is commentary on the seventh day of creation, when God 
    rested. The top match is commentary on the fourth commandment--to obey the 
    sabbath (Exodus Chapter 20).
    """
    
    print('Searching for docs similar to document number 73...')
    print('')
    
    # Display the source document.
    print('Input - (Doc 73):')
    ksearch.printDocSourcePretty(doc_id=73, max_lines=5)
    
    print('')
    
    # Perform the search
    results = ssearch.findSimilarToDoc(doc_id=73, topn=1)
    
    # Print the top results
    ssearch.printResultsBySourceText(results, max_lines=8)
    
    # Retrieve the tf-idf vectors for the input document and it's closest match.
    vec1_tfidf = ksearch.getTfidfForDoc(73)
    vec2_tfidf = ksearch.getTfidfForDoc(results[0][0])

    print('')    
    
    # Interpret the top match.
    ssearch.interpretMatch(vec1_tfidf, vec2_tfidf)

def runExample2(ksearch, ssearch):
    """
    ======== Example 2 ========
    Clusters the entire corpus using k-means, then displays the top words
    for each cluster.
    
    Some fun results that I've seen:
      Theme of money:
        "poor rich charity man riches good poverty wealth money christ"
      Theme of sin:
        "sin sins god repentance sinners iniquity evil ruin judgments 
         punishment"
      Theme of Christ's resurrection:
        "death life dead die body christ resurrection live grave soul"
    """
    
    # Number of clusters to find...
    k = 20    
    
    print('Clustering all documents in corpus...')
    # Initialize a k-means clustering object.
    # Note: There is no 'cosine' metric for k-means. But if the vectors are
    # all normalized (they are) than Euclidean is equivalent.
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

        print('  Cluster %d: (%d docs) %s' % (cluster_id, len(cluster_doc_ids), " ".join(top_words)))


# Load the pre-built corpus.
print('Loading the saved SimSearch and corpus...')
(ksearch, ssearch) = SimSearch.load(save_dir='./mhc_corpus/')

print('\n======== Example 1 ========\n')
runExample1(ksearch, ssearch)
print('\n======== Example 2 ========\n')
runExample2(ksearch, ssearch)

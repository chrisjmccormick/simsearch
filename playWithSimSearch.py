# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 16:29:31 2016

@author: Chris
"""

from simsearch import SimSearch

# Load the pre-built corpus.
print 'Loading the saved SimSearch and corpus...'
ssearch = SimSearch.load(save_dir='./mhc_corpus/')


# Find documents similar to 'document' number 73, which is mhc1.txt lines 1617
# - 1647. This is commentary on the seventh day of creation, when God rested.
# The top match is commentary on the fourth commandment--to obey the sabbath.
# (Exodus Chapter 20).
print ''
print 'Searching for docs similar to document number 73...'
print ''

# Display the source document.
print 'Doc 73:'
lines = ssearch.cb.printDocSourcePretty(doc_id=73, max_lines=5)

print ''

# Perform the search
results = ssearch.findSimilarToDoc(doc_id=73, topn=1)

# Print the top results
ssearch.printResultsBySourceText(results, max_lines=8)

# Retrieve the tf-idf vectors for the input document and it's closest match.
vec1_tfidf = ssearch.cb.getDocTfidfVector(73)
vec2_tfidf = ssearch.cb.getDocTfidfVector(results[0][0])

# Interpret the top match.
ssearch.interpretMatch(vec1_tfidf, vec2_tfidf)


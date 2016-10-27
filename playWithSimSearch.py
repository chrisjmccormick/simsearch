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

results = ssearch.findSimilarToDoc(doc_id=73, topn=10)

# Print the results.
ssearch.printResultsByLineNumbers(results)


print ''
print 'Searching for docs similar to some input text...'

# Find documents similar to some text.
# A quote from St. Augustine... The search results for this don't seem 
# very relevant...
ssearch.findSimilarToText('You have made us for yourself, O Lord, and our hearts are restless until they rest in you.')

# Print the results.
ssearch.printResultsByLineNumbers(results)
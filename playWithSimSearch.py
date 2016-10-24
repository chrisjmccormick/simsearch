# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 16:29:31 2016

@author: Chris
"""

from simsearch import SimSearch

# Load the pre-built corpus.
print 'Loading the saved SimSearch...'
ssearch = SimSearch.load(save_dir='./mhc_corpus/')

# Find documents similar to 'document' number 100.
# Entry 100 is commentary on the seventh day of creation, when God rested.
# The top match is commentary on the fourth commandment--to obey the sabbath.
# (Exodus Chapter 20).
ssearch.findSimilarToDoc(doc_id=100, topn=10)

# Find documents similar to some text.
# A quote from St. Augustine...
ssearch.findSimilarToText('You have made us for yourself, O Lord, and our hearts are restless until they rest in you.')

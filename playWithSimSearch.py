# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 16:29:31 2016

@author: Chris
"""

from corpusbuilder import CorpusBuilder

cb = CorpusBuilder()

# Load the pre-built corpus.
print 'Loading the saved corpus...'
cb.load(save_dir='./mhc_corpus/')

# Initialize a SimSearch object from the corpus.
ssearch = cb.toSimSearch()

# Train LSI with 100 topics.
print 'Training LSI...'
ssearch.trainLSI(num_topics=100)

# Find documents similar to entry number 100.
# Entry 100 is commentary on the seventh day of creation, when God rested.
# The top match is commentary on the fourth commandment (Exodus Chapter 20).
ssearch.findSimilarToEntry(entry_id=100, topn=10)
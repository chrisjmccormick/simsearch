# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 16:29:31 2016

@author: Chris
"""

from corpusbuilder import CorpusBuilder

cb = CorpusBuilder()

# Load the pre-built corpus.
cb.load(save_dir='./mhc_corpus/')

# Initialize a SimSearch object from the corpus.
ssearch = cb.toSimSearch()

ssearch.findSimilarToEntry(10)
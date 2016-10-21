# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 21:00:58 2016

@author: Chris
"""

from gensim.models import LsiModel
from gensim import similarities
from corpusbuilder import CorpusBuilder
import numpy as np

class SimSearch(object):
    
    
    def __init__(self, corpus_builder):
        """
        Initialize the SimSearch with a CorpusBuilder object, which holds
        the processed and vectorized corpus.
        
        """        
        self.cb = corpus_builder
    
    def initWithoutCB(self, corpus_tfidf=None, titles=None, dictionary=None, tagsToEntries=None):
        """
        SimSearch is generally intented to be used along with a CorpusBuilder,
        but it is also possible to provide the required objects separately.
        TODO - This needs to be implemented.
        """        
        #self.corpus_tfidf = corpus_tfidf
        #self.titles = titles
        #self.dictionary = dictionary
        #self.tagsToEntries = tagsToEntries
        

    def trainLSI(self, num_topics=100):
        """
        Train the Latent Semantic Indexing model.
        """
        self.num_topics = num_topics        
        # Train LSA
        
        # Look-up the number of features in the tfidf model.
        #self.num_tfidf_features = max(self.corpus_tfidf.dfs) + 1        
        
        self.lsi = LsiModel(self.cb.corpus_tfidf, num_topics=self.num_topics, id2word=self.cb.dictionary)   
    
        # Transform corpus to LSI space and index it
        self.index = similarities.MatrixSimilarity(self.lsi[self.cb.corpus_tfidf], num_features=num_topics) 
    
    
    def findSimilarToVector(self, input_tfidf, topn=10, in_corpus=False, verbose=True):
        """
        Find documents in the corpus similar to the provided document, 
        represented by its tf-idf vector 'input_tfidf'.
        """
        
        # Find the most similar entries to the input tf-idf vector.
        #  1. Project it onto the LSI vector space.
        #  2. Compare the LSI vector to the entire collection.
        sims = self.index[self.lsi[input_tfidf]]        
        
        # Sort the similarities from largest to smallest.
        # 'sims' becomes a list of tuples of the form: 
        #    (entry_id, similarity_value)
        sims = sorted(enumerate(sims), key=lambda item: -item[1])   

        # Select just the top N results.
        # If the input vector exists in the corpus, skip the first one since
        # this will just be the document itself.
        if in_corpus:        
            # Select just the top N results, skipping the first one.
            results = sims[1:1 + topn]    
        else:
            results = sims[0:topn] 
        
        # Print each of the results.
        if verbose:            
            print 'Most similar documents:'
            for i in range(0, len(results)):
                # Print the similarity value followed by the entry title.
                print '  %.2f    %s' % (results[i][1], self.cb.titles[results[i][0]])
            
        return results
    
    
    def findSimilarToEntry(self, entry_id, topn=10, verbose=True):
        """
        Find documents similar to the specified entry number in the corpus.
        
        This will not return the input document in the results list.
        
        Returns the results as a list of tuples in the form:
            (entry_id, similarity_value)
        """
        
        # Find the most similar entries to 'entry_id'
        #  1. Look up the tf-idf vector for the entry.
        #  2. Project it onto the LSI vector space.
        #  3. Compare the LSI vector to the entire collection.
        tfidf_vec = self.cb.corpus_tfidf[entry_id]
        
        # Pass the call down, specifying that the input is a part of the 
        # corpus.
        return self.findSimilarToVector(tfidf_vec, topn=topn, in_corpus=True, verbose=verbose)
    
    def findSimilarToText(self, text, topn=10, verbose=True):
        """
        Find documents in the corpus similar to the provided input text.

        `text` should be a single string. It will be parsed, tokenized, and
        converted to a tf-idf vector by the CorpusBuilder following the same
        procedure that was used to process the corpus.
        
        Returns the results as a list of tuples in the form:
            (entry_id, similarity_value)
        """
        # Parse the input text and create a tf-idf representation.        
        tfidf_vec = self.cb.newTextToTfidfVector(text)
        
        # Pass the call down.        
        return self.findSimilarToVector(tfidf_vec, topn=topn, in_corpus=False, verbose=verbose)
        
        
    def findMoreOfTag(self, tag, topn=10, verbose=True):
        """
        Find entries in the corpus which are similar to those tagged with 
        'tag'. That is, find more entries in the corpus that we might want to
        tag with 'tag'.

        """
        
        # All tags should be lower case to avoid mistakes.
        tag = tag.lower()        
        
        # I pre-pend a '!' to indicate that a journal entry does not belong under
        # a specific tag (I do this to create negative samples)
        if ('!' + tag) in self.cb.tagsToEntries:
            exclude_ids = set(self.cb.tagsToEntries['!' + tag])
        else:
            exclude_ids = set()
        
        # Find all journal entries marked with 'tag'.
        input_ids = self.cb.tagsToEntries[tag]
        
        if verbose:
            print '\nMost similar documents to "' + tag + '":'
            print '\nInput documents:'
        
        # Calculate the combined similarities for all input vectors.
        sims_sum = []
        
        for i in input_ids:
            # Get the LSI vector for this journal.    
            input_vec = self.lsi[self.cb.corpus_tfidf[i]]
        
            print '  ' + self.cb.titles[i]
        
            # Calculate the similarities between this and all other entries.
            sims = self.index[input_vec]
        
            # Accumulate the similarities across all input vectors.
            if len(sims_sum) == 0:
                sims_sum = sims
            else:
                sims_sum = np.sum([sims, sims_sum], axis=0)
                    
        # Sort the combined similarities.
        sims_sum = sorted(enumerate(sims_sum), key=lambda item: -item[1])    
        
        
        print '\nResults:'

        results = []        
        
        shown = 0
        for i in range(0, len(sims_sum)):
            entry_id = sims_sum[i][0]    
        
            # If the result is not one of the inputs, and not a negative sample,
            # show it.
            if entry_id not in input_ids and entry_id not in exclude_ids:
                
                results.append(sims_sum[i])
                print '  %.2f    %s' % (sims_sum[i][1], self.cb.titles[sims_sum[i][0]])
                shown = shown + 1
            
            # Stop when we've displayed 'topn' results.
            if shown == topn:
                break

        return results
        
    def save(self, save_dir='./'):
        """
        Save this SimSearch object to disk for later use.
        
        This also saves the underlying CorpusBuilder object to disk.
        """

        # Save the LSI model and the LSI index.        
        self.index.save(save_dir + 'index.mm')
        self.lsi.save(save_dir + 'lsi.model')

        # Save the underlying CorpusBuilder as well.        
        self.cb.save(save_dir)
        
    @classmethod
    def load(cls, save_dir='./'):
        """
        Load a SimSearch object from the specified directory.
        """
        # First create and load the underlying CorpusBuilder.
        cb = CorpusBuilder()
        cb.load(save_dir)        

        ssearch = SimSearch(cb)        
        
        # Load the LSI index.
        ssearch.index = similarities.MatrixSimilarity.load(save_dir + 'index.mm')
        
        # Load the LSI model.
        ssearch.lsi = LsiModel.load(save_dir + 'lsi.model')
        
        return ssearch        
        
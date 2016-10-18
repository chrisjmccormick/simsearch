# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 21:19:15 2016

@author: Chris
"""

import nltk
from gensim import corpora
from gensim.models import TfidfModel
from collections import defaultdict
from simsearch import SimSearch
import pickle

class CorpusBuilder(object):
    
    def __init__(self, stop_words_file='stop_words.txt', enc_format='utf-8'):
        """
        
        
        """
        # This list holds all of the documents as lists of words.
        self.documents = []
        
        # Create mappings for the entry tags.
        self.tagsToEntries = {}
        self.entriesToTags = []

        # Count the occurrences of each word and store in 'frequency'.        
        self.frequency = defaultdict(int)        
        
        # Read in all of the stop words (one per line) and store them as a set.
        # The call to f.read().splitlines() reads the lines without the newline
        # char.        
        with open(stop_words_file) as f:
            self.stoplist = set(f.read().splitlines())
            
        self.enc_format = enc_format
        
    def addDocument(self, title, lines, tags):
        """
        Add a document to this corpus.
        
        Parameters:
            title - String title of this document.
            lines - List of strings representing the document.
            tags - List of tags, each tag is a separate string.
        """

        doc = []        
        
        entryID = len(self.documents)        

        lineNum = 0        
        
        for line in lines:
            
            lineNum += 1
            
            # Decode the string into Unicode so the NLTK can handle it.
            try:    
                line = line.decode(self.enc_format)        
            except:
                print 'Failed to decode line', lineNum, ': ', line
                raise
            
            # If the string ends in a newline, remove it.
            line = line.replace('\n', '')

            # Convert everything to lowercase, then use NLTK to tokenize.
            tokens = nltk.word_tokenize(line.lower())
           
            # Remove stop words.
            tokens = [word for word in tokens if word not in self.stoplist]
            
            # Tally the occurrences of each word.
            for token in tokens:
                self.frequency[token] += 1            

            # Add these tokens to the list of tokens in the document.
            doc = doc + tokens
         
        # Add this document to the list of all documents.
        self.documents.append(doc)
        
        # Store the list of tags for this journal entry.
        self.entriesToTags.append(tags)                         
            
        # Add mappings from the tags to this journal entry.
        for tag in tags:
            # Convert tags to lower case.        
            tag = tag.lower()
                
            # Add the tag to the dictionary.
            if tag in self.tagsToEntries:
                self.tagsToEntries[tag].append(entryID)
            else:
                self.tagsToEntries[tag] = [entryID]    
    
   
    def buildCorpus(self):
        """
        
        """
        # Remove words that only appear once.
        self.documents = [[token for token in doc if self.frequency[token] > 1]
                          for doc in self.documents]
        
        print 'Building dictionary...'
        
        # Build a dictionary from the text.
        self.dictionary = corpora.Dictionary(self.documents)
        
        print 'Mapping to vector space (creating corpus)...'
        # Map the documents to vectors.
        corpus = [self.dictionary.doc2bow(text) for text in self.documents]

        # Convert the simple bag-of-words vectors to a tf-idf representation.        
        self.corpus_tfidf = TfidfModel(corpus)
        
        # Get the dictionary as a list of tuples.
        # The tuple is (word_id, count)
        word_counts = [(key, value) for (key, value) in self.dictionary.dfs.iteritems()]
        
        # Sort the list by the 'value' of the tuple (incidence count) 
        from operator import itemgetter
        word_counts = sorted(word_counts, key=itemgetter(1))
        
        # Print the most common words.
        # The list is sorted smallest to biggest, so...
        topn = 30
        print 'Top', topn, 'most frequent words'
        for i in range(-1, -topn, -1):
            print '  ', self.dictionary[word_counts[i][0]], '    ', word_counts[i][1]
    

        
    def save(self, save_dir='./'):
        """
        Write out the built corpus to a save directory.
        """
        # Store the tag tables.
        pickle.dump((self.tagsToEntries, self.entriesToTags), open(save_dir + "tag-tables.pickle", "wb"))
        
        # Write out the corpus.
        self.corpus_tfidf.save(save_dir + 'documents.tfidf_model')
        #corpora.MmCorpus.serialize(save_dir + 'journals.mm', self.corpus_tfidf)  # store to disk, for later use

        self.dictionary.save(save_dir + 'documents.dict')  # store the dictionary, for future reference        
        
    def load(save_dir='./'):
        """
        Load the corpus from a save directory.
        """
        
    def toSimSearch(self):
        """
        Initialize a SimSearch object using the data in this corpus.
        """        
        return SimSearch(corpus_tfidf=self.corpus_tfidf, titles=self.titles, 
                         dictionary=self.dictionary, 
                         tagsToEntries=self.tagsToEntries)
        
        

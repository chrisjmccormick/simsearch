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
    """
    The CorpusBuilder object helps turn a collection of plain text documents 
    into a gensim corpus (that is, a collection of vectors representing the
    documents).  
    
    To use the CorpusBuilder with your documents, you will need to:
      * Convert your documents to a plain text representation (copy and paste
        into notepad works well enough).
      * Provide a representative "title" for each document, which will be used
        to refer to the document, e.g., in search results.
      * If you want to create separate vectors for different paragraphs or 
        sections of your documents, you will need to write code to split them
        up.
    
    The CorpusBuilder will tokenize the documents for you using NLTK, so you
    do not need to remove punctuation, whitespace, etc.
    
    The intended useage is as follows:
        1. Create a CorpusBuilder object.
        2. Call `addDocument` for each doc or piece of text in your corpus.
        3. Call `buildCorpus` to build the corpus.
        4. Call `save` to write it out to disk 
        5. Call `toSimSearch` to initialize a SimSearch object and begin 
           performing similarity searches on the corpus.
    
    The `addDocument` step will convert all characters to lowercase, tokenize
    your document with NLTK, filter stop words, and gather word frequency 
    information.
    
    The `buildCorpus` step takes the final collection of documents (now 
    represented as filtered lists of tokens), removes words that only occur 
    once, builds the dictionary, then converts the documents into tf-idf 
    vectors. 
    
    Once the corpus has been built, you cannot call `addDocument`.
    
    The finalized corpus can be saved to or loaded from disk with `save` and 
    `load`.
    
    A finalized corpus can be used to initialize a SimSearch object (for 
    performing similarity searches on the corpus) by calling `toSimSearch`.
    
    """
    def __init__(self, stop_words_file='stop_words.txt', enc_format='utf-8'):
        """
        `stop_words_file` is the path and filename to a file containing stop
        words to be filtered from the input text. There should be one token
        per line in this file.
        
        `enc_format` is the encoding format of the input text documents.        
        """
        # This list holds all of the documents as lists of words.
        self.documents = []
        self.titles = []        
        
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
        
    def addDocument(self, title, lines, tags=[]):
        """
        Add a document (or piece of text) to this corpus. The text is 
        represented by a list of strings. It's ok if the strings contain
        newlines and other whitespace.
        
        Provide a `title` to be displayed for this document when it is returned
        as a search result. Titles are not required to be unique. 
        
        `addDocument` Performs the following steps:
            1. Record document title and tags.
            2. Convert all letters to lowercase.
            2. Tokenize the document using NLTK.
            3. Filter stop words.
            4. Accumulate word frequency information.
                    
        Parameters:
            title - String title of this document.
            lines - List of strings representing the document.
            tags - Optional list of tags, each tag is a separate string.
        """

        # Do not call `addDocument` after the corpus has been built.
        assert(not hasattr(self, 'dictionary'))

        # Get the ID (index) of this document.
        docID = len(self.documents)        

        # Store the title.
        self.titles.append(title)

        # Store the list of tags for this journal entry.
        self.entriesToTags.append(tags)                         
            
        # Add mappings from the tags to this journal entry.
        for tag in tags:
            # Convert tags to lower case.        
            tag = tag.lower()
                
            # Add the tag to the dictionary.
            if tag in self.tagsToEntries:
                self.tagsToEntries[tag].append(docID)
            else:
                self.tagsToEntries[tag] = [docID]   


        # Parse the document into a list of tokens.
        doc = []        
        lineNum = 0        
        
        # Parse each line in the document.
        for line in lines:
            
            lineNum += 1
            
            # Decode the string into Unicode so the NLTK can handle it.
            try:    
                line = line.decode(self.enc_format)        
            except:
                print 'Failed to decode line', lineNum, ': ', line
                raise
            
            # If the string ends in a newline, remove it.
            line = line.replace('\n', ' ')

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
    
   
    def buildCorpus(self):
        """
        Build the corpus from the documents:
            1. Remove words that only appeared once.
            2. Create the Dictionary object.
            3. Convert the documents to simple bag-of-words representation.
            4. Convert the bag-of-words vectors to tf-idf.
        """
        # Remove words that only appear once.
        self.documents = [[token for token in doc if self.frequency[token] > 1]
                          for doc in self.documents]
        
        # Build a dictionary from the text.
        self.dictionary = corpora.Dictionary(self.documents)
        
        # Map the documents to vectors.
        corpus = [self.dictionary.doc2bow(text) for text in self.documents]

        # Convert the simple bag-of-words vectors to a tf-idf representation.        
        self.tfidf_model = TfidfModel(corpus)
        self.corpus_tfidf = self.tfidf_model[corpus]
        
        
    def printTopNWords(self, topn=10):
        """
        Print the 'topn' most frequent words in the corpus.
        
        This is useful for checking to see if you have any common, bogus tokens
        that need to be filtered out of the corpus.
        """
        
        # Get the dictionary as a list of tuples.
        # The tuple is (word_id, count)
        word_counts = [(key, value) for (key, value) in self.dictionary.dfs.iteritems()]
        
        # Sort the list by the 'value' of the tuple (incidence count) 
        from operator import itemgetter
        word_counts = sorted(word_counts, key=itemgetter(1))
        
        # Print the most common words.
        # The list is sorted smallest to biggest, so...
        print 'Top', topn, 'most frequent words'
        for i in range(-1, -topn, -1):
            print '  %s   %d' % (self.dictionary[word_counts[i][0]].ljust(10), word_counts[i][1])
    
    def getVocabSize(self):
        """
        Returns the number of unique words in the final vocabulary (after all
        filtering).
        """
        return len(self.dictionary.keys())
        
    def save(self, save_dir='./'):
        """
        Write out the built corpus to a save directory.
        """
        # Store the tag tables.
        pickle.dump((self.tagsToEntries, self.entriesToTags), open(save_dir + 'tag-tables.pickle', 'wb'))
        
        # Store the document titles.
        pickle.dump(self.titles, open(save_dir + 'titles.pickle', 'wb'))
        
        # Write out the tfidf model.
        self.tfidf_model.save(save_dir + 'documents.tfidf_model')
        
        # Write out the tfidf corpus.
        corpora.MmCorpus.serialize(save_dir + 'documents_tfidf.mm', self.corpus_tfidf)  

        # Write out the dictionary.
        self.dictionary.save(save_dir + 'documents.dict')  
        
    def load(self, save_dir='./'):
        """
        Load the corpus from a save directory.
        """
        tables = pickle.load(open(save_dir + 'tag-tables.pickle', 'rb'))
        self.tagsToEntries = tables[0]
        self.entriesToTags = tables[1]        
        self.titles = pickle.load(open(save_dir + 'titles.pickle', 'rb'))
        self.tfidf_model = TfidfModel.load(fname=save_dir + 'documents.tfidf_model')
        self.corpus_tfidf = corpora.MmCorpus(save_dir + 'documents_tfidf.mm')
        self.dictionary = corpora.Dictionary.load(fname=save_dir + 'documents.dict')
        
    def toSimSearch(self):
        """
        Initialize a SimSearch object using the data in this corpus.
        """        
        return SimSearch(corpus_tfidf=self.corpus_tfidf, titles=self.titles, 
                         dictionary=self.dictionary, 
                         tagsToEntries=self.tagsToEntries)
        
        

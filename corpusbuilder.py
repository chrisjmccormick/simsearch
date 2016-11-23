# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 21:19:15 2016

@author: Chris
"""

import nltk
import re
from gensim import corpora
from gensim.models import TfidfModel
from collections import defaultdict
from keysearch import KeySearch
from os import listdir, makedirs
from os.path import isfile, join, exists


# I lazily made this a global constant so that I wouldn't have to include
# it in the save and load features.
enc_format='utf-8'

class CorpusBuilder(object):
    """
    The CorpusBuilder object helps turn a collection of plain text documents 
    into a gensim corpus (that is, a collection of vectors representing the
    documents).  
    
    Document Format
    ===============
    
    To use the CorpusBuilder with your source documents, you will need to 
    convert your source documents to a plain text representation (copy and 
    paste into notepad is one simple approach, but tools also exist).

    The CorpusBuilder will tokenize the documents for you using NLTK, so you
    do not need to remove punctuation, whitespace, etc.

    It's possible to create multiple "documents" from a single text file.
    For example, you might choose to create a separate for every paragraph in
    the file. You can specify a regular expression to use for matching the
    beginnings of "documents" within the file. 
    
    Intended Usage
    ==============    
    The intended usage is as follows:
        1. Create a CorpusBuilder object, specifying a few parameters.
        2. Add all your text (must be *.txt) files to the CorpusBuilder using 
           either `addDirectory` or `addFile`.
        3. Call `buildCorpus` to build the corpus.
        5. Create a KeySearch object using `toKeySearch`. At this point, you
           are done with the CorpusBuilder--the KeySearch object contains
           the finished corpus.
    
    The next step is to create a SimSearch object and start performing 
    similarity searches!
    
    The CorpusBuilder does not have any 'save' and 'load' functionality 
    because you don't need it once the corpus is built. You can save and load
    the resulting KeySearch object instead.
    
    Parsing
    =======
    The CorpusBuilder will convert all characters to lowercase, tokenize your
    document with NLTK, filter stop words, and gather word frequency
    information.
    
    The `buildCorpus` step takes the final collection of documents (now 
    represented as filtered lists of tokens), removes words that only occur 
    once, builds the dictionary, then converts the documents into tf-idf 
    vectors. 
    
    Once the corpus has been built, you cannot add additional text to it.
    However, SimSearch and KeySearch do support providing new input text to use
    as the query for a search.
    
    Document Metadata
    =================    
    There are several pieces of metadata you can provide for each document,
    when calling `addDocument` (though they are all optional).
      * Document title: A string to represent the document, which can be
        useful when presenting search results.
      * Document file path & line numbers: If your source documents are 
        plain text, you can provide the path to the original source file and 
        the line numbers corresponding to the "document". This can be useful
        as another way to display a search result. The CorpusBuilder even 
        includes functions for reading the source file and retrieving the text 
        on the specified line numbers.
      * Document tags: You can supply a list of tags to associate with each
        document. In SimSearch, you can then search for more documents similar
        to those with a specified tag.      
            
    """
    def __init__(self, stop_words_file='', doc_start_pattern='', doc_start_is_separator=True, sub_patterns=[]):
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
        self.tagsToDocs = {}
        self.docsToTags = []

        self.files = []
        self.doc_line_nums = []

        # Count the occurrences of each word and store in 'frequency'.
        # This is a temporary data structure used for filtering out words
        # that only occur once in the corpus.
        # The final word counts can be found in self.dictionary        
        self.frequency = defaultdict(int)

        # Read in all of the stop words (one per line) and store them as a set.
        # The call to f.read().splitlines() reads the lines without the newline
        # char.           
        with open(stop_words_file) as f:
            lines = f.read().splitlines()
            
            stoplist = []            
            
            # Decode the stop words            
            for line in lines:
                stoplist.append(line.decode(enc_format))
                
            # Convert to a set representation.    
            self.stoplist = set(stoplist)               
        
        # Record the regex pattern for the start of a document.
        self.doc_start_pattern = doc_start_pattern

        # Record whether the doc start pattern is the first line of the doc
        # or just a separator (for example, an empty line)
        self.doc_start_is_separator = doc_start_is_separator        
        
        # Record any regex substitutions to make.
        self.sub_patterns = sub_patterns
    
    def applyRegExFilters(self, line):
        """
        Remove tokens matching some regex filters.
        """

        for substitution in self.sub_patterns:                    
            line = re.sub(substitution[0], substitution[1], line)
            
        return line    
    
    def addDocument(self, title, lines, tags=[], filename=None, doc_start=None, doc_end=None):
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

        # Store the list of tags for this doc.
        self.docsToTags.append(tags)                         
            
        # Add mappings from the tags to this journal entry.
        for tag in tags:
            # Convert tags to lower case.        
            tag = tag.lower()
                
            # Add the tag to the dictionary.
            if tag in self.tagsToDocs:
                self.tagsToDocs[tag].append(docID)
            else:
                self.tagsToDocs[tag] = [docID]   

        # Store the filenames only once.
        if filename in self.files:        
            fileID = self.files.index(filename)
        else:
            self.files.append(filename)            
            fileID = len(self.files) - 1

        # Store the file and line numbers.
        self.doc_line_nums.append((fileID, doc_start, doc_end))

        # Parse the document into a list of tokens.
        doc = []        
        lineNum = 0        
        
        # Parse each line in the document.
        for line in lines:
            
            lineNum += 1
            
            # Decode the string into Unicode so the NLTK can handle it.
            try:    
                line = line.decode(enc_format)        
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
    
    
    def addFile(self, filepath, filename):
        """
        
        """
        # Read in the text file
        with open(filepath) as f:
            content = f.readlines()

        # TODO! Add tag support.
        doc_tags = []    
        
        doc_title = ""       
        doc = []
        doc_start = -1
        
        # For each line in the file...
        for lineNum in range(0, len(content)):

            # Get the next line.
            line = content[lineNum]                        
                        
            # Check for the pattern matching the start of a document.
            matchStart = re.search(self.doc_start_pattern, line)                        

            # Whether to add this line to the current document.
            add_line = True            
            
            # If we've found the start of a new doc...            
            if matchStart and doc:                
                
                # Note the line number of the last line of the current document.
                doc_end = lineNum - 1 + 1                
                
                # Only add entries that are longer than one line.
                # This filters out section headings.                
                # TODO - Make this configurable.
                if len(doc) > 1:
                    self.addDocument(title=doc_title, lines=doc, tags=doc_tags, filename=filepath,doc_start=doc_start, doc_end=doc_end)
                
                doc = []
                doc_title = ""
                doc_start = -1

                # If the doc_start_pattern is a separator, don't add it to
                # the new doc (start the doc on the next line).            
                if self.doc_start_is_separator:
                    add_line = False
                
            # Add this line to the current doc.
            if add_line:
                # If this doc doesn't have a title yet, use this line.                
                if not doc_title:                
                    doc_title = filename + ' - ' + line       
                    doc_start = lineNum + 1 # Line numbers are numbered from 1.
                
                # Run reg ex filters to remove some tokens.
                line = self.applyRegExFilters(line)                
                
                # Add the line to the doc.
                doc.append(line)
        
        # End of file reached.
        doc_end = lineNum - 1 + 1;        
        self.addDocument(doc_title, doc, doc_tags, filename=filepath ,doc_start=doc_start, doc_end=doc_end)
   
    def addDirectory(self, dir_path):
        """
        Add all of the .txt files in the specified directory to the corpus.
        """
        # Get all the source text files.
        files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]

        # For each file in the directory:
        for f in files:

            # Only parse the .txt files.    
            if not (f[-4:] == '.txt'):
                continue   
    
            print '  Parsing file:', f
            
            # Pass down to addFile.
            self.addFile(filepath=(dir_path + f), filename=f[0:-4])

   
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

        # Delete the tokenized representation of the documents--no need to
        # carry this around!
        del self.documents[:]

        # Convert the simple bag-of-words vectors to a tf-idf representation.        
        self.tfidf_model = TfidfModel(corpus)
        self.corpus_tfidf = self.tfidf_model[corpus]
    
    def toKeySearch(self):
        ksearch = KeySearch(self.dictionary, self.tfidf_model, 
                            self.corpus_tfidf, self.titles, self.tagsToDocs,
                            self.docsToTags, self.files, self.doc_line_nums)        
        
        return ksearch

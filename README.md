simsearch
=========

Python tools for performing similarity searches on text documents.

SimSearch allows you to search a collection of documents by providing conceptually similar text as the search query, as opposed to the typical keyword-based approach. This technique is also referred to as semantic search or concept search.

I originally created these tools out of a desire to search my personal journals based on their topic rather than by keyword. Since I so often seem to forget the lessons life teaches me, I wanted to be able to find any past insights that might be relevant to the challenges I’m facing today.

## Framework
These tools are built around the powerful topic modeling framework in the [gensim](https://radimrehurek.com/gensim/) Python package by [Radim Rehurek](https://radimrehurek.com/).

SimSearch brings together the different features of `gensim`, and addresses many of the practical details needed to build a similarity search system. 

SimSearch consists of two classes:

* CorpusBuilder - This class helps you convert a collection of plain text documents into a gensim-style corpus. 
* SimSearch - This class helps you perform similarity searches on the corpus, where the search is performed by supplying a piece of example text. 

## Building a Corpus  - From Plain Text to Vectors.
The CorpusBuilder takes a collection of documents in the form of unprocessed plain text, and converts them into bag-of-words style vectors. 

A key step which the CorpusBuilder does not currently perform is the conversion of documents from, e.g., HTML or Word documents, into plain text.

The intended usage of the CorpusBuilder is as follows:

1. Create a CorpusBuilder object.
2. Call `setStopWordList` to provide the list of stop words.
3. Call `addDocument` for each doc or piece of text in your corpus.
4. Call `buildCorpus` to build the corpus.
5. Create a SimSearch object (providing the built corpus to the 
   SimSearch constructor) and start performing similarity searches!
6. Save and load state by calling the save and load functions of the
   SimSearch object--this will save the built corpus as well.

## Performing Searches

simsearch
=========

Python tools for performing similarity searches on text documents.

SimSearch allows you to search a collection of documents by providing conceptually similar text as the search query, as opposed to the typical keyword-based approach. This technique is also referred to as "semantic search" or "concept search". The input text for the query can come from the corpus itself, or from new input text. You can even use multiple input documents for a single query to improve the quality of the results.

The other key piece of functionality in this project is the automation of parsing your plain text documents into a corpus. You can give it a folder of .txt files and it will automatically learn the necessary models and representations.

I originally created these tools out of a desire to search my personal journals based on their topic rather than by keyword. Since I so often seem to forget the lessons life teaches me, I wanted to be able to find any past insights that might be relevant to the challenges I’m facing today. The idea is that I would take a journal entry which I wrote today, and search for past similar entries. In this case, each journal entry is a separate "document" in the corpus.

Contents
========

* [Framework](#framework)
  * [CorpusBuilder](#corpusbuilder)
  * [KeySearch](#keysearch)
  * [SimSearch](#simsearch)
* [Example Code](#exmaple-code)
  * [Preparing Corpus](#preparing-corpus)
  * [Description of Examples](#description-of-examples)
* [Installation and Dependencies](#installation-and-dependencies)
  * [Punkt Installation](#punkt-installation)
* [Converting Documents to Plain Text](#converting-documents-to-plain-text) 

# Framework
These tools are built around the powerful topic modeling framework in the [gensim](https://radimrehurek.com/gensim/) Python package by [Radim Rehurek](https://radimrehurek.com/).

SimSearch brings together the different features of `gensim`, and fills in many of the practical details needed to build a similarity search system. 

SimSearch consists of three classes:

* CorpusBuilder - This class helps you convert a collection of plain text documents into a gensim-style corpus. 
* KeySearch - This class stores and provides access tf-idf corpus
* SimSearch - This class helps you perform similarity searches on the corpus using LSI, where the search is performed by supplying a piece of example text. 

## CorpusBuilder
This class is all about text parsing. It helps you get from a collection of plain text documents to a gensim-style corpus.

It handles things like tokenizing the text and filtering stop words and infrequent words.

Once all of the documents have been parsed and filtered and represented as lists of tokens, it builds a gensim dictionary and learns a tf-idf model from the corpus. Finally, it converts all of the documents to their tf-idf representation.

At this point, the corpus is complete, and is passed on to the KeySearch and SimSearch classes. Once the corpus is built, all of your interactions will be with the KeySearch and SimSearch objects--the CorpusBuilder is no longer needed.

## KeySearch
KeySearch is short for "keyword search". 

The KeySearch object stores:

* The completed dictionary
* The learned tf-idf model
* The tf-idf corpus
* Additional document metadata

Because the KeySearch object has a tf-idf representation of the documents, it knows what words appear in each document. This allows it to support boolean keyword searches, as well as provide insight into search results (What words contribute most to the conceptual similarity between these two documents? What are the key terms in this cluster of documents?)

The KeySearch class is also what you will use to create vector representations of new text that is not in the corpus. You'll notice that the CorpusBuilder does a lot more filtering of the input text than KeySearch does--KeySearch does not need to do this for new text because the dictionary has already been constructed, and any token that's not in the dictionary will simply be ignored.

## SimSearch
SimSearch is short for "similarity search".

This class uses LSI to create a higher quality conceptual representation of the documents. It is built on top of a KeySearch object, and learns an LSI model from the tf-idf corpus. It supports conceptual search operations like "find documents in the corpus similar to this input text". It also has a method for helping you interpret the results of a similarity search.

# Example Code

I've provided an example corpus to demonstrate the use of SimSearch. 

The sample document collection is an exhaustive Biblical commentary written around 1710 by Matthew Henry. His entire commentary is [available online](https://www.ccel.org/ccel/henry/mhc) in plain text, and is public domain. His thoughts are divided into subsections which are neatly separated by blank lines. This allowed me to parse them easily, and I treat each subsection as a separate "document". This creates a total of 30,707 documents! That man spent a lot of time writing...

## Preparing Corpus
The first step to running the example is preparing the corpus.

Step 1: Unzip mhc.zip (under `/mhc/`, MHC = "Matthew Henry's Commentary") so that you have `/mhc/mhc1.txt`, `/mhc/mhc2.txt`, ... `/mhc/mhc6.txt`.

Step 2: Run the script `parseMHC.py`. This will use CorpusBuilder to create a gensim corpus from the commentary. This will take a couple minutes. Finally, the resulting CorpusBuilder object is used to create a SimSearch object, and this is saved to the subdirectory `/mhc_corpus/`.

## Description of Examples
I've included several example scripts, all named with the format run*.py

`runSearchByDoc.py`: This example takes a particular section of the MHC text, and searches for closest matches to it in the corpus. It also interprets the top match, displaying which words contributed most to the similarity.

`runKMeansClustering.py`: This example clusters the entire MHC corpus with k-means and displays the top words for each resulting cluster.

`runSearchByText.py`: This example demonstrates my most typical useage of SimSearch--searching the corpus given some new input text. Paste your query text into `input.txt` and then run this script.

`runSearchByKeyword.py`: This example shows how to search the corpus by keywords--note that it's not indexed, so it's slow.

# Installation and Dependencies

You'll need to install:

1. [gensim](https://radimrehurek.com/gensim/install.html) (for topic modeling)
2. [sklearn](http://scikit-learn.org/stable/install.html) (for clustering)
3. [NLTK](http://www.nltk.org/install.html) and the Punkt Tokenizer Models (for tokenizing).

## Punkt Installation
Once you've installed NLTK, type the following in Python to launch the NLTK downloader utility.

```python
>> import nltk

>> nltk.download()
```

Then download `punkt` from the Models tab.
![NLTK Downloader](http://www.mccormickml.com/assets/nltk/nltk_downloader_punkt.png)

# Converting Documents to Plain Text
A key step which the CorpusBuilder does not currently perform is the conversion of documents from, e.g., HTML or Word documents, into plain text.

For small document collections, I've found it sufficient to open the document, select-all, then simply copy and paste the text into a plain text editor such as notepad.

For larger document collections, there are utilities out there which can help automate the process.

For Word documents, the [DocX for Python](https://python-docx.readthedocs.io/en/latest/) package looks solid. See an example of using it to extract the plain text [here](http://stackoverflow.com/questions/42482/best-way-to-extract-text-from-a-word-doc-without-using-com-automation).
   
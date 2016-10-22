simsearch
=========

Python tools for performing similarity searches on text documents.

SimSearch allows you to search a collection of documents by providing conceptually similar text as the search query, as opposed to the typical keyword-based approach. This technique is also referred to as "semantic search" or "concept search".

I originally created these tools out of a desire to search my personal journals based on their topic rather than by keyword. Since I so often seem to forget the lessons life teaches me, I wanted to be able to find any past insights that might be relevant to the challenges I’m facing today. The idea is that I would take a journal entry which I wrote today, and search for past similar entries. In this case, each journal entry is a separate "document" in the corpus.

## Framework
These tools are built around the powerful topic modeling framework in the [gensim](https://radimrehurek.com/gensim/) Python package by [Radim Rehurek](https://radimrehurek.com/).

SimSearch brings together the different features of `gensim`, and fills in many of the practical details needed to build a similarity search system. 

SimSearch consists of two classes:

* CorpusBuilder - This class helps you convert a collection of plain text documents into a gensim-style corpus. 
* SimSearch - This class helps you perform similarity searches on the corpus, where the search is performed by supplying a piece of example text. 

## Installation & Dependencies

You'll need to install:

1. [gensim](https://radimrehurek.com/gensim/install.html) (for topic modeling)
2. [NLTK](http://www.nltk.org/install.html)
3. The Punkt Tokenizer Models (for tokenizing) using the NLTK downloader (Run nltk.download() in Python to launch the downloader).

## Example Code

I've provided an example corpus to demonstrate the use of SimSearch. 

The sample document collection is an exhaustive Biblical commentary written around 1710 by Matthew Henry. His entire commentary is available online in plain text, and is public domain. His thoughts are divided into subsections which are neatly separated by blank lines. This allowed me to parse them easily, and I treat each subsection as a separate "document". This creates a total of 39,172 documents! That's a whole lot of Biblical exposition!

Here are the steps to run the example.

Step 1: Unzip mhc.zip (under `/mhc/`, MHC = "Matthew Henry's Commentary") so that you have `/mhc/mhc1.txt`, `/mhc/mhc2.txt`, ... `/mhc/mhc6.txt`.

Step 2: Run the script `parseMHC.py`. This will use CorpusBuilder to create a gensim corpus from the commentary. This will take a couple minutes. Finally, the resulting CorpusBuilder object is used to create a SimSearch object, and this is saved to the subdirectory `/mhc_corpus/`.

Step 3: Run the script `playWithSimSearch.py`. This will load the SimSearch object from disk, then perform some example searches.

## Building a Corpus - From Plain Text to Vectors.
The CorpusBuilder takes a collection of documents in the form of unprocessed plain text, and converts them into bag-of-words style vectors. 

The intended usage of the CorpusBuilder is as follows:

1. Create a CorpusBuilder object.
2. Call `setStopWordList` to provide the list of stop words.
3. Call `addDocument` for each doc or piece of text in your corpus.
4. Call `buildCorpus` to build the corpus.
5. Create a SimSearch object (providing the built corpus to the 
   SimSearch constructor) and start performing similarity searches!
6. Save and load state by calling the save and load functions of the
   SimSearch object--this will save the built corpus as well.

## Converting Documents to Plain Text
A key step which the CorpusBuilder does not currently perform is the conversion of documents from, e.g., HTML or Word documents, into plain text.

For small document collections, I've found it sufficient to open the document, select-all, then simply copy and paste the text into a plain text editor such as notepad.

For larger document collections, there are utilities out there which can help automate the process.

For Word documents, the [DocX for Python](https://python-docx.readthedocs.io/en/latest/) package looks solid. See an example of using it to extract the plain text [here](http://stackoverflow.com/questions/42482/best-way-to-extract-text-from-a-word-doc-without-using-com-automation).
   
## Performing Searches
Currently there are two ways to search the corpus:

Option 1: `findSimilarToDoc` allows you to take one of the documents _from the corpus_ to use as a query, and searches the rest of the corpus for similar documents.

Option 2: `findSimilarToText` allows you to provide new input text (that does not have to be from the corpus), and finds conceptually similar documents in the corpus. 

## Punkt Installation
Once you've installed NLTK, type the following in Python to launch the NLTK downloader utility.

>> import nltk

>> nltk.download()

Then download `punkt` from the Models tab.
![NLTK Downloader](http://www.mccormickml.com/assets/nltk/nltk_downloader_punkt.png)



from corpusbuilder import CorpusBuilder
from simsearch import SimSearch
from os import listdir, makedirs
from os.path import isfile, join, exists


# Get all the source text files.
mhcFiles = [f for f in listdir('./mhc/') if isfile(join('./mhc/', f))]

cb = CorpusBuilder()

cb.setStopWordList('stop_words.txt')

print 'Parsing Matthew Henry\'s Commentary...'    

# For each text file:
for mhcFile in mhcFiles:

    # Only parse the .txt files.    
    if not (mhcFile[-4:] == '.txt'):
        continue   
    
    print '  Parsing file:', mhcFile        
    
    # No tags in this example.
    entry_tags = []    
    
    # Read in the text file
    with open('./mhc/' + mhcFile) as f:
        content = f.readlines()

        # Get the name of the file without the ".txt"
        filename = mhcFile[0:-4]
        
        entry_title = ""       
        entry = ""
        
        # For each line in the file...
        for line in content:
                        
            # If we find a blank line, treat that as a break in sections.
            if line == "\n" and entry:                
                cb.addDocument(entry_title, [entry], entry_tags)
                entry = ""
                entry_title = ""
                            
            # Otherwise, it's an entry.
            else:
                # If this entry doesn't have a title yet, use this line.                
                if not entry_title:                
                    entry_title = filename + ' - ' + line       
                
                # Append the words to the entry, separated by spaces.
                entry += " " + line
        
        # End of file reached.        
        cb.addDocument(entry_title, [entry], entry_tags)

print 'Done.'

print 'Building corpus...'

cb.buildCorpus()

# Print the top 30 most common words.
cb.printTopNWords(topn=30)

print '\nVocabulary contains', cb.getVocabSize(), 'unique words.'

print 'Corpus contains', len(cb.corpus_tfidf), '"documents" represented by tf-idf vectors.'

# Initialize a SimSearch object from the corpus.
ssearch = SimSearch(cb)

# Train LSI with 100 topics.
print '\nTraining LSI...'
ssearch.trainLSI(num_topics=100)

print '\nSaving to disk...'
if not exists('./mhc_corpus/'):
    makedirs('./mhc_corpus/')

ssearch.save(save_dir='./mhc_corpus/')

print 'Done!'
    
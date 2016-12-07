from corpusbuilder import CorpusBuilder
from simsearch import SimSearch
from os import makedirs
from os.path import exists


# Create the CorpusBuilder.
cb = CorpusBuilder()

# Set the list of stop words.
cb.setStopList('stop_words_mhc.txt')

# Match blank lines as the separator between "documents".    
cb.setDocStartPattern(r'^\s*$')

#The MHC text includes a number of patterns which need to be filtered out...    
sub_patterns = []

# Remove line breaks of the form "___________"
sub_patterns.append((r'_+', ' '))

# Remove verse references like "ver. 5-7" or "ver. 25, 26"
sub_patterns.append((r'ver\. \d+(-|, )*\d*', ' '))

# Remove verse numbers of the form "18.", "2.", etc., as well as any other
# remaining numbers.
sub_patterns.append((r'\d+\.?', ' '))
    
cb.setSubstitutions(sub_patterns)
                   

print 'Parsing Matthew Henry\'s Commentary...'    
# Parse all of the text files in the directory.
cb.addDirectory('./mhc/')

print 'Done.'

print 'Building corpus...'

cb.buildCorpus()

# Initialize a KeySearch object from the corpus.
ksearch = cb.toKeySearch()

# Print the top 30 most common words.
ksearch.printTopNWords(topn=30)

print '\nVocabulary contains', ksearch.getVocabSize(), 'unique words.'

print 'Corpus contains', len(ksearch.corpus_tfidf), '"documents" represented by tf-idf vectors.'

# Initialize a SimSearch object from the KeySearch.
ssearch = SimSearch(ksearch)

# Train LSI with 100 topics.
print '\nTraining LSI...'
ssearch.trainLSI(num_topics=300)

print '\nSaving to disk...'
if not exists('./mhc_corpus/'):
    makedirs('./mhc_corpus/')

ssearch.save(save_dir='./mhc_corpus/')

print 'Done!'

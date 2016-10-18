from corpusbuilder import CorpusBuilder
from os import listdir, makedirs
from os.path import isfile, join, exists

   

# Get all the journal files.
mhcFiles = [f for f in listdir('./mhc/') if isfile(join('./mhc/', f))]

cb = CorpusBuilder()

print 'Parsing Matthew Henry\'s Commentary...'    

# For each journal file:
for mhcFile in mhcFiles:
    
    #name = journalFile[0:-4]    
    print '  Parsing file:', mhcFile        
    
    entry_tags = []    
    
    # Read in the journal file
    with open('./mhc/' + mhcFile) as f:
        content = f.readlines()

        volume_name = mhcFile[0:-4]

        # Create a list to hold all of the parsed journal entries.
        entry_title = ""       
           
        entry = ""
        
        # For each line in the file...
        for line in content:
                        
            # If we find a blank line
            if line == "\n" and entry:
                                                
                if entry:
                    cb.addDocument(entry_title, [entry], entry_tags)
                    entry = ""
                    entry_title = ""
                            
            # Otherwise, it's an entry.
            else:
                # If this entry doesn't have a title yet, use this line.                
                if not entry_title:                
                    entry_title = volume_name + ' - ' + line       
                
                # Append the words to the entry, separated by spaces.
                entry += " " + line
        
        # End of file reached.        
        cb.addDocument(entry_title, [entry], entry_tags)

print 'Done.'

print 'Building corpus!'

cb.buildCorpus()

print 'CB contains', len(cb.documents), 'documents.'

if not exists('./mhc_corpus/'):
    makedirs('./mhc_corpus/')

cb.save(save_dir='./mhc_corpus/')


    
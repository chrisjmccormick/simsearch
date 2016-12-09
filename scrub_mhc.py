# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 11:04:05 2016

@author: Chris
"""

from os import listdir
from os.path import isfile, join


# Get all the source text files.
mhcFiles = [f for f in listdir('./mhc/') if isfile(join('./mhc/', f))]


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

    last_line = -1
    # Search for the indexes section.
    for i in range(0, len(content)):
        if 'Indexes' in content[i]:
            print 'Found it at line', i, '!'
            last_line = i - 2
            break
    
    # Write the contents back minus the indexes.
    if not last_line == -1:
        print 'Writing back without indexes'
        with open('./mhc/' + mhcFile, 'wb') as f:
            f.writelines(content[0:last_line])
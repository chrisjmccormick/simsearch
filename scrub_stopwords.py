# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 12:14:19 2016

@author: Chris
"""

# Open the stop words list.
with open('./stop_words.txt', 'rb') as f:
    content = f.readlines()
    
    
# Convert to a set to remove duplicates, then back to a list.
content = list(set(content))
 
# Sort the words.
content.sort()

with open('stop_words.txt', 'wb') as f:
    content = f.writelines(content)


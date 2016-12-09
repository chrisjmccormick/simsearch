# -*- coding: utf-8 -*-
"""
This script demonstrates how to search the corpus by some new input text,
specified in a text file 'input.txt'.

It also provides a simple user interface for reviewing the results.

@author: Chris
"""

from simsearch import SimSearch

# Load the pre-built corpus.
print('Loading the saved SimSearch and corpus...')
(ksearch, ssearch) = SimSearch.load(save_dir='./mhc_corpus/')

# Load 'input.txt' as the input to the search.
input_vec = ksearch.getTfidfForFile('input.txt')

# Number of results to go through.
topn = 10

print 'Searching by contents of input.txt...'

# Perform the search.
results = ssearch.findSimilarToVector(input_vec, topn=topn)

for i in range(0, topn):

    # Show the text for the result.
    ssearch.printResultsBySourceText([results[i]], max_lines=8)

    # Get the tf-idf vector for the result.
    result_vec = ksearch.getTfidfForDoc(results[i][0])

    print('')

    # Interpret the match.
    ssearch.interpretMatch(input_vec, result_vec, min_pos=0)

    # Wait for user input.
    command = raw_input("[N]ext result  [F]ull text  [Q]uit\n: ").lower()
        
    # q -> Quit.
    if (command == 'q'):
        break
    # f -> Display full doc source.
    elif (command == 'f'):
        ksearch.printDocSourcePretty(results[i][0], max_lines=100)
        raw_input('Press enter to continue...')
        


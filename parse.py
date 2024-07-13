"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
CS131 - Artificial Intelligence
by Madelyn Silveira
04/12/2023

Implementation of a Bayesian Network that classifies objects as either a bird
or an airplane.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import linecache
from bayesian import calculate

# parses one line of input into a python list
def parse_line(file, line_num):

    # Read the line from the file
    line = linecache.getline(file, line_num)

    # Split the line into individual entries
    entries = line.split()

    # Get the data
    data = []
    for entry in entries:
        val = float(entry)
        data.append(val)

    return data

# parses a whole line of input and calculates probabilities
def parse_tracks(filename, num_tracks, sampling_length, L_bird, L_plane, prior):
    print("Parsing " + filename + "...")
    line = 1
    while line < num_tracks + 1:
        type = calculate(parse_line(filename, line), 0, sampling_length, 
                        L_bird, L_plane, prior, prior)
        
        print("Track #" + str(line) + " classification: " + type)
        line += 1

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
CS131 - Artificial Intelligence
by Madelyn Silveira
04/12/2023

Implementation of a Bayesian Network that classifies objects as either a bird
or an airplane.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

from parse import parse_line, parse_tracks
from bayesian import calculate

# initial probabilities and variables: 
sampling_length = 600
prior_probability = 0.5 # for both b and p

# calculate from the likelihood: L(C|V)
L_bird = parse_line('likelihood.txt', 1)  # 400 entries
L_plane = parse_line('likelihood.txt', 2) # 400 entries

# modify based on testing
parse_tracks('training.txt', 20, sampling_length, L_bird, L_plane, prior_probability)

# parse the test tracks
parse_tracks('testing.txt', 10, sampling_length, L_bird, L_plane, prior_probability)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
CS131 - Artificial Intelligence
by Madelyn Silveira
04/12/2023

Implementation of a Bayesian Network that classifies objects as either a bird
or an airplane.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import math

# finds the probability of either bird or plane for a given v on interval 0.5
def P_dist(L_b_v, L_p_v, velocity):
    # 400 samples per line, 0-200 x range
    sample = int(velocity * 2)
    P_v_b = L_b_v[sample - 1] * 0.5
    P_v_p = L_p_v[sample - 1] * 0.5
    return (P_v_b, P_v_p)

# calculates posterior probabilities from probability distributions
def P_post(P_v_b, P_v_p):
    # prior probabilities of both are 0.5
    P_v = (P_v_b + P_v_p) * 0.5

    # posterior probabilities
    P_b_v = P_v_b * 0.5 / P_v
    P_p_v = P_v_p * 0.5 / P_v
    
    return (P_b_v, P_p_v)

# returns proper letter given the max of two probibilities
def classify(P_bird, P_plane):
    # print("P_bird: " + str(P_bird))
    # print("P_plane: " + str(P_plane))
    # print()
    if (max(P_bird, P_plane) == P_bird):
        return 'b'
    return 'a'

# calculates the probility of distribution and posterior probability to 
# determine the current classification given the previous assignment
def calculate(data, i, end, L_bird, L_plane, P_b_prev, P_p_prev):
    # base case: end of track
    if i == end:
        result = classify(P_b_prev, P_p_prev)
        return result
    else:
        # move to  next velocity
        v = data[i]
        i += 1

        # skip NaN values for now
        if math.isnan(v):
            return calculate(data, i, end, L_bird, L_plane, P_b_prev, P_p_prev)

        # get the probabilities
        P_v_b, P_v_p = P_dist(L_bird, L_plane, v)
        P_b_v, P_p_v = P_post(P_v_b, P_v_p)

        # ideally first run through only
        if (P_b_prev == P_p_prev):
            P_b = P_b_prev * P_b_v 
            P_p = P_p_prev * P_p_v 
            # print("P_bird: " + str(P_b))
            # print("P_plane: " + str(P_p))
            # print(classify(P_b, P_p), end=" ")
            return calculate(data, i, end, L_bird, L_plane, P_b, P_p)
            
        # recursive call based on probability of transition, .9 or .1
        elif (max(P_b_prev, P_p_prev) == P_b_prev):
            P_b_b = P_b_prev * P_b_v * .9
            P_b_p = P_p_prev * P_b_v * .1

            # print(classify(P_b_b, P_b_p), end=" ")
            return calculate(data, i, end, L_bird, L_plane, P_b_b, P_b_p)
        else:
            P_p_b = P_b_prev * P_p_v * .1
            P_p_p = P_p_prev * P_p_v * .9

            # print(classify(P_p_b, P_p_p), end = " ")
            return calculate(data, i, end, L_bird, L_plane, P_p_b, P_p_p)
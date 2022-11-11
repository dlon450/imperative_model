import pandas as pd 
import numpy as np

def need_weights():
    '''
    Return weights for respective needs
    '''
    return np.array([0.5, 0.25, 0.125, 0.0625, 0.03125])

def get_default_rates():
    '''
    Return default rates for each community j
    '''
    return 0.1

def get_social_utilities(df, proportions, households):
    '''
    Return social utilities for each community j
        S_j = (1 - \pi_j) * H_j * \sum_i {w_i * p_{ij}}
    '''
    return (1 - get_default_rates()) * households * proportions.dot(need_weights())

if __name__ == "__main__":
    pass
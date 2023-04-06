import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from random import choice
from os.path import exists

import sys
"""
sys.path.append("../Configs/")
import constants
"""
def generate_unused_prefix(constants):
    """
    For use in naming different deployment runs under the same prefix. 
    """
    adjectives = [
        "tired",
        "happy",
        "fiery",
        "confused",
        "slender",
        "calm",
        "drab",
        "gaudy",
        "worried",
        "cheeky",
    ]
    
    animals = [
        "kitten",
        "cow",
        "pig",
        "snake",
        "otter",
        "panda",
        "mouse",
        "gerbil",
        "puppy",
    ]
    
    prefix = None
    # generate a prefix combination that has not already been used
    while prefix is None or exists(constants.PREDICTION_ROOT+prefix+'_results.csv'):
        adj = choice(adjectives)
        animal = choice(animals)
        
        prefix = f"{adj}_{animal}"
        
    return prefix
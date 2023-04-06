from descarteslabs import Storage, Auth
import numpy as np
from random import choice
import sys
sys.path.append("../UI_Labeling")
import config

def generate_unused_prefix():
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
    storage = Storage(auth=Auth(client_id=config.ID,client_secret=config.SECRET))
    
    # generate a prefix combination that has not already been used
    while prefix is None or len(storage.list(prefix=prefix)) > 1:
        adj = choice(adjectives)
        animal = choice(animals)
        
        prefix = f"{adj}_{animal}"
        
    return prefix
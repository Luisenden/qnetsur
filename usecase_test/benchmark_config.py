import pickle, time
import pandas as pd
from functools import partial, wraps
import numpy as np
from datetime import datetime
import argparse
import sys
sys.path.append('../')

def test_simwrapper(simulation, kwargs: dict):
    return -simulation(**kwargs), 

# specify fixed parameters of quantum network simulation
vals = {}

# specify variables and bounds of quantum network simulation
vars = {
        'range':{
            'x': ([-5,5], 'float'),
            'y': ([-5,5], 'float')
        },
        'choice':{},
        'ordinal':{}
        } 

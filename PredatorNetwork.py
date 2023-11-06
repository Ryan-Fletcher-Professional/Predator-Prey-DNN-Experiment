"""
THIS FILE WRITTEN BY RYAN FLETCHER AND
"""

import numpy as np
import Networks


class PredatorNetwork(Networks.CreatureNetwork):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)

"""
THIS FILE WRITTEN BY RYAN FLETCHER AND
"""

import numpy as np
import math
import Networks
import main


def filter_out_predators(creature):
    return creature["type"] == main.PREY


class PredatorNetwork(Networks.CreatureNetwork):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
    
    def loss(self, state_info):
        creature_states = filter(filter_out_predators, state_info["creature_states"])
        closest = {"distance" : math.inf}
        for creature in creature_states:
            if creature["distance"] < closest["distance"]:
                closest = creature
        return closest["distance"]

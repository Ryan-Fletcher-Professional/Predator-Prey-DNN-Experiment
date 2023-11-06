"""
THIS FILE WRITTEN BY RYAN FLETCHER AND
"""

import math
import Globals
import Networks


def filter_out_predators(creature):
    return creature["type"] == Globals.PREY


class PredatorNetwork(Networks.CreatureNetwork):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
    
    def transform(self, state_info):
        """
        TODO
        """
        return [1.0]*(4*len(filter(filter_out_predators, state_info["creature_states"])))  # Placeholder
    
    def loss(self, state_info):
        creature_states = filter(filter_out_predators, state_info["creature_states"])
        closest = {"distance" : math.inf}
        for creature in creature_states:
            if creature["distance"] < closest["distance"]:
                closest = creature
        return closest["distance"]

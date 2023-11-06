"""
THIS FILE WRITTEN BY RYAN FLETCHER AND
"""

import math
import Globals
import Networks


def filter_out_preys(creature):
    return creature["type"] == Globals.PREDATOR


class PreyNetwork(Networks.CreatureNetwork):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        self.loss_mode = hyperparameters.get("loss_mode", Globals.SUBTRACT_MODE)
        
    def transform(self, state_info):
        """
        TODO
        """
        return [1.0]*(4*len(filter(filter_out_preys, state_info["creature_states"])))  # Placeholder
    
    def loss(self, state_info):
        creature_states = filter(filter_out_preys, state_info["creature_states"])
        closest = {"distance" : math.inf}
        for creature in creature_states:
            if creature["distance"] < closest["distance"]:
                closest = creature
        if self.loss_mode == Globals.RECIPROCAL_MODE:
            return 1 / closest["distance"]
        elif self.loss_mode == Globals.SUBTRACT_MODE:
            return -closest["distance"]

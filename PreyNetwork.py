"""
THIS FILE WRITTEN BY RYAN FLETCHER AND
"""

import numpy as np
import math
import Networks
import main


RECIPROCAL_MODE = "reciprocal"
SUBTRACT_MODE = "subtract"


def filter_out_preys(creature):
    return creature["type"] == main.PREDATOR


class PreyNetwork(Networks.CreatureNetwork):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        self.loss_mode = hyperparameters.get("loss_mode", SUBTRACT_MODE)
    
    def loss(self, state_info):
        creature_states = filter(filter_out_preys, state_info["creature_states"])
        farthest = {}
        closest = {"distance" : math.inf}
        for creature in creature_states:
            if creature["distance"] < closest["distance"]:
                closest = creature
        if self.loss_mode == RECIPROCAL_MODE:
            return 1 / closest["distance"]
        elif self.loss_mode == SUBTRACT_MODE:
            return -closest["distance"]

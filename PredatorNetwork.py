"""
THIS FILE WRITTEN BY RYAN FLETCHER AND SANATH UPADHYA
"""

import torch
import math
from Globals import *
import Networks


class PredatorNetwork(Networks.CreatureFullyConnected):
    def __init__(self, hyperparameters, self_id):
        super().__init__(hyperparameters)
        self.id = self_id
    
    def transform(self, state_info):
        # Own energy + own other characteristics + other creatures' other characteristics IN THAT ORDER
        this = None
        for creature_state in state_info["creature_states"]:
            if creature_state["id"] == self.id:
                this = creature_state
                break
        flattened = [this["energy"], this["type"], this["distance"], this["speed"]]
        for prey_state in filter(FILTER_OUT_PREDATOR_DICTS, state_info["creature_states"]):
            flattened.append(prey_state["type"])
            flattened.append(prey_state["distance"])
            flattened.append(prey_state["speed"])
        return torch.FloatTensor(flattened)
    
    def loss(self, state_info):
        creature_states = filter(FILTER_OUT_PREDATOR_DICTS, state_info["creature_states"])
        closest = {"distance" : math.inf}
        for creature in creature_states:
            if creature["distance"] < closest["distance"]:
                closest = creature
        return torch.tensor(closest["distance"], requires_grad=True)

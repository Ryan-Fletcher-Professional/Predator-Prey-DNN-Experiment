"""
THIS FILE WRITTEN BY RYAN FLETCHER AND SANATH UPADHYA
"""

import torch
import math
from Globals import *
import Networks


class PreyNetwork(Networks.CreatureFullyConnected):
    def __init__(self, hyperparameters, self_id):
        super().__init__(hyperparameters)
        self.loss_mode = hyperparameters.get("loss_mode", SUBTRACT_MODE)
        self.id = self_id
        
    def transform(self, state_info):
        # Own energy + own other characteristics + other creatures' other characteristics IN THAT ORDER
        this = None
        for creature_state in state_info["creature_states"]:
            if creature_state["id"] == self.id:
                this = creature_state
                break
        flattened = [this["energy"], this["relative_speed"]]
        for predator_state in filter(FILTER_OUT_PREY_DICTS, state_info["creature_states"]):
            flattened += [predator_state["relative_speed"], predator_state["perceived_type"], predator_state["distance"]]
        return torch.FloatTensor(flattened)
    
    def loss(self, state_info):
        creature_states = filter(FILTER_OUT_PREY_DICTS, state_info["creature_states"])
        closest = {"distance" : math.inf}
        for creature in creature_states:
            if creature["distance"] < closest["distance"]:
                closest = creature
        if self.loss_mode == RECIPROCAL_MODE:
            return torch.tensor(1 / closest["distance"], requires_grad=True)
        elif self.loss_mode == SUBTRACT_MODE:
            return torch.tensor(-closest["distance"], requires_grad=True)

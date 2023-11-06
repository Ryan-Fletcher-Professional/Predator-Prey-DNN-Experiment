"""
THIS FILE WRITTEN BY RYAN FLETCHER AND
"""

import torch
import math
import Globals
import Networks


def filter_out_predators(creature):
    return creature["type"] == Globals.PREY


class PredatorNetwork(Networks.CreatureFullyConnected):
    def __init__(self, hyperparameters, self_id):
        super().__init__(hyperparameters)
        self.id = self_id
    
    def transform(self, state_info):
        # Own energy + own other characteristics + other creatures' other characteristics IN THAT ORDER
        # return torch.FloatTensor([1.0] * ((3 * len([predator for predator in filter(filter_out_predators, state_info["creature_states"])])) + 4))  # Placeholder
        
        this = None
        for creature_state in state_info["creature_states"]:
            if creature_state["id"] == self.id:
                this = creature_state
                break
        flattened = [this["energy"], this["type"], this["distance"], this["speed"]]
        for prey_state in filter(filter_out_predators, state_info["creature_states"]):
            flattened.append(prey_state["type"])
            flattened.append(prey_state["distance"])
            flattened.append(prey_state["speed"])
        return torch.FloatTensor(flattened)
    
    def loss(self, state_info):
        creature_states = filter(filter_out_predators, state_info["creature_states"])
        closest = {"distance" : math.inf}
        for creature in creature_states:
            if creature["distance"] < closest["distance"]:
                closest = creature
        return torch.tensor(closest["distance"], requires_grad=True)

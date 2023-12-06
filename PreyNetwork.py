"""
THIS FILE WRITTEN BY RYAN FLETCHER, SANATH UPADHYA, AND ADVAIT GOSAI
"""

import torch
import math
from Globals import *
import Networks


class PreyNetwork(Networks.CreatureFullyConnectedShallow):
    def __init__(self, hyperparameters, self_id):
        super().__init__(hyperparameters)
        self.loss_mode = hyperparameters.get("loss_mode", DEFAULT_PREY_LOSS_MODE)
        self.id = self_id
        
    def transform(self, state_info):
        # Own energy + own other characteristics + other creatures' other characteristics IN THAT ORDER
        this = None
        for creature_state in state_info["creature_states"]:
            if creature_state["id"] == self.id:
                this = creature_state
                break
        flattened = [this[key] for key in self.self_inputs]  #[this["stun"], this["energy"], this["relative_speed"]]
        for predator_state in filter(FILTER_IN_PREDATOR_DICTS, state_info["creature_states"]):
            flattened += [predator_state[key] for key in self.other_inputs]  #[predator_state["relative_speed"], predator_state["perceived_type"], predator_state["distance"]]
        return torch.FloatTensor(flattened)
    
    def loss(self, state_info):
        creature_states = filter(FILTER_IN_PERCEIVED_PREDATOR_DICTS, state_info["creature_states"])
        if self.print_state:
            print(f"\nState info for {self.id}:\n\t{state_info['creature_states']}")
        closest = { "distance" : self.max_distance }  # Instantiated dynamically according to creature's sight range
        for creature in creature_states:
            if creature["distance"] < closest["distance"]:
                closest = creature
        if self.loss_mode == RECIPROCAL_MODE:
            r = torch.tensor(1.0 / closest["distance"], requires_grad=True)
            if self.print_loss:
                print(f"\nLoss for {self.id}:\n\t{r}")
            return r
        elif self.loss_mode == SUBTRACT_MODE:
            r = torch.tensor(-closest["distance"], requires_grad=True)
            if self.print_loss:
                print(f"\nLoss for {self.id}:\n\t{r}")
            return r

"""
THIS FILE WRITTEN BY RYAN FLETCHER AND
"""

import numpy as np


class CreatureNetwork:
    def __init__(self, hyperparameters):
        pass

    def get_inputs(self, state_info):
        """
        :param state_info: See Environment.get_state_info() return specification.
        :return: [ 2d-array : [float forward force, float rightwards force], float clockwise rotation in [0,2pi) ]
        """
        # TODO : NN STUFF HERE
        return [[0.001, 0.000], np.pi / 2]

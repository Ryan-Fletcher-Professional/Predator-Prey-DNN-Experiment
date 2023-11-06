"""
THIS FILE WRITTEN BY RYAN FLETCHER AND
"""

import pygame
import numpy as np
import math
# from multiprocessing import Process  # Interferes with PyGame. Use when not drawing?
# import time
DTYPE = np.float64
import Environment
import PreyNetwork
import PredatorNetwork

DRAW = True
ALWAYS_OVERRIDE_PREY_MOVEMENT = True

# Constants and globals
# <>_ATTRS = {
#     "fov"                       : float in (0,1)
#     "num_rays"                  : positive int
#     "sight_range"               : positive number
#     "mass"                      : positive number
#     "size"                      : positive int
#     "max_forward_force"         : positive number
#     "max_backward_force"        : positive number
#     "max_lr_force"              : positive number
#     "max_rotate_force"          : positive number
#     "max_speed"                 : positive number
#     "max_rotate_speed"          : positive number
#     "force_energy_quotient"     : positive number (can functionally be negative, but will violate thermodynamics)
#     "rotation_energy_quotient"  : positive number (can functionally be negative, but will violate thermodynamics)
# }
# <>_PARAMS = {
#     "x"                 : positive number            : starting position
#     "y"                 : positive number            : starting position
#     "initial_direction" : positive number in [0,2pi) : starting rotation
#     "attrs"             : <>_ATTRS
# }
PREY = "PREY"
PREY_ATTRS = {
    "fov"                       : 10 / 12,
    "num_rays"                  : 18,
    "sight_range"               : 75,
    "mass"                      : 2.5,
    "size"                      : 10,
    "max_forward_force"         : 3 / 1000,
    "max_backward_force"        : 3 / 1000,
    "max_lr_force"              : 2 / 1000,
    "max_rotate_force"          : 10 / 1000,
    "max_speed"                 : 30 / 1000,
    "max_rotate_speed"          : 2 * np.pi * (8 / 12) / 1000,
    "force_energy_quotient"     : 1,
    "rotation_energy_quotient"  : 0.001  # ADJUST LATER
}
PREY_PARAMS = {
    "x"                 : 400,
    "y"                 : 300,
    "initial_direction" : 0.0,
    "initial_energy"    : 100.0,
    "attrs"             : PREY_ATTRS,
    "DTYPE"             : DTYPE
}
PREDATOR = "PREDATOR"
PREDATOR_ATTRS = {
    "fov"                       : 5 / 12,
    "num_rays"                  : 10,
    "sight_range"               : 100,
    "mass"                      : 5,
    "size"                      : 10,
    "max_forward_force"         : 3.5 / 1000,
    "max_backward_force"        : 3 / 1000,
    "max_lr_force"              : 1.5 / 1000,
    "max_rotate_force"          : 11.5 / 1000,
    "max_speed"                 : 30 / 1000,
    "max_rotate_speed"          : 2 * np.pi * (9 / 12) / 1000,
    "force_energy_quotient"     : 1,
    "rotation_energy_quotient"  : 0.001  # ADJUST LATER
}
PREDATOR_PARAMS = {
    "x"                 : 200,
    "y"                 : 150,
    "initial_direction" : 0.0,
    "initial_energy"    : 100.0,
    "attrs"             : PREDATOR_ATTRS,
    "DTYPE"             : DTYPE
}

PREY_NETWORK_HYPERPARAMETERS = {

}
PREDATOR_NETWORK_HYPERPARAMETERS = {

}

MAX_TPS = 60  # TODO: Should be increased to maximum stable value for experiments
# ENVIRONMENT_PARAMETERS = {
#     "DRAG_COEFFICIENT"  : positive number (can functionally be negative, but will violate thermodynamics)
#     "DRAG_DIRECTION"    : np.array([number_a, number_b])
#                           (negative number_a for drag, positive number_a for tailwind,
#                            non-zero number_b for sidewind)
#     "MIN_TPS"           : number          : for stability
#     "EAT_EPSILON"       : number in [0,1] : proportion of overlap between creatures required for predation to occur
#     "DTYPE"             : np DTYPE        : for consistency; used in all numpy stuff
# }
ENVIRONMENT_PARAMETERS = {  # These mostly shouldn't need to change
    "DRAG_COEFFICIENT"  : 0,  # TUNE THIS (3e-10?)
    "DRAG_DIRECTION"    : np.array([-1.0, 0.0], dtype=DTYPE),
    "MIN_TPS"           : MAX_TPS,  # Should probably match MAX_TPS
    "EAT_EPSILON"       : .15,
    "DTYPE"             : DTYPE,
}

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
id_count = 0
PREY_EATEN = "PREY_EATEN"
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
SPEED_ESTIMATION_DECAY = 0.1  # float in [0,inf). 0 means perfect speed estimation,
                              #                   higher k means worse speed estimation past sight_distance
OUT_OF_ENERGY = "out of energy"
SUCCESSFUL_STEP = "success"


def get_id():
    global id_count
    id_count += 1
    return id_count


class Model:
    def __init__(self, creature_type, attrs):
        self.type = creature_type
        if creature_type == PREY:
            self.NN = PreyNetwork.PreyNetwork(PREY_NETWORK_HYPERPARAMETERS)
        elif creature_type == PREDATOR:
            self.NN = PredatorNetwork.PredatorNetwork(PREDATOR_NETWORK_HYPERPARAMETERS)
        else:
            self.NN = None
        self.sight_range = attrs["sight_range"]
        self.mass = attrs["mass"]
        self.size = attrs["size"]
        self.max_forward_force = attrs["max_forward_force"]
        self.max_backward_force = attrs["max_backward_force"]
        self.max_sideways_force = attrs["max_lr_force"]
        self.max_rotation_force = attrs["max_rotate_force"]
        self.max_velocity = attrs["max_speed"]
        self.max_rotation_speed = attrs["max_rotate_speed"]
        self.fov = attrs["fov"]
        self.creature = None  # Instantiate dynamically
        self.environment = None  # Instantiate dynamically
    
    def get_inputs(self):
        state_info = self.environment.get_state_info()
        relative_state_info = {"time": state_info["time"]}
        creature_states = state_info["creature_states"]
        relative_creature_states = []
        for state in creature_states:
            if state["id"] != self.creature.id:
                distance = np.linalg.norm(self.creature.position - state["position"])
                relative_creature_states.append({
                    "type"      : state["type"],
                    "distance"  : distance,
                    "speed"     : state["speed"] * math.pow(math.e,
                                                            -SPEED_ESTIMATION_DECAY *
                                                            (distance - self.creature.sight_range)),
                    "id"        : state["id"]
                })
        return self.NN.get_inputs(relative_state_info)


def main():
    pygame.init()
    if DRAW:
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    # queue = multiprocessing.Queue()  # multiprocessing interferes with pygame's loop
    
    running = True
    
    models = [(PREY_PARAMS, Model(PREY, PREY_ATTRS)), (PREDATOR_PARAMS, Model(PREDATOR, PREDATOR_ATTRS))]
    
    env = Environment.Environment(ENVIRONMENT_PARAMETERS, models)
    
    if DRAW:
        prey_pos = []
        pred_pos = []

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        ##############################################################################
        # multiprocessing interferes with pygame's loop                              #
        ##############################################################################
        # all_inputs = [None]*len(creatures)
        #
        # processes = []
        # for c in range(len(creatures)):
        #     process = Process(target=creatures[c].model.get_inputs, args=(queue, c))
        #     process.start()
        #     processes.append(process)
        # while queue.qsize() < len(creatures):
        #     time.sleep(.001)
        # for process in processes:
        #     process.join()
        # while not queue.empty():
        #     result = queue.get()
        #     all_inputs[result[0]] = (result[1], result[2])
        ##############################################################################
        ##############################################################################
        
        if DRAW:
            screen.fill(BLACK)
        
        delta_time = min(1.0 / 60.0 * 1000, clock.tick(MAX_TPS))
        
        step_result = env.step(delta_time, screen=screen if DRAW else None)
        
        ##################################################################################
        # This is for testing, and is specified for one prey and one predator.           #
        ##################################################################################
        if DRAW:
            prey_pos.append(env.creatures[0].position.tolist())
            pred_pos.append(env.creatures[1].position.tolist())
            for i in range(len(prey_pos)):
                pygame.draw.circle(screen, (np.array(GREEN, dtype=DTYPE) * i / len(prey_pos)).tolist(),
                                   (int(prey_pos[i][0]), int(prey_pos[i][1])), 2)
        ##################################################################################
        # End testing                                                                    #
        ##################################################################################
        
        if step_result == PREY_EATEN or step_result.endswith(OUT_OF_ENERGY):
            running = False

        if DRAW:
            pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()

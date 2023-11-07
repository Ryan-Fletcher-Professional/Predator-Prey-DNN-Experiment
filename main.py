"""
THIS FILE WRITTEN BY RYAN FLETCHER AND
"""

import pygame
import numpy as np
import math
# from multiprocessing import Process  # Interferes with PyGame. Use when not drawing?
# import time
from Globals import *
import Environment
import PreyNetwork
import PredatorNetwork

id_count = 0
SPEED_ESTIMATION_DECAY = 0.1  # float in [0,inf). 0 means perfect speed estimation,
                              #                   higher k means worse speed estimation past sight_distance


def get_id():
    global id_count
    id_count += 1
    return id_count


class Model:
    def __init__(self, creature_type, attrs):
        self.type = creature_type
        new_id = get_id()
        if creature_type == PREY:
            self.NN = PreyNetwork.PreyNetwork(PREY_NETWORK_HYPERPARAMETERS, new_id)
        elif creature_type == PREDATOR:
            self.NN = PredatorNetwork.PredatorNetwork(PREDATOR_NETWORK_HYPERPARAMETERS, new_id)
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
            distance = np.linalg.norm(self.creature.position - state["position"])
            relative_creature_states.append({
                "type"      : state["type"],
                "distance"  : distance,
                "speed"     : state["speed"] * math.pow(math.e,
                                                        -SPEED_ESTIMATION_DECAY *
                                                        (distance - self.creature.sight_range)),
                "id"        : state["id"],
                "energy"    : state["energy"]
            })
            # MAKE SURE YOU UPDATE GLOBALS.EXTERNAL_CHARACTERISTICS_PER_CREATURE and GLOBALS.INTERNAL_CHARACTERISTICS_PER_CREATURE
        relative_state_info["creature_states"] = relative_creature_states
        return self.NN.get_inputs(relative_state_info)


def main():
    pygame.init()
    if DRAW:
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    # queue = multiprocessing.Queue()  # multiprocessing interferes with pygame's loop
    
    running = True
    
    # Currently specified relative to the manually-set default x and y in <>__PARAMS in Globals
    PREY_NETWORK_HYPERPARAMETERS["dimensions"][0] = INTERNAL_CHARACTERISTICS_PER_CREATURE + EXTERNAL_CHARACTERISTICS_PER_CREATURE +\
                                                    ((NUM_TOTAL_CREATURES // 2) * EXTERNAL_CHARACTERISTICS_PER_CREATURE)
                                                    # "self" plus enemies
    PREDATOR_NETWORK_HYPERPARAMETERS["dimensions"][0] = PREY_NETWORK_HYPERPARAMETERS["dimensions"][0]
    models = [(PREY_PARAMS.copy(), Model(PREY, PREY_ATTRS)) for __ in range(NUM_TOTAL_CREATURES // 2)] +\
             [(PREDATOR_PARAMS.copy(), Model(PREDATOR, PREDATOR_ATTRS)) for __ in range(NUM_TOTAL_CREATURES // 2)]
    for i in range(1, NUM_TOTAL_CREATURES // 2):
        models[i][0]["x"] += PREY_ATTRS["size"] * 1.25
        models[i][0]["y"] += PREY_ATTRS["size"] * 1.25
    for i in range(NUM_TOTAL_CREATURES // 2, NUM_TOTAL_CREATURES):
        models[i][0]["x"] -= PREDATOR_ATTRS["size"] * 1.25
        models[i][0]["y"] -= PREDATOR_ATTRS["size"] * 1.25
    
    env = Environment.Environment(ENVIRONMENT_PARAMETERS, models)
    
    if DRAW:
        FOCUS = 0  # Index of focused creature in environment's list
        focus_pos = []

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
        # print(delta_time * MAX_TPS / 1000)  # ~=1 if on target TPS
        
        step_result = env.step(delta_time, screen=screen if DRAW else None)
        
        ##################################################################################
        # This is for testing.                                                           #
        ##################################################################################
        if DRAW:
            focus_pos.append(env.creatures[FOCUS].position.tolist())
            for i in range(len(focus_pos)):
                pygame.draw.circle(screen, (np.array(GREEN, dtype=DTYPE) * i / len(focus_pos)).tolist(),
                                   (int(focus_pos[i][0]), int(focus_pos[i][1])), 2)
        ##################################################################################
        # End testing                                                                    #
        ##################################################################################
        if step_result == ALL_PREY_EATEN or step_result.endswith(OUT_OF_ENERGY):
            running = False

        if DRAW:
            pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()

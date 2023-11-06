"""
THIS FILE WRITTEN BY RYAN FLETCHER AND
"""

import pygame
import numpy as np
import math
# from multiprocessing import Process  # Interferes with PyGame. Use when not drawing?
# import time
import Globals
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
        if creature_type == Globals.PREY:
            self.NN = PreyNetwork.PreyNetwork(Globals.PREY_NETWORK_HYPERPARAMETERS, new_id)
        elif creature_type == Globals.PREDATOR:
            self.NN = PredatorNetwork.PredatorNetwork(Globals.PREDATOR_NETWORK_HYPERPARAMETERS, new_id)
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
        relative_state_info["creature_states"] = relative_creature_states
        return self.NN.get_inputs(relative_state_info)


def main():
    pygame.init()
    if Globals.DRAW:
        screen = pygame.display.set_mode((Globals.SCREEN_WIDTH, Globals.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    # queue = multiprocessing.Queue()  # multiprocessing interferes with pygame's loop
    
    running = True
    
    # Specified for one prey and one predator. CHANGE FOR EXPERIMENTS!
    Globals.PREY_NETWORK_HYPERPARAMETERS["dimensions"][0] = 7
    Globals.PREDATOR_NETWORK_HYPERPARAMETERS["dimensions"][0] = 7
    models = [(Globals.PREY_PARAMS, Model(Globals.PREY, Globals.PREY_ATTRS)), (Globals.PREDATOR_PARAMS, Model(Globals.PREDATOR, Globals.PREDATOR_ATTRS))]
    
    env = Environment.Environment(Globals.ENVIRONMENT_PARAMETERS, models)
    
    if Globals.DRAW:
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
        
        if Globals.DRAW:
            screen.fill(Globals.BLACK)
        
        delta_time = min(1.0 / 60.0 * 1000, clock.tick(Globals.MAX_TPS))
        
        step_result = env.step(delta_time, screen=screen if Globals.DRAW else None)
        
        ##################################################################################
        # This is for testing, and is specified for one prey and one predator.           #
        ##################################################################################
        if Globals.DRAW:
            prey_pos.append(env.creatures[0].position.tolist())
            pred_pos.append(env.creatures[1].position.tolist())
            for i in range(len(prey_pos)):
                pygame.draw.circle(screen, (np.array(Globals.GREEN, dtype=Globals.DTYPE) * i / len(prey_pos)).tolist(),
                                   (int(prey_pos[i][0]), int(prey_pos[i][1])), 2)
        ##################################################################################
        # End testing                                                                    #
        ##################################################################################
        
        if step_result == Globals.PREY_EATEN or step_result.endswith(Globals.OUT_OF_ENERGY):
            running = False

        if Globals.DRAW:
            pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()

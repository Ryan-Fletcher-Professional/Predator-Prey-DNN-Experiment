"""
THIS FILE WRITTEN BY RYAN FLETCHER AND
"""


import multiprocessing
import numpy as np
import math
import time
from Globals import *
if DRAW:
    import pygame
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
    
    def get_inputs(self, queue=None, index=None):
        state_info = self.environment.get_state_info()
        relative_state_info = {"time": state_info["time"]}
        creature_states = state_info["creature_states"]
        sights = self.creature.see_others(self.environment)  # [(id, distance), (id, distance), ...]
        relative_creature_states = []
        for state in creature_states:
            hit = False
            if not (state["id"] == self.creature.id):
                for sight in sights:
                    if sight[0] == state["id"]:
                        hit = True
                        perceived_type = state["type"]
                        distance = sight[1]
                        relative_speed = np.linalg.norm(self.creature.velocity - state["velocity"]) * math.pow(math.e, -SPEED_ESTIMATION_DECAY * (distance - self.creature.sight_range))
            if not hit:
                # These may need to be implemented differently if the networks don't like implicit multimodality
                perceived_type = UNKNOWN_TYPE
                distance = 0.0
                relative_speed = 0.0
                
            relative_creature_states.append({
                "type"           : state["type"],
                "perceived_type" : perceived_type,
                "distance"       : distance,
                "relative_speed" : relative_speed,
                "id"             : state["id"],
                "energy"         : state["energy"],
                "stun"           : state["stun"]
            })
            # MAKE SURE YOU UPDATE GLOBALS.EXTERNAL_CHARACTERISTICS_PER_CREATURE and GLOBALS.INTERNAL_CHARACTERISTICS_PER_CREATURE
        relative_state_info["creature_states"] = relative_creature_states
        inputs = self.NN.get_inputs(relative_state_info) if self.creature.alive else NETWORK_OUTPUT_DEFAULT
        if (queue is None) or (index is None):
            return inputs
        queue.put((index, inputs))


def main():
    if DRAW:
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        clock = pygame.time.Clock()
    
    running = True
    
    # Currently specified relative to the manually-set default x and y in <>__PARAMS in Globals
    PREY_NETWORK_HYPERPARAMETERS["dimensions"][0] = INTERNAL_CHARACTERISTICS_PER_CREATURE +\
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
        focus_creature = FOCUS_CREATURE  # Index of focused creature in environment's list
        focus_pos = []

    while running:
        try:
            if DRAW:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                screen.fill(BACKGROUND_COLOR)
                delta_time = min(1.0 / env.MIN_TPS * 1000, clock.tick(MAX_TPS))
                # print(delta_time * MAX_TPS / 1000)  # ~=1 if on target TPS
            else:
                
                delta_time = 1.0 / MAX_TPS * 1000
            
            step_result = env.step(delta_time, screen=screen if DRAW else None)
            
            ##################################################################################
            # This is for testing.                                                           #
            ##################################################################################
            if DRAW:
                if(env.creatures[focus_creature].alive):
                    focus_pos.append(env.creatures[focus_creature].position.tolist())
                for i in range(min(len(focus_pos), FOCUS_PATH_LENGTH)):
                    pygame.draw.circle(screen, (np.array(GREEN, dtype=DTYPE) * i / min(len(focus_pos), FOCUS_PATH_LENGTH)).tolist(),
                                    (int(focus_pos[max(0, len(focus_pos) - FOCUS_PATH_LENGTH) + i][0]),
                                        int(focus_pos[max(0, len(focus_pos) - FOCUS_PATH_LENGTH) + i][1])),
                                    2)
                env.creatures[focus_creature].draw(screen)
            ##################################################################################
            # End testing                                                                    #
            ##################################################################################
            
            if step_result == ALL_PREY_EATEN:
                running = False

            if DRAW:
                pygame.display.flip()
        except KeyboardInterrupt:
            running = False

    if DRAW:
        pygame.quit()
    if USE_MULTIPROCESSING:
        for _ in range(env.expected_num_children):
                env.task_queue.put(None)
        time.sleep(.1)
        for process in multiprocessing.active_children():
                process.join()
    print("Caught KeyboardInterrupt")


if __name__ == "__main__":
    print("Setting up...")
    main()

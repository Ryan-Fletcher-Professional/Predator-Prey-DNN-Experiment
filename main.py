"""
THIS FILE WRITTEN BY RYAN FLETCHER AND
"""


import multiprocessing
import pickle
import numpy as np
import math
import random
import copy
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
    def __init__(self, creature_type, attrs, hyperparameters, network=None):
        self.type = creature_type
        new_id = get_id()
        if not (network is None):
            self.NN = network
        elif creature_type == PREY:
            self.NN = PreyNetwork.PreyNetwork(hyperparameters, new_id)
        elif creature_type == PREDATOR:
            self.NN = PredatorNetwork.PredatorNetwork(hyperparameters, new_id)
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
    experiments = []
    previous_experiment = DEFAULT_EXPERIMENT
    for i in range(5):#100):
        #############################################################################################################
        experiment = copy.deepcopy(previous_experiment)
        # For these two lines the copied index can be changed, but that's it.
        experiment[PREY_PARAMS_NAME]["attrs"] = copy.deepcopy(previous_experiment[PREY_PARAMS_NAME]["attrs"])
        experiment[PREDATOR_PARAMS_NAME]["attrs"] = copy.deepcopy(previous_experiment[PREDATOR_PARAMS_NAME]["attrs"])
        #############################################################################################################
        
        # Modify experiment parameters
        # In these experiments we're starting the creatures off with low energy so they can learn what it means.
        # We're also starting with a smaller screen size so loss values are bigger early on to encourage learning control.
        experiment[PREY_PARAMS_NAME]["initial_energy"] = min(100, math.pow(1.09, i))  # Should hit 100 at the 55th experiment
        experiment[PREDATOR_PARAMS_NAME]["initial_energy"] = min(100, math.pow(1.09, i))  # Should hit 100 at the 55th experiment
        experiment[ENV_PARAMS_NAME]["screen_width"] = min(DEFAULT_SCREEN_WIDTH, 300 * ((i / 4.0) + 1))
        experiment[ENV_PARAMS_NAME]["screen_height"] = min(DEFAULT_SCREEN_HEIGHT, 300 * ((i / 4.0) + 1))
        experiment[PREY_HYPERPARAMS_NAME]["dimensions"][0] = INTERNAL_CHARACTERISTICS_PER_CREATURE +\
                                                             ((experiment[ENV_PARAMS_NAME]["num_predators"]) * EXTERNAL_CHARACTERISTICS_PER_CREATURE)
                                                             # "self" plus enemies
        experiment[PREDATOR_HYPERPARAMS_NAME]["dimensions"][0] = INTERNAL_CHARACTERISTICS_PER_CREATURE +\
                                                                 ((experiment[ENV_PARAMS_NAME]["num_preys"]) * EXTERNAL_CHARACTERISTICS_PER_CREATURE)
                                                                 # "self" plus enemies
        
        #############################################################################################################
        experiments.append(experiment)
        previous_experiment = experiment
        #############################################################################################################
        
    experiment_results = []
    for i in range(len(experiments)):
        print(f"Starting experiment {i + 1}")
        experiment = experiments[i]
        
        if DRAW:
            pygame.init()
            screen = pygame.display.set_mode((experiment[ENV_PARAMS_NAME["screen_width"]], experiment[ENV_PARAMS_NAME["screen_height"]]))
            clock = pygame.time.Clock()
        
        running = True
        
        # Currently specified relative to the manually-set default x and y in <>__PARAMS in Globals
        num_preys = experiment[ENV_PARAMS_NAME]["num_preys"]
        num_predators = experiment[ENV_PARAMS_NAME]["num_predators"]
        models = [(copy.deepcopy(experiment[PREY_PARAMS_NAME]), Model(PREY, experiment[PREY_ATTRS_NAME], experiment[PREY_HYPERPARAMS_NAME], network=None if ((not experiment[KEEP_WEIGHTS]) or (i == 0)) else experiment_results[i - 1][PREY][j % len(experiment_results[i - 1][PREY])]["NETWORK"])) for j in range(num_preys)] +\
                 [(copy.deepcopy(experiment[PREDATOR_PARAMS_NAME]), Model(PREDATOR, experiment[PREDATOR_ATTRS_NAME], experiment[PREDATOR_HYPERPARAMS_NAME], network=None if ((not experiment[KEEP_WEIGHTS]) or (i == 0)) else experiment_results[i - 1][PREDATOR][j % len(experiment_results[i - 1][PREDATOR])]["NETWORK"])) for j in range(num_predators)]
        
        env = Environment.Environment(experiment[ENV_PARAMS_NAME], models)
        
        screen_width = env.screen_width
        screen_height = env.screen_height
        for i in range(num_preys + num_predators):
            tryX = None
            tryY = None
            too_close = True
            while too_close:
                tryX = 1 + (random.random() * (screen_width - 2))
                tryY = 1 + (random.random() * (screen_height - 2))
                too_close = False
                for model in models:
                    if np.linalg.norm(np.array([tryX, tryY], dtype=experiment[ENV_PARAMS_NAME]["DTYPE"]) - np.array([model[0]["x"], model[0]["y"]], dtype=experiment[ENV_PARAMS_NAME]["DTYPE"])) < DEFAULT_CREATURE_SIZE + PLACEMENT_BUFFER:
                        too_close = True
                        break
            models[i][0]["x"] = tryX
            models[i][0]["y"] = tryY
        
        if DRAW:
            focus_creature = FOCUS_CREATURE  # Index of focused creature in environment's list
            focus_pos = []
        
        env.start_real_time = time.time()

        end_reason = None
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
                    end_reason = ALL_PREY_EATEN

                if DRAW:
                    pygame.display.flip()
            except KeyboardInterrupt:
                running = False
                end_reason = "TERMINATED"
                print("Caught KeyboardInterrupt")
            if (env.time / 1000) >= experiment[MAX_SIM_SECONDS]:
                running = False
                end_reason = MAX_SIM_SECONDS
        
        results = {}
        results["real_time"] = time.time() - env.start_real_time
        results["sim_time"] = env.time
        results["end_reason"] = end_reason
        results[PREY] = [ creature.get_results() for creature in filter(FILTER_OUT_PREDATOR_OBJECTS, env.creatures) ]
        results[PREDATOR] = [ creature.get_results() for creature in filter(FILTER_OUT_PREY_OBJECTS, env.creatures) ]
        experiment_results.append(results)
        
        if DRAW:
            pygame.quit()
        if USE_MULTIPROCESSING:
            for _ in range(env.expected_num_children):
                    env.task_queue.put(None)
            time.sleep(.1)
            for process in multiprocessing.active_children():
                    process.join()

    print(experiment_results)
    total_sim_time = 0.0
    total_real_time = 0.0
    for i in range(len(experiment_results)):
        total_sim_time += experiment_results[i]["sim_time"]
        total_real_time += experiment_results[i]["real_time"]
        print(f"End reason {i + 1}: {experiment_results[i]['end_reason']}")
    print(f"Total simulated time:    {int((total_sim_time / 1000) // 3600)}h {int(((total_sim_time / 1000) % 3600) // 60)}m {((((total_sim_time / 1000) % 3600) % 60)):.3f}s\nTotal real time:         {int(total_real_time // 3600)}h {int((total_real_time % 3600) // 60)}m {(((total_real_time % 3600) % 60)):.3f}s")
    with open('serialized_data.pkl', 'wb') as file:
        pickle.dump(experiment_results, file)
    # To read experiment_results later:
    # with open('serialized_data.pkl', 'rb') as file:
    #   loaded_object = pickle.load(file)


if __name__ == "__main__":
    print("Setting up...")
    main()

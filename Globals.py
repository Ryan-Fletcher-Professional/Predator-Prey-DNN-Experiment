import math
import numpy as np
import pyautogui as ag


DRAW = False
ALLOW_PREDATOR_ENERGY_DEATH = True
DEFAULT_STORE_CREATURE_POSITIONS = math.inf
FIRST = "FIRST"
RECENT = "RECENT"
DEFAULT_STORE_CREATURE_POSITIONS_REFERENCE = RECENT
# Currently multiprocessing is ~40x slower than serial network feeding :( Maybe it will be usefull for large quantities of creatures?
USE_MULTIPROCESSING = (not DRAW) and False  # DO NOT REMOVE "(not DRAW) and"; multiprocessing interferes with pygame's loop
__width, __height = ag.size()
__scale_to_window_size = DRAW
__draw_size_coefficient = 0.8
__noscale_size_coefficient = 10
DEFAULT_CREATURE_SIZE = 10
PLACEMENT_BUFFER = 2
DEFAULT_NUM_TOTAL_CREATURES = 6
__override_noscale_size = (True, 1800, 1800)

if __scale_to_window_size:
    DEFAULT_SCREEN_WIDTH = __width * __draw_size_coefficient
    DEFAULT_SCREEN_HEIGHT = __height * __draw_size_coefficient
elif not __override_noscale_size[0]:
    DEFAULT_SCREEN_WIDTH = DEFAULT_CREATURE_SIZE * (DEFAULT_NUM_TOTAL_CREATURES ** 2) * __noscale_size_coefficient / 2
    DEFAULT_SCREEN_HEIGHT = DEFAULT_CREATURE_SIZE * (DEFAULT_NUM_TOTAL_CREATURES ** 2) * __noscale_size_coefficient / 2
else:
    DEFAULT_SCREEN_WIDTH = __override_noscale_size[1]
    DEFAULT_SCREEN_HEIGHT = __override_noscale_size[2]
DTYPE = np.float64
PRINT_PROGRESS_STEPS = 9000000
ALWAYS_OVERRIDE_PREY_MOVEMENT = False
FOCUS_CREATURE = 0  # Index in environment.creatures
PREY = -1.0
UNKNOWN_TYPE = 0.0
PREDATOR = 1.0
MAX_TPS = 60  # Maximum physics ticks per SIMULATED second.
              # Increasing this number will not increase simulation speed. Increase creature speed to do that.
              # (Make sure to inrease MAX_TPS if needed for numerical stability.)
ALL_PREYS_DEAD = "ALL PREYS_DEAD"
ALL_PREDATORS_DEAD = "ALL PREDATORS_DEAD"
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (100, 100, 100)
BACKGROUND_COLOR = BLACK
CREATURE_COLORS = { PREY: GREEN, PREDATOR: RED }
SUCCESSFUL_STEP = "success"
RECIPROCAL_MODE = "reciprocal"
SUBTRACT_MODE = "subtract"
USE_GPU = False
REFERENCE_ANGLE = [1.0, 0.0]
STUN_TICK_TIME = (85 * 60) / MAX_TPS  # ms : currently ~5 ticks
STUN_IGNORE_PUNISHMENT_QUOTIENT = 0.0  # multiplier for adding stun time when creature tries to move while already stunned
NETWORK_OUTPUT_DEFAULT = ([0.0, 0.0, 0.0], math.inf)  # Mainly for dead creatures
DRAG_COEFFICIENT = .015
DRAG_MINIMUM_SPEED = 30 * .0025 / MAX_TPS
FOCUS_PATH_LENGTH = 1000

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
#     "x"                 : positive number                         : starting position
#     "y"                 : positive number                         : starting position
#     "initial_direction" : positive number in [0,2pi)              : starting rotation
#     "initial_energy"    : positive number                         : starting energy level
#     :store_positions"   : (non-negative number, <FIRST/RECENT>)   : [0]: determines number of stored positions for serialization (math.inf accepted), [1]: determines if first [0] positions or last [0] positions are stored
#     "DTYPE"             : numpy dtype                             : used for all numpy in Creature class
# }
PREY_ATTRS_NAME = "PREY_ATTRS"
DEFAULT_PREY_ATTRS = {
    "fov"                       : 5 / 6,
    "num_rays"                  : 23,
    "sight_range"               : 143,
    "mass"                      : 2.5,
    "size"                      : DEFAULT_CREATURE_SIZE,
    "max_forward_force"         : 3 / 1000,
    "max_backward_force"        : 3 / 1000,
    "max_lr_force"              : 2 / 1000,
    "max_rotate_force"          : 10 / 1000,
    "max_speed"                 : 30 / 1000,
    "max_rotate_speed"          : 2 * np.pi * (8 / 12) / 1000,
    "force_energy_quotient"     : 1,
    "rotation_energy_quotient"  : .01 / (2 * math.pi)
}
PREY_PARAMS_NAME = "PREY_PARAMS"
DEFAULT_PREY_PARAMS = {
    "x"                 : (DEFAULT_SCREEN_WIDTH // 2) + 150,
    "y"                 : (DEFAULT_SCREEN_HEIGHT // 2) + 150,
    "initial_direction" : 0.0,
    "initial_energy"    : 100.0,
    "store_positions"   : (DEFAULT_STORE_CREATURE_POSITIONS, DEFAULT_STORE_CREATURE_POSITIONS_REFERENCE),
    "DTYPE"             : DTYPE
}
PREDATOR_ATTRS_NAME = "PRED_ATTRS"
DEFAULT_PREDATOR_ATTRS = {
    "fov"                       : 5 / 12,
    "num_rays"                  : 13,
    "sight_range"               : 143,
    "mass"                      : 5,
    "size"                      : DEFAULT_CREATURE_SIZE,
    "max_forward_force"         : 3.5 / 1000,
    "max_backward_force"        : 3 / 1000,
    "max_lr_force"              : 1.5 / 1000,
    "max_rotate_force"          : 11.5 / 1000,
    "max_speed"                 : 30 / 1000,
    "max_rotate_speed"          : 2 * np.pi * (9 / 12) / 1000,
    "force_energy_quotient"     : 1,
    "rotation_energy_quotient"  : .01 / (2 * np.pi)
}
PREDATOR_PARAMS_NAME = "PRED_PARAMS"
DEFAULT_PREDATOR_PARAMS = {
    "x"                 : (DEFAULT_SCREEN_WIDTH // 2) - 150,
    "y"                 : (DEFAULT_SCREEN_HEIGHT // 2) - 150,
    "initial_direction" : 0.0,
    "initial_energy"    : 100.0,
    "store_positions"   : (DEFAULT_STORE_CREATURE_POSITIONS, DEFAULT_STORE_CREATURE_POSITIONS_REFERENCE),
    "DTYPE"             : DTYPE
}

PREDATION_ENERGY_BOOST = DEFAULT_PREY_PARAMS["initial_energy"] * 0.1
# ENVIRONMENT_PARAMETERS = {
#     "DRAG_COEFFICIENT"  : positive number
#     "MIN_TPS"           : number          : for stability
#     "EAT_EPSILON"       : number in [0,1] : proportion of overlap between creatures required for predation to occur
#     "DTYPE"             : np DTYPE        : for consistency; used in all numpy stuff
#     "screen_width"      : positive int
#     "screen_height"     : positive int
#     "num_preys"         : non-negative int
#     "num_predators"     : non-negative int
# }
ENV_PARAMS_NAME = "ENV_PARAMS"
DEFAULT_ENVIRONMENT_PARAMETERS = {  # These mostly shouldn't need to change
    "DRAG_COEFFICIENT"  : DRAG_COEFFICIENT,
    "MIN_TPS"           : MAX_TPS,  # Used only to prevent pygame slowdowns. Should probably match MAX_TPS
    "EAT_EPSILON"       : .15,
    "DTYPE"             : DTYPE,
    "screen_width"      : DEFAULT_SCREEN_WIDTH,
    "screen_height"     : DEFAULT_SCREEN_HEIGHT,
    "num_preys"         : 3,
    "num_predators"     : 3
}

DEFAULT_SELF_INPUTS = ["stun", "energy"]
DEFAULT_OTHER_INPUTS = ["relative_speed_x", "relative_speed_y", "perceived_type", "distance"]
DEFAULT_OUTPUT_DIM = 4
PREY_HYPERPARAMS_NAME = "PREY_HYPERPARAMS"
DEFAULT_PREY_NETWORK_HYPERPARAMETERS = {
    "input_keys"    : (DEFAULT_SELF_INPUTS, DEFAULT_OTHER_INPUTS),
    "dimensions"    : [len(DEFAULT_SELF_INPUTS) + (len(DEFAULT_OTHER_INPUTS) * DEFAULT_ENVIRONMENT_PARAMETERS["num_predators"]), 32, 16, 8, DEFAULT_OUTPUT_DIM],
    "print_state"   : True,
    "print_loss"    : True,
    "loss_mode"     : SUBTRACT_MODE
}
PREDATOR_HYPERPARAMS_NAME = "PRED_HYPERPARAMS"
DEFAULT_PREDATOR_NETWORK_HYPERPARAMETERS = {
    "input_keys"    : (DEFAULT_SELF_INPUTS, DEFAULT_OTHER_INPUTS),
    "dimensions"    : [len(DEFAULT_SELF_INPUTS) + (len(DEFAULT_OTHER_INPUTS) * DEFAULT_ENVIRONMENT_PARAMETERS["num_preys"]), 32, 16, 8, DEFAULT_OUTPUT_DIM],
    "print_state"   : True,
    "print_loss"    : True,
}

NUM_TOTAL_CREATURES = DEFAULT_ENVIRONMENT_PARAMETERS["num_preys"] + DEFAULT_ENVIRONMENT_PARAMETERS["num_predators"]
NUM_SUBPROCESSES = NUM_TOTAL_CREATURES

EXPERIMENT_LABELS = [PREY_ATTRS_NAME, PREY_PARAMS_NAME, PREY_HYPERPARAMS_NAME,
                     PREDATOR_ATTRS_NAME, PREDATOR_PARAMS_NAME, PREDATOR_HYPERPARAMS_NAME,
                     ENV_PARAMS_NAME]
EXPERIMENT_DICTS = [DEFAULT_PREY_ATTRS, DEFAULT_PREY_PARAMS, DEFAULT_PREY_NETWORK_HYPERPARAMETERS,
                    DEFAULT_PREDATOR_ATTRS, DEFAULT_PREDATOR_PARAMS, DEFAULT_PREDATOR_NETWORK_HYPERPARAMETERS,
                    DEFAULT_ENVIRONMENT_PARAMETERS]

KEEP_WEIGHTS = "KEEP_WEIGHTS"
MAX_SIM_SECONDS = "MAX_SIM_SECONDS"
DEFAULT_EXPERIMENT = { **{ label : dictionary for label, dictionary in zip(EXPERIMENT_LABELS, EXPERIMENT_DICTS) },
                       **{ KEEP_WEIGHTS : True, MAX_SIM_SECONDS : 1800 } }


def NORMALIZE(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def ANGLE_BETWEEN(v1, v2, DTYPE=DTYPE):
    """
    Got this from StackOverflow
    """
    v1_u = NORMALIZE(v1)
    v2_u = NORMALIZE(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0, dtype=DTYPE), dtype=DTYPE) % (2 * np.pi)
    return angle


def ANGLE_TO_VEC(angle, DTYPE=DTYPE):
    return np.array([math.cos(angle), math.sin(angle)], dtype=DTYPE)


def FILTER_IN_PREDATOR_DICTS(creature):
    return creature["type"] == PREDATOR


def FILTER_IN_PREY_DICTS(creature):
    return creature["type"] == PREY

def FILTER_IN_PERCEIVED_PREDATOR_DICTS(creature):
    return creature["perceived_type"] == PREDATOR


def FILTER_IN_PERCEIVED_PREY_DICTS(creature):
    return creature["perceived_type"] == PREY


def FILTER_IN_PREDATOR_OBJECTS(creature):
    return creature.model.type == PREDATOR


def FILTER_IN_PREY_OBJECTS(creature):
    return creature.model.type == PREY

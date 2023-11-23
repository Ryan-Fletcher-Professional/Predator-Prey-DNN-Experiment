import math
import numpy as np
import pyautogui as ag


DRAW = False
# Currently multiprocessing is ~40x slower than serial network feeding :( Maybe it will be usefull for large quantities of creatures?
USE_MULTIPROCESSING = (not DRAW) and False  # DO NOT REMOVE "(not DRAW) and"; multiprocessing interferes with pygame's loop
__width, __height = ag.size()
__scale_to_window_size = DRAW
__draw_size_coefficient = 0.8
__noscale_size_coefficient = 10
DEFAULT_CREATURE_SIZE = 10
NUM_TOTAL_CREATURES = 6  # Currently specified for only an even number of creatures!
__override_noscale_size = (True, 1800, 1800)
EXTERNAL_CHARACTERISTICS_PER_CREATURE = 3
INTERNAL_CHARACTERISTICS_PER_CREATURE = 3

if __scale_to_window_size:
    SCREEN_WIDTH = __width * __draw_size_coefficient
    SCREEN_HEIGHT = __height * __draw_size_coefficient
elif not __override_noscale_size[0]:
    SCREEN_WIDTH = DEFAULT_CREATURE_SIZE * (NUM_TOTAL_CREATURES ** 2) * __noscale_size_coefficient / 2
    SCREEN_HEIGHT = DEFAULT_CREATURE_SIZE * (NUM_TOTAL_CREATURES ** 2) * __noscale_size_coefficient / 2
else:
    SCREEN_WIDTH = __override_noscale_size[1]
    SCREEN_HEIGHT = __override_noscale_size[2]
DTYPE = np.float64
PRINT_PROGRESS_STEPS = 100
ALWAYS_OVERRIDE_PREY_MOVEMENT = False
FOCUS_CREATURE = 0  # Index in environment.creatures
PREY = -1.0
UNKNOWN_TYPE = 0.0
PREDATOR = 1.0
MAX_TPS = 60  # Maximum physics ticks per SIMULATED second.
              # Increasing this number will not increase simulation speed. Increase creature speed to do that.
              # (Make sure to inrease MAX_TPS if needed for numerical stability.)
ALL_PREY_EATEN = "ALL PREY_EATEN"
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
STUN_IGNORE_PUNISHMENT_QUOTIENT = 0.5  # multiplier for adding stun time when creature tries to move while already stunned
NETWORK_OUTPUT_DEFAULT = [0.0, 0.0, 0.0]  # Mainly for dead creatures
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
#     "x"                 : positive number            : starting position
#     "y"                 : positive number            : starting position
#     "initial_direction" : positive number in [0,2pi) : starting rotation
#     "attrs"             : <>_ATTRS
# }
PREY_ATTRS = {
    "fov"                       : 5 / 6,
    "num_rays"                  : 23,
    "sight_range"               : 75,
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
PREY_PARAMS = {
    "x"                 : (SCREEN_WIDTH // 2) + 150,
    "y"                 : (SCREEN_HEIGHT // 2) + 150,
    "initial_direction" : 0.0,
    "initial_energy"    : 100.0,
    "attrs"             : PREY_ATTRS,
    "DTYPE"             : DTYPE
}
PREDATOR_ATTRS = {
    "fov"                       : 5 / 12,
    "num_rays"                  : 13,
    "sight_range"               : 100,
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
PREDATOR_PARAMS = {
    "x"                 : (SCREEN_WIDTH // 2) - 150,
    "y"                 : (SCREEN_HEIGHT // 2) - 150,
    "initial_direction" : 0.0,
    "initial_energy"    : 100.0,
    "attrs"             : PREDATOR_ATTRS,
    "DTYPE"             : DTYPE
}

PREY_NETWORK_HYPERPARAMETERS = {
    "dimensions" : [-1, 10, 10, 10, 4],
    "loss_mode"  : SUBTRACT_MODE,
}
PREDATOR_NETWORK_HYPERPARAMETERS = {
    "dimensions" : [-1, 10, 10, 10, 4]
}

# ENVIRONMENT_PARAMETERS = {
#     "DRAG_COEFFICIENT"  : positive number
#     "MIN_TPS"           : number          : for stability
#     "EAT_EPSILON"       : number in [0,1] : proportion of overlap between creatures required for predation to occur
#     "DTYPE"             : np DTYPE        : for consistency; used in all numpy stuff
# }
ENVIRONMENT_PARAMETERS = {  # These mostly shouldn't need to change
    "DRAG_COEFFICIENT"  : DRAG_COEFFICIENT,
    "MIN_TPS"           : MAX_TPS,  # Used only to prevent pygame slowdowns. Should probably match MAX_TPS
    "EAT_EPSILON"       : .15,
    "DTYPE"             : DTYPE,
}


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


def FILTER_OUT_PREY_DICTS(creature):
    return creature["type"] == PREDATOR


def FILTER_OUT_PREDATOR_DICTS(creature):
    return creature["type"] == PREY


def FILTER_OUT_PREY_OBJECTS(creature):
    return creature.model.type == PREDATOR


def FILTER_OUT_PREDATOR_OBJECTS(creature):
    return creature.model.type == PREY

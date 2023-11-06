import numpy as np

DTYPE = np.float64
DRAW = True
ALWAYS_OVERRIDE_PREY_MOVEMENT = True
PREY = "PREY"
PREDATOR = "PREDATOR"
MAX_TPS = 60  # TODO: Should be increased to maximum stable value for experiments
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
PREY_EATEN = "PREY_EATEN"
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
OUT_OF_ENERGY = "out of energy"
SUCCESSFUL_STEP = "success"
RECIPROCAL_MODE = "reciprocal"
SUBTRACT_MODE = "subtract"
USE_GPU = False

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
    "loss_mode" : SUBTRACT_MODE,
}
PREDATOR_NETWORK_HYPERPARAMETERS = {

}

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
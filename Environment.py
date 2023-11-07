"""
THIS FILE WRITTEN BY RYAN FLETCHER
"""

import math
import numpy as np
import pygame
from Globals import *
import main


def angle_to_vec(angle, DTYPE=DTYPE):
    return np.array([math.cos(angle), math.sin(angle)], dtype=DTYPE)


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def angle_between(v1, v2, DTYPE=DTYPE):
    """
    Got this from StackOverflow
    """
    v1_u = normalize(v1)
    v2_u = normalize(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0, dtype=DTYPE), dtype=DTYPE) % (2 * np.pi)
    return angle


def rotate_vector(v, angle, DTYPE=DTYPE):
    rotation_matrix = np.array([
        [np.cos(angle, dtype=DTYPE), -np.sin(angle, dtype=DTYPE)],
        [np.sin(angle, dtype=DTYPE), np.cos(angle, dtype=DTYPE)]
    ], dtype=DTYPE)
    return rotation_matrix @ v


def copy_dir(from_vec, to_vec, a=None, DTYPE=DTYPE):
    if a is not None:
        angle = a
    else:
        b = normalize(from_vec)
        angle = angle_between(np.array(REFERENCE_ANGLE, dtype=DTYPE), b)
    return rotate_vector(to_vec, angle)


class Creature:
    def __init__(self, params, model, creature_id=None):
        """
        :param params: See main
        :param model: main.Model
        :param creature_id: Optional int (will call main.get_id() by default)
        """
        self.id = creature_id if creature_id is not None else main.get_id()
        self.DTYPE = params["DTYPE"]
        attrs = params["attrs"]
        self.sight_range = attrs["sight_range"]
        self.mass = attrs["mass"]
        self.size = attrs["size"]
        self.max_forward_force = attrs["max_forward_force"]
        self.max_backward_force = attrs["max_backward_force"]
        self.max_sideways_force = attrs["max_lr_force"]
        self.max_rotation_force = attrs["max_rotate_force"]
        self.max_velocity = attrs["max_speed"]
        self.position = np.array([params["x"], params["y"]], dtype=self.DTYPE)
        self.velocity = np.array([0, 0], dtype=self.DTYPE)
        self.acceleration = np.array([0, 0], dtype=self.DTYPE)
        self.max_rotation_speed = attrs["max_rotate_speed"]
        self.initial_direction = params["initial_direction"]
        self.direction = self.initial_direction
        self.energy = params["initial_energy"]
        self.force_energy_quotient = attrs["force_energy_quotient"]
        self.rotation_speed = 0.0
        self.rotation_energy_quotient = attrs["rotation_energy_quotient"]
        self.rays = [Ray(self.position, angle) for angle in np.linspace(-(attrs["fov"] / 2) * 2 * np.pi,
                                                                        (attrs["fov"] / 2) * 2 * np.pi,
                                                                        num=attrs["num_rays"], dtype=self.DTYPE)]
        self.model = model
    
    def apply_force(self, force):
        """
        :param force: 2d-array : [ float forward force, float rightward force ]
        """
        f = np.clip(force, [-self.max_backward_force, -self.max_sideways_force],
                    [self.max_forward_force, self.max_sideways_force], dtype=self.DTYPE)
        self.energy -= np.linalg.norm(f) * self.force_energy_quotient
        self.acceleration += copy_dir(None, f / self.mass, a=self.direction)
    
    def update_velocity(self, delta_time):
        """
        :param delta_time: float milliseconds
        """
        self.velocity += self.acceleration * delta_time
        velocity_magnitude = np.linalg.norm(self.velocity) / delta_time
        if velocity_magnitude > self.max_velocity:
            self.velocity *= self.max_velocity / velocity_magnitude
    
    def update_rotation_speed(self, speed):
        """
        :param speed: float ∝ radians per millisecond
        """
        self.rotation_speed = speed
        if abs(self.rotation_speed) > self.max_rotation_speed:
            self.rotation_speed = math.copysign(self.max_rotation_speed, self.rotation_speed)
    
    def rotate(self, delta_time):
        """
        :param delta_time: float milliseconds
        """
        self.energy -= abs(self.rotation_speed) * delta_time * self.rotation_energy_quotient
        for ray in self.rays:
            ray.angle = (ray.angle + (self.rotation_speed * delta_time)) % (2 * np.pi)
        self.direction = (self.direction + (self.rotation_speed * delta_time)) % (2 * np.pi)
    
    def update_position(self, delta_time):
        """
        :param delta_time: float milliseconds
        """
        self.position += self.velocity * delta_time
        self.acceleration = 0
    
    def draw(self, screen):
        pygame.draw.circle(screen, WHITE, self.position.astype(int), self.size)
        bulge_radius = DEFAULT_CREATURE_SIZE
        bulge_position = self.position + angle_to_vec(self.direction) * bulge_radius
        # Draw the bulge as a smaller circle or an arc
        bulge_size = 4
        pygame.draw.circle(screen, WHITE, bulge_position.astype(int), bulge_size)
        for ray in self.rays:
            ray.cast(screen, self.sight_range)


class Environment:
    def __init__(self, env_params, models):
        self.DRAG_COEFFICIENT = env_params["DRAG_COEFFICIENT"]
        self.DRAG_DIRECTION = env_params["DRAG_DIRECTION"]
        self.MIN_TPS = env_params["MIN_TPS"]
        self.EAT_EPSILON = env_params["EAT_EPSILON"]
        self.DTYPE = env_params["DTYPE"]
        self.creatures = []
        self.time = 0.0
        for model in models:
            self.creatures.append(Creature(model[0], model[1], creature_id=model[1].NN.id))
        for creature in self.creatures:
            creature.model.creature = creature
            creature.model.environment = self
    
    def step(self, delta_time, screen=None):
        for creature in self.creatures:
            if creature.energy <= 0:
                print("Creature " + creature.id + " ran " + OUT_OF_ENERGY)
                return "Creature " + creature.id + " ran " + OUT_OF_ENERGY
        
        all_creature_pairs = [(a, b) for idx, a in enumerate(self.creatures) for b in self.creatures[idx + 1:]]  # Got this from GeeksForGeeks
        for a, b in all_creature_pairs:
            if (a.model.type != b.model.type) and (np.linalg.norm(a.position - b.position) <\
                                                   ((1 - self.EAT_EPSILON) * (a.size + b.size))):
                self.creatures.remove(a if a.mode.type == PREY else b)
        if len([c for c in filter(FILTER_OUT_PREDATOR_OBJECTS, self.creatures)]) < 1:
            print(ALL_PREY_EATEN)
            return ALL_PREY_EATEN
        
        all_inputs = [creature.model.get_inputs() for creature in self.creatures]
        
        ######################################################################
        # The following is for testing                                       #
        ######################################################################
        override = np.array([0.0, 0.0], dtype=self.DTYPE)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            override += np.array([1.0, 0.0], dtype=self.DTYPE)
        if keys[pygame.K_a]:
            override += np.array([0.0, -1.0], dtype=self.DTYPE)
        if keys[pygame.K_s]:
            override += np.array([-1.0, 0.0], dtype=self.DTYPE)
        if keys[pygame.K_d]:
            override += np.array([0.0, 1.0], dtype=self.DTYPE)
        if keys[pygame.K_SPACE]:
            self.creatures[0].velocity = np.array([0.0, 0.0], dtype=self.DTYPE)
        override = normalize(override)
        if np.linalg.norm(override) > 0 or ALWAYS_OVERRIDE_PREY_MOVEMENT:
            all_inputs[0][0] = override.tolist()
        override = 0.0
        if keys[pygame.K_LEFT]:
            override += 2 * np.pi * (1 / 2) / 1000
        if keys[pygame.K_RIGHT]:
            override += -2 * np.pi * (1 / 2) / 1000
        if keys[pygame.K_DOWN]:
            override = -all_inputs[0][1]
        if ALWAYS_OVERRIDE_PREY_MOVEMENT:
            all_inputs[0][1] = 0
        all_inputs[0][1] += override
        ######################################################################
        # END TESTING                                                        #
        ######################################################################
        
        ################################################################################################################
        # Creature positions are updated here                                                                          #
        ################################################################################################################
        for creature, inputs in zip(self.creatures, all_inputs):
            creature.update_rotation_speed(-inputs[1])  # Negated because the screen is flipped
            creature.rotate(delta_time)
            creature.apply_force(self.DRAG_COEFFICIENT * (np.linalg.norm(creature.velocity) ** 2) * self.DRAG_DIRECTION)
            creature.apply_force(np.array(inputs[0], dtype=self.DTYPE))
            creature.update_velocity(delta_time)
            creature.update_position(delta_time)
            if DRAW:
                creature.draw(screen)
        ################################################################################################################
        ################################################################################################################
        
        self.time += delta_time

        return SUCCESSFUL_STEP
    
    def get_state_info(self):
        """
        :return: { "creature_states" : [ see below ], "time" : float elapsed time }
        """
        state_info = {}
        creature_states = []
        for creature in self.creatures:
            creature_states.append({
                "type"      : creature.model.type,
                "position"  : creature.position,
                "direction" : creature.direction,
                "speed"     : np.linalg.norm(creature.velocity),
                "id"        : creature.id,
                "energy"    : creature.energy
            })
        state_info["creature_states"] = creature_states
        state_info["time"] = self.time
        # TODO : WHAT OTHER STUFF?
        return state_info


class Ray:
    def __init__(self, position, angle):
        self.position = position
        self.angle = angle
        
    def cast(self, screen, length):
        end_point = self.position + angle_to_vec(self.angle) * length  # Arbitrary length
        pygame.draw.line(screen, WHITE, self.position, end_point, 1)

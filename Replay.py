import argparse
import pygame
import Loader
from Globals import *

parser = argparse.ArgumentParser(description='Process the .pkl filename.')
parser.add_argument('--filename', type=str, help='Name of the .pkl file to load')
parser.add_argument('--tick_delay', type=int, default=0, help="Optional number of ticks to skip between position updates")
parser.add_argument('--update_skip', type=int, default=0, help="Optional number of position updates to skip each tick")
parser.add_argument('--start_time', type=int, default=0, help="Optional number of ticks to ignore before beginning display")
args = parser.parse_args()

experiments = Loader.LoadPlaintext(args.filename)  # CHANGE THIS METHOD CALL APPROPRIATELY

for experiment in experiments:
    screen_width = DEFAULT_ENVIRONMENT_PARAMETERS["screen_width"]
    screen_height = DEFAULT_ENVIRONMENT_PARAMETERS["screen_height"]
    try:
        # FORMATTING FOR OLD SERIALIZATION
        # get screen_width
        # get screen_height
        pass
    except:
        pass
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()
    preys = experiment["PREYS"]
    predators = experiment["PREDATORS"]
    running = True
    time = args.start_time
    skip_time = 0
    while(running):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        screen.fill(BACKGROUND_COLOR)
        clock.tick(DEFAULT_MIN_TPS)
        
        dead = 0
        for creature, color in zip(preys + predators, ([GREEN] * len(experiment["PREYS"])) + ([RED] * len(experiment["PREDATORS"]))):
            try:
                position = creature["POSITIONS"][time]
                print(position)
                try:
                    direction = creature["DIRECTIONS"][time]
                except:
                    direction = 0.0
                try:
                    size = creature.size
                except:
                    size = DEFAULT_CREATURE_SIZE       
                try:
                    alive = creature["ALIVES"][time]
                except:
                    alive = True
                pygame.draw.circle(screen, color if alive else GRAY, position, size)
                bulge_radius = DEFAULT_CREATURE_SIZE
                bulge_position = position + ANGLE_TO_VEC(direction) * bulge_radius
                bulge_size = 4
                pygame.draw.circle(screen, color if alive else GRAY, bulge_position.astype(int), bulge_size)
            except Exception as e:
                dead += 1
                print(e)
        if dead >= len(experiment["PREYS"]) + len(experiment["PREDATORS"]):
            running = False
        
        pygame.display.flip()
        
        if skip_time >= args.tick_delay:
            skip_time = 0
            time += 1 + args.update_skip
        else:
            skip_time += 1
    pygame.quit()
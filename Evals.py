"""
THIS FILE WRITTEN BY ADVAIT GOSAI
"""

from main import Model
import matplotlib.pyplot as plt
import pickle
import argparse
import numpy as np
from Globals import *

# parser = argparse.ArgumentParser(description='Process the .pkl filename.')
# parser.add_argument('filename', type=str, help='Name of the .pkl file to load')
# args = parser.parse_args()

# data = open(args.filename, "rb")
# experiments = pickle.load(data)

# end_reason_counts = {}
# end_reason_plots = {}

# unique_end_reasons = set()
# markers = ['o', 's', '^', 'x']

# for i, exp in enumerate(experiments):
#     print(f"EXPERIMENT {i+1}")
#     preds = exp[1]
#     preys = exp[-1]
#     print(f"num_predators: {len(preds)}, num_preys: {len(preys)}")
#     print(f"sim_time: {exp['sim_time']}")
#     print(f"end_reason: {exp['end_reason']}")

#     unique_end_reasons.add(exp['end_reason'])

#     if exp['end_reason'] in end_reason_counts:
#         end_reason_counts[exp['end_reason']] += 1
#     else:
#         end_reason_counts[exp['end_reason']] = 1
#     if exp['end_reason'] not in end_reason_plots:
#         end_reason_plots[exp['end_reason']] = []
#     end_reason_plots[exp['end_reason']].append((i+1, exp['sim_time']))

#     num_plots = 10 
#     fig, axs = plt.subplots(num_plots // 2, 2, figsize=(15, 2 * num_plots))
#     fig.suptitle(f"Predator Loss; {preds[0]['NETWORK'].name}; Experiment {i+1}")
#     for j in range(num_plots):
#         row, col = divmod(j, 2)
#         if j < len(preds):
#             axs[row, col].plot(preds[j]["LOSSES"])
#             axs[row, col].set_xlabel("Iteration")
#             axs[row, col].set_ylabel("Losses")
#     plt.tight_layout(pad=3.0)
#     plt.show()

#     fig, axs = plt.subplots(num_plots // 2, 2, figsize=(15, 2 * num_plots))
#     fig.suptitle(f"Prey Loss; {preys[0]['NETWORK'].name}; Experiment {i+1}")
#     for j in range(num_plots):
#         row, col = divmod(j, 2)
#         if j < len(preys):
#             axs[row, col].plot(preys[j]["LOSSES"])
#             axs[row, col].set_title(f"Prey {j+1}")
#             axs[row, col].set_xlabel("Iteration")
#             axs[row, col].set_ylabel("Losses")
#     plt.tight_layout(pad=3.0)
#     plt.show()
#     print("----------------------------")

# print(end_reason_counts)

# end_reason_marker = {reason: markers[i % len(markers)] for i, reason in enumerate(unique_end_reasons)}

# plt.figure(figsize=(10, 6))
# for reason, data in end_reason_plots.items():
#     x, y = zip(*data)
#     plt.scatter(x, y, marker=end_reason_marker[reason], label=reason)

# plt.title("End Reasons for Each Experiment")
# plt.xlabel("Experiment Number")
# plt.ylabel("Simulation Time")
# plt.legend()
# plt.show()

parser = argparse.ArgumentParser(description='Process the .pkl filename.')
parser.add_argument('filename', type=str, help='Name of the .pkl file to load')
args = parser.parse_args()

data = open(args.filename, "r")
experiment_strs = []
# Num prey, num predators, sim time, end reason, losses per creature, positions per creature, number of preys eaten per predator later
experiments = []
data_str = data.read().replace(' ', '')
experiment_indeces = []
end = 0
# CORRECT vvv
while(end > -1):
    start = data_str.find("{\'real_time", end + 1)
    end = max(data_str.find("{\'real_time", start + 1) - 1, -1)
    experiment_strs.append(data_str[start:end])

for exp_str in experiment_strs:
    last_index = 0
    index = -1
    experiment = {}
    index = exp_str.find("sim_time\':")
    comma_index = exp_str.find(",", index + len("sim_time\':"))
    experiment["sim_time"] = exp_str[index + len("sim_time\':") : comma_index]
    index = exp_str.find("end_reason\':\'")
    apos_index = exp_str.find("\'", index + len("end_reason\':\'"))
    experiment["end_reason"] = exp_str[index + len("end_reason\':\'") : apos_index]
    preys = []
    predators = exp_str.find("," + str(PREDATOR) + ":")
    last_index = 0
    index = -1
    while(index < predators):
        prey = {}
        index = exp_str.find("{", last_index + 1)
        loss_index = exp_str.find("LOSSES\':", index) + len("LOSSES\':")
        positions_index = exp_str.find("POSITIONS\':", loss_index)
        loss_end = positions_index - 2
        positions_end = exp_str.find("}", positions_index)
        prey["LOSSES"] = list(map(lambda x : float(x), exp_str[loss_index:loss_end][1:-1].split(',')))
        positions_index +=  len("POSITIONS\':")
        prey["POSITIONS"] = list(map(lambda x : np.array(list(map(lambda y : float(y), x[5:][1:-1][1:-1].split(',')))), exp_str[positions_index:positions_end][1:-1].split(',array')))
        preys.append(prey)
        last_index = positions_end
        
    index = predators
    predators = []
    end_of_exp = exp_str.find("}]}") + 3
    last_index = index
    while(last_index < len(exp_str) - 6):
        predator = {}
        index = exp_str.find("{", last_index + 1)
        loss_index = exp_str.find("LOSSES\':", index) + len("LOSSES\':")
        #print("loss index: " + str(loss_index))
        positions_index = exp_str.find("POSITIONS\':", loss_index)
        #print("positions index: " + str(positions_index))
        loss_end = positions_index - 2
        #print("loss end: " + str(loss_end))
        positions_end = exp_str.find("}", positions_index)
        #print("positions end: " + str(positions_end))
        predator["LOSSES"] = list(map(lambda x : float(x), exp_str[loss_index:loss_end][1:-1].split(',')))
        positions_index +=  len("POSITIONS\':")
        predator["POSITIONS"] = list(map(lambda x : np.array(list(map(lambda y : float(y), x[5:][1:-1][1:-1].split(',')))), exp_str[positions_index:positions_end][1:-1].split(',array')))
        predators.append(predator)
        last_index = positions_end
    experiment["PREYS"] = preys
    experiment["PREDATORS"] = predators
    experiment["num_preys"] = len(preys)
    experiment["num_predators"] = len(predators)
    experiments.append(experiment)
    
    

end_reason_counts = {}
end_reason_plots = {}

unique_end_reasons = set()
markers = ['o', 's', '^', 'x']

for i, exp in enumerate(experiments):
    if (i % PRINT_EVAL_STEPS == 0) or (i == len(experiments)):
        print(f"EXPERIMENT {i+1}")
        preds = exp["PREDATORS"]
        preys = exp["PREYS"]
        print(f"num_predators: {len(preds)}, num_preys: {len(preys)}")
        print(f"sim_time: {exp['sim_time']}")
        print(f"end_reason: {exp['end_reason']}")

        unique_end_reasons.add(exp['end_reason'])

        if exp['end_reason'] in end_reason_counts:
            end_reason_counts[exp['end_reason']] += 1
        else:
            end_reason_counts[exp['end_reason']] = 1
        if exp['end_reason'] not in end_reason_plots:
            end_reason_plots[exp['end_reason']] = []
        end_reason_plots[exp['end_reason']].append((i+1, exp['sim_time']))

        num_plots = 10 
        fig, axs = plt.subplots(num_plots // 2, 2, figsize=(15, 2 * num_plots))
        fig.suptitle(f"Predator Loss; Experiment {i+1}")  # ; {preds[0]['NETWORK'].name}
        for j in range(num_plots):
            row, col = divmod(j, 2)
            if j < len(preds):
                axs[row, col].plot(preds[j]["LOSSES"])
                axs[row, col].set_xlabel("Iteration")
                axs[row, col].set_ylabel("Losses")
        plt.tight_layout(pad=3.0)
        plt.show()

        fig, axs = plt.subplots(num_plots // 2, 2, figsize=(15, 2 * num_plots))
        fig.suptitle(f"Prey Loss; Experiment {i+1}")  # ; {preys[0]['NETWORK'].name}
        for j in range(num_plots):
            row, col = divmod(j, 2)
            if j < len(preys):
                axs[row, col].plot(preys[j]["LOSSES"])
                axs[row, col].set_title(f"Prey {j+1}")
                axs[row, col].set_xlabel("Iteration")
                axs[row, col].set_ylabel("Losses")
        plt.tight_layout(pad=3.0)
        plt.show()
        print("----------------------------")

print(end_reason_counts)

end_reason_marker = {reason: markers[i % len(markers)] for i, reason in enumerate(unique_end_reasons)}

plt.figure(figsize=(10, 6))
for reason, data in end_reason_plots.items():
    x, y = zip(*data)
    plt.scatter(x, y, marker=end_reason_marker[reason], label=reason)

plt.title("End Reasons for Each Experiment")
plt.xlabel("Experiment Number")
plt.ylabel("Simulation Time")
plt.legend()
plt.show()

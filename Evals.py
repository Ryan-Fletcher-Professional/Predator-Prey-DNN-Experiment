"""
THIS FILE WRITTEN BY ADVAIT GOSAI
"""

from main import Model
import matplotlib.pyplot as plt
import pickle

data = open("serialized_data_335418.pkl", "rb")
experiments = pickle.load(data)

print(f"num_experiments: {len(experiments)}")
print("----------------------------")

end_reason_counts = {}
end_reason_plots = {}

unique_end_reasons = set()
markers = ['o', 's', '^', 'x']

for i, exp in enumerate(experiments):
    print(f"EXPERIMENT {i+1}")
    preds = exp[1]
    preys = exp[-1]
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
    fig.suptitle(f"Predator Loss; {preds[0]['NETWORK'].name}; Experiment {i+1}")
    for j in range(num_plots):
        row, col = divmod(j, 2)
        if j < len(preds):
            axs[row, col].plot(preds[j]["LOSSES"])
            axs[row, col].set_xlabel("Iteration")
            axs[row, col].set_ylabel("Losses")
    plt.tight_layout(pad=3.0)
    plt.show()

    fig, axs = plt.subplots(num_plots // 2, 2, figsize=(15, 2 * num_plots))
    fig.suptitle(f"Prey Loss; {preys[0]['NETWORK'].name}; Experiment {i+1}")
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

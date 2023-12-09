import subprocess
import re
import yaml
import json

def find_best_run(commands, n):
    """
    returns: the index that corresponds to the best run
    """

    processes = []
    last_rewards = []

    for command in commands:
        # Use shell=True to run the command through the shell
        processes.append([])
        for i in range(n):
            command_extra = f"--seed {i+1}"
            process = subprocess.Popen(command + " " + command_extra, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            processes[-1].append(process)

    for process in processes:
        rewards = []
        
        # iterate over the n processes for each command
        for p in process:
            p.wait()
            output, _ = p.communicate()
            output = output.decode('utf-8')

            # Use regular expressions to extract the value of last_reward
            match = re.search(r'last_reward = ([\d.]+)', output)
            if match:
                last_reward_value = float(match.group(1))
                rewards.append(last_reward_value)
            else:
                print("Couldn't extract last_reward value from the output.")
                print(output)
        
        # average the rewards for each command 
        if len(rewards) == 0:
            last_rewards.append(float("-inf"))
        else:
            avg_reward = sum(rewards) / len(rewards)
            last_rewards.append(avg_reward)

    # Determine the index of the best run based on the highest last_reward value
    best_run_index = last_rewards.index(max(last_rewards))
    return best_run_index

def find_best_hyperparameters_for_env(env, exploration, all_hyperparameters, num_iterations):
    """
    returns: best hyperparameters for each environment, schedule pair.
    """
    hist = []
    best_hyperparameters = all_hyperparameters.copy()
    for param in best_hyperparameters:
        best_hyperparameters[param] = all_hyperparameters[param][0]

    for iter in range(num_iterations):
        print(f"iteration {iter}")
        print(best_hyperparameters)
        for param, values in all_hyperparameters.items():
            
            if len(all_hyperparameters[param]) < 2:
                continue
            
            print(f"param {param}")
            schedule_hyperparameters = all_hyperparameters.copy()

            # set the rest to the current best hyperparameters
            for p in schedule_hyperparameters:
                if param != p:
                    schedule_hyperparameters[p] = best_hyperparameters[p]

            commands = [
                f"python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/{env}_{i}.yaml -esf experiments/exploration/{exploration}_{i}.yaml"
                for i in range(len(values))
            ]
            
            # try all possibilities for the current hyperparameter in question
            for i, v in enumerate(values):
                schedule_hyperparameters[param] = v

                # TODO: save parameters to ith file. need to split into two files.
                env_dict = {
                    "lr": schedule_hyperparameters["lr"],
                    "batch_size": schedule_hyperparameters["batch_size"],
                    "base_config": schedule_hyperparameters["base_config"],
                    "env_name": schedule_hyperparameters["env_name"],
                    "target_update_period": schedule_hyperparameters["target_update_period"],
                    "total_steps": schedule_hyperparameters["total_steps"],
                }

                exploration_dict = dict()
                for key in all_hyperparameters:
                    if key not in env_dict:
                        exploration_dict[key] = schedule_hyperparameters[key]
                with open(f"experiments/exploration/{exploration}_{i}.yaml", 'w') as file:
                    yaml.dump(exploration_dict, file)

                env_dict['exp_name'] = "_".join([key + "_" + str(value) for key, value in env_dict.items()])
                env_dict['exp_name'] += "_" + "_".join([key + "_" + str(value) for key, value in exploration_dict.items()])
                with open(f"experiments/dqn/{env}_{i}.yaml", 'w') as file:
                    yaml.dump(env_dict, file)
            
            best_index = find_best_run(commands, n=3)
            # TODO: update this hyperparameter
            best_hyperparameters[param] = values[best_index]
        
        hist.append(best_hyperparameters.copy())

    json.dump(hist, open(f"results", 'w'))
    
    return best_hyperparameters

all_hyperparameters = {
    "alpha": [0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 0.99],
    "n": [100, 500, 1000, 2000, 5000, 10000],
    "p": [0.01, 0.05, 0.1, 0.25, 0.5],
    "threshold": [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8],
    "eps_max": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "schedule_type": ["adaptive"],
    
    "lr": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
    "batch_size": [10, 50, 100, 200],
    "base_config": ["dqn_basic"],
    "env_name": ["CartPole-v1"],
    "target_update_period": [1000],
    "total_steps": [20000],
}

best_hyperparameters = find_best_hyperparameters_for_env("cartpole", "adaptive", all_hyperparameters, 5)
print("best_hyperparameters")
print(best_hyperparameters)
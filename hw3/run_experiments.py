import subprocess

def find_best_run(commands):
    """
    commands: len(commands) commands to run that sweep one hyperparameter

    returns: the index that corresponds to the best run
    """

    for command in commands:
        # Use shell=True to run the command through the shell
        process = subprocess.Popen(command, shell=True)
        processes.append(process)
    
    for process in processes:
        process.wait()

    # TODO: return the best run
    return

def find_best_hyperparameters_for_env(base_command, hyperparameters, num_iterations):
    """
    hyperparameters
    [
        ["-lr 0.9", "-lr 0.9", ...],
        ["-d 0.89", "-d 0.9", ...],   
    ]

    returns: best hyperparameters for each environment, schedule pair."""

    best_indices = [
        0,
        0,
        0,
        0,
        0
    ]
    for _ in range(num_iterations):
        for i, parameter in enumerate(hyperparameters):
            # TODO: sweep one hyperparameter
            # TODO: vary the ith hyperparameter. use the best indices for the rest
            best_index = find_best_run(commands)
            # TODO: update this hyperparameter
            pass

    return

commands = [f'sleep 0.5 && echo Process finished && touch test/{i}' for i in range(100)]

hyperparameters = {
    "alpha": [],
    "N": [],
    "P": [],
    "Threshold": [],
    "Impulse Strength": [],
}

best_hyperparameters = {
    "alpha": 0,
    "N": 0,
    "P": 0,
    "Threshold": 0,
    "Impulse Strength": 0,
}

step = 5
for i in range(0, len(commands), step):
    processes = []
    for j in range(step):
        print(f"Spawning processes {i + j}")
        command = commands[i + j]
        # Use shell=True to run the command through the shell
        process = subprocess.Popen(command, shell=True)
        processes.append(process)
    
    for process in processes:
        process.wait()

    # TODO: read from all log files. find the best hyperparameter
    print(f"Process for step {i} finished.")
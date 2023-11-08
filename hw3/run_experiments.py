import subprocess

commands = [f'sleep 0.5 && echo Process finished && touch test/{i}' for i in range(100)]

step = 5
for i in range(0, len(commands), step):
    processes = []
    for j in range(step):
        print(f"Process {i + j}")
        command = commands[i + j]
        # Use shell=True to run the command through the shell
        process = subprocess.Popen(command, shell=True)
        processes.append(process)
    
    for process in processes:
        process.wait()
    print(f"Process for step {i} finished.")
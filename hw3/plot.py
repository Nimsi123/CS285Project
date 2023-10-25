import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

log_files_q2 = {
    "cartpole": "/Users/danieletaat/Desktop/fa23/cs285/homework_fall2023/hw3/data/hw3_dqn_dqn_CartPole-v1_s64_l2_d0.99_20-10-2023_18-55-41",
    "cartpole lr 0.05": "/Users/danieletaat/Desktop/fa23/cs285/homework_fall2023/hw3/data/hw3_dqn_dqn_CartPole-v1_s64_l2_d0.99_20-10-2023_18-59-18"
}

log_files_q2_5a = {
    "lunar 1 vanilla": "/Users/danieletaat/Desktop/fa23/cs285/homework_fall2023/hw3/data 2.4/hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_15-10-2023_12-05-26",
    "lunar 2 vanilla": "/Users/danieletaat/Desktop/fa23/cs285/homework_fall2023/hw3/data 2.4/hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_15-10-2023_12-11-22",
    "lunar 3 vanilla": "/Users/danieletaat/Desktop/fa23/cs285/homework_fall2023/hw3/data 2.4/hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_15-10-2023_12-11-23",
}
log_files_q2_5b = {
    "lunar 1 double": "/Users/danieletaat/Desktop/fa23/cs285/homework_fall2023/hw3/data 2.5a/hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_doubleq_15-10-2023_12-29-13",
    "lunar 2 double": "/Users/danieletaat/Desktop/fa23/cs285/homework_fall2023/hw3/data 2.5a/hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_doubleq_15-10-2023_12-29-20",
    "lunar 3 double": "/Users/danieletaat/Desktop/fa23/cs285/homework_fall2023/hw3/data 2.5a/hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_doubleq_15-10-2023_12-29-26"
}
log_files_q2_5c = {
    "mspacman": "/Users/danieletaat/Desktop/fa23/cs285/homework_fall2023/hw3/data 2.5b/data/hw3_dqn_dqn_MsPacmanNoFrameskip-v0_d0.99_tu2000_lr0.0001_doubleq_clip10.0_15-10-2023_20-44-20"
}
log_files_q2_6 = {
    "lunar": "/Users/danieletaat/Desktop/fa23/cs285/homework_fall2023/hw3/data 2.6/hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_15-10-2023_12-05-26",
    "lunar lr 1e-1": "/Users/danieletaat/Desktop/fa23/cs285/homework_fall2023/hw3/data 2.6/hw3_dqn_lunar_lander_exp_1e-1_LunarLander-v2_s64_l2_d0.99_15-10-2023_18-47-15",
    "lunar lr 1e-2": "/Users/danieletaat/Desktop/fa23/cs285/homework_fall2023/hw3/data 2.6/hw3_dqn_lunar_lander_exp_1e-2_LunarLander-v2_s64_l2_d0.99_15-10-2023_18-47-42",
    "lunar lr 1e-4": "/Users/danieletaat/Desktop/fa23/cs285/homework_fall2023/hw3/data 2.6/hw3_dqn_lunar_lander_exp_1e-4_LunarLander-v2_s64_l2_d0.99_15-10-2023_18-47-57"
}

log_files_hopper = {
    "hopper clip": "/Users/danieletaat/Desktop/fa23/cs285/homework_fall2023/hw3/data 3/hw3_sac_sac_hopper_clipq_Hopper-v4_reparametrize_s128_l3_alr0.0003_clr0.0003_b256_d0.99_t0.05_stu0.005_min_16-10-2023_02-50-29",
    "hopper double": "/Users/danieletaat/Desktop/fa23/cs285/homework_fall2023/hw3/data 3/hw3_sac_sac_hopper_doubleq_Hopper-v4_reparametrize_s128_l3_alr0.0003_clr0.0003_b256_d0.99_t0.05_stu0.005_doubleq_16-10-2023_02-50-29",
    "hopper single": "/Users/danieletaat/Desktop/fa23/cs285/homework_fall2023/hw3/data 3/hw3_sac_sac_hopper_singlecritic_Hopper-v4_reparametrize_s128_l3_alr0.0003_clr0.0003_b256_d0.99_t0.05_stu0.005_16-10-2023_02-50-30"
}

log_files_humanoid = {
    "humanoid": "/Users/danieletaat/Desktop/fa23/cs285/homework_fall2023/hw3/data 3/hw3_sac_sac_humanoid_Humanoid-v4_reparametrize_s256_l3_alr0.0003_clr0.0003_b256_d0.99_t0.05_stu0.005_min_16-10-2023_02-50-29"
}

log_files_cheetah1 = {
    "cheetah reinforce1": "/Users/danieletaat/Desktop/fa23/cs285/homework_fall2023/hw3/data 3/hw3_sac_reinforce1_HalfCheetah-v4_reinforce_s128_l3_alr0.0003_clr0.0003_b128_d0.99_t0.2_stu0.005_15-10-2023_19-22-50",
    "cheetah reinforce10": "/Users/danieletaat/Desktop/fa23/cs285/homework_fall2023/hw3/data/hw3_sac_reinforce10_HalfCheetah-v4_reinforce_s128_l3_alr0.0003_clr0.0003_b128_d0.99_t0.2_stu0.005_20-10-2023_09-36-34"
}

log_files_cheetah2 = {
    "cheetah reinforce1": "/Users/danieletaat/Desktop/fa23/cs285/homework_fall2023/hw3/data 3/hw3_sac_reinforce1_HalfCheetah-v4_reinforce_s128_l3_alr0.0003_clr0.0003_b128_d0.99_t0.2_stu0.005_15-10-2023_19-22-50",
    "cheetah reinforce10": "/Users/danieletaat/Desktop/fa23/cs285/homework_fall2023/hw3/data/hw3_sac_reinforce10_HalfCheetah-v4_reinforce_s128_l3_alr0.0003_clr0.0003_b128_d0.99_t0.2_stu0.005_20-10-2023_09-36-34",
    "cheetah reparam": "/Users/danieletaat/Desktop/fa23/cs285/homework_fall2023/hw3/data 3/hw3_sac_reparametrize_HalfCheetah-v4_reparametrize_s128_l3_alr0.0003_clr0.0003_b128_d0.99_t0.1_stu0.005_16-10-2023_02-50-30"
}

def load_data(log_file, value='Eval_AverageReturn', data='Train_EnvstepsSoFar'):
    event_acc = EventAccumulator(log_file)
    event_acc.Reload()
    values = event_acc.Scalars(value)
    if data:
        env_steps = [
            e.value for e in event_acc.Scalars(data)
        ]
    else:
        env_steps = [
            e.step for e in values
        ]
    values = [
        e.value for e in event_acc.Scalars(value)
    ]

    return env_steps, values

def smooth_data(data, n):
    smoothed_data = []
    for i in range(len(data)):
        ds = data[max(i-n, 0): i+1]
        smoothed_data.append(sum(ds) / len(ds))
    return smoothed_data

def q2a():
    for label in log_files_q2_5a:
        log_file = log_files_q2_5a[label]
        x, y = load_data(log_file, value='eval_return', data=None)
        y = smooth_data(y, 5)
        plt.plot(x, y, "b", label=label)
        
    for label in log_files_q2_5b:
        log_file = log_files_q2_5b[label]
        x, y = load_data(log_file, value='eval_return', data=None)
        y = smooth_data(y, 5)
        plt.plot(x, y, "r", label=label)
        
    plt.title("Learning Curves (Q2.5)")
    plt.ylabel("Average Return")
    plt.legend()
    plt.show()

def q2b():
    for label in log_files_q2_5c:
        log_file = log_files_q2_5c[label]
        x1, y1 = load_data(log_file, value='eval_return', data=None)
        x2, y2 = load_data(log_file, value='train_return', data=None)
        y1 = smooth_data(y1, 5)
        y2 = smooth_data(y2, 15)
        plt.plot(x1, y1, label=label+" eval return")
        plt.plot(x2, y2, label=label+" train return")
        
    plt.title("Returns (Q2.5)")
    plt.ylabel("Average Return")
    plt.legend()
    plt.show()
      
def q2c():
    for label in log_files_q2_6:
        log_file = log_files_q2_6[label]
        x1, y1 = load_data(log_file, value='eval_return', data=None)
        y1 = smooth_data(y1, 5)
        plt.plot(x1, y1, label=label)
        
    plt.title("Varying Learning Rate (Q2.6)")
    plt.ylabel("Average Return")
    plt.legend()
    plt.show()
    
def q3_hopper():
    for label in log_files_hopper:
        log_file = log_files_hopper[label]
        x1, y1 = load_data(log_file, value='eval_return', data=None)
        y1 = smooth_data(y1, 5)
        plt.plot(x1, y1, label=label)
    plt.title("Hoppers (Q3.1.5)")
    plt.ylabel("Average Return")
    plt.legend()
    plt.show()
    
    for label in log_files_hopper:
        log_file = log_files_hopper[label]
        x1, y1 = load_data(log_file, value='q_values', data=None)
        y1 = smooth_data(y1, 5)
        plt.plot(x1, y1, label=label)
    plt.title("Hoppers (Q3.1.5)")
    plt.ylabel("Q Values")
    plt.legend()
    plt.show()
    
    
    
def q3_humanoid():
    for label in log_files_humanoid:
        log_file = log_files_humanoid[label]
        x1, y1 = load_data(log_file, value='eval_return', data=None)
        y1 = smooth_data(y1, 5)
        plt.plot(x1, y1, label=label)
    plt.title("Humanoid (Q3.1.5)")
    plt.ylabel("Average Return")
    plt.legend()
    plt.show()
    
def q3_cheetah1():
    for label in log_files_cheetah1:
        log_file = log_files_cheetah1[label]
        x1, y1 = load_data(log_file, value='eval_return', data=None)
        y1 = smooth_data(y1, 5)
        plt.plot(x1, y1, label=label)
    plt.title("Cheetahs (Q3.1.3)")
    plt.ylabel("Average Return")
    plt.legend()
    plt.show()
    
def q3_cheetah2():
    for label in log_files_cheetah2:
        log_file = log_files_cheetah2[label]
        x1, y1 = load_data(log_file, value='eval_return', data=None)
        y1 = smooth_data(y1, 5)
        plt.plot(x1, y1, label=label)
    plt.title("Cheetahs (Q3.1.4)")
    plt.ylabel("Average Return")
    plt.legend()
    plt.show()
    
def q2_cartpole():
    for label in log_files_q2:
        log_file = log_files_q2[label]
        x1, y1 = load_data(log_file, value='eval_return', data=None)
        y1 = smooth_data(y1, 5)
        plt.plot(x1, y1, label=label)
    plt.title("Cartpole (Q2.4)")
    plt.ylabel("Average Return")
    plt.legend()
    plt.show()
    
    for label in log_files_q2:
        log_file = log_files_q2[label]
        x1, y1 = load_data(log_file, value='q_values', data=None)
        y1 = smooth_data(y1, 5)
        plt.plot(x1, y1, label=label)
    plt.title("Cartpole Q Values (Q2.4)")
    plt.ylabel("Q Value")
    plt.legend()
    plt.show()
    
    for label in log_files_q2:
        log_file = log_files_q2[label]
        x1, y1 = load_data(log_file, value='critic_loss', data=None)
        # y1 = smooth_data(y1, 5)
        plt.plot(x1, y1, label=label)
    plt.title("Cartpole (Q2.4)")
    plt.ylabel("Critic Error")
    plt.legend()
    plt.show()  

# q2a()
q2b()
# q2c()
# q3_hopper()
# q3_humanoid()
# q3_cheetah1()
# q3_cheetah2()
# q2_cartpole()

"""
'eval_return', 
'eval_ep_len', 
'eval/return_std', 
'eval/return_max', 
'eval/return_min', 
'eval/ep_len_std', 
'eval/ep_len_max', 
'eval/ep_len_min', 
'train_return', 
'train_ep_len', 
'critic_loss', 
'q_values', 
'target_values', 
'grad_norm', 
'epsilon', 
'lr'
"""
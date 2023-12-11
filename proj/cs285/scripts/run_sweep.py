import argparse

from cs285.agents.dqn_agent import DQNAgent

import gym
import numpy as np
import torch
from cs285.infrastructure import pytorch_util as ptu
import tqdm

from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger
from cs285.infrastructure.replay_buffer import MemoryEfficientReplayBuffer, ReplayBuffer

from scripting_utils import make_logger, make_config

import cs285.env_configs.schedule as schedule

import yaml
import json
import time
import os

MAX_NVIDEO = 2


def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    env = config["make_env"]()
    eval_env = config["make_env"]()
    exploration_schedule = config["exploration_schedule"]
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    assert discrete, "DQN only supports discrete action spaces"

    agent = DQNAgent(
        env.observation_space.shape,
        env.action_space.n,
        **config["agent_kwargs"],
    )

    ep_len = env.spec.max_episode_steps

    observation = None

    # Replay buffer
    if len(env.observation_space.shape) == 3:
        frame_history_len = env.observation_space.shape[0]
        assert frame_history_len == 4, "only support 4 stacked frames"
        replay_buffer = MemoryEfficientReplayBuffer(
            frame_history_len=frame_history_len
        )
    elif len(env.observation_space.shape) == 1:
        replay_buffer = ReplayBuffer()
    else:
        raise ValueError(
            f"Unsupported observation space shape: {env.observation_space.shape}"
        )

    def reset_env_training():
        nonlocal observation

        observation = env.reset()

        assert not isinstance(
            observation, tuple
        ), "env.reset() must return np.ndarray - make sure your Gym version uses the old step API"
        observation = np.asarray(observation)

        if isinstance(replay_buffer, MemoryEfficientReplayBuffer):
            replay_buffer.on_reset(observation=observation[-1, ...])

    reset_env_training()
    # need to make sure this stuff is getting properly reset between the ends of each episodes
    episode_timestep, episode_return, rollout_number = 0, 0, 0
    rewards = []
    for step in tqdm.trange(config["total_steps"], dynamic_ncols=True):
        epsilon = exploration_schedule.value(step, rewards)
        
        # TODO(student): Compute action
        action = agent.get_action(observation, epsilon)

        # TODO(student): Step the environment
        next_observation, reward, done, info = env.step(action)
        next_observation = np.asarray(next_observation)
        truncated = info.get("TimeLimit.truncated", False)
        
        # store rewards
        episode_return = episode_return + reward
        rewards.append(episode_return)
        if len(rewards) > 50000:
            rewards = rewards[-50000:]

        # TODO(student): Add the data to the replay buffer
        if isinstance(replay_buffer, MemoryEfficientReplayBuffer):
            # We're using the memory-efficient replay buffer,
            # so we only insert next_observation (not observation)
            replay_buffer.insert(action, reward, next_observation[-1], done)
        else:
            # We're using the regular replay buffer
            replay_buffer.insert(observation, action, reward, next_observation, done)

        # Handle episode termination
        if done or truncated:
            episode_timestep = 0
            episode_return = 0
            rollout_number += 1
            
            reset_env_training()
            logger.log_scalar(info["episode"]["r"], "train_return", step)
            logger.log_scalar(info["episode"]["l"], "train_ep_len", step)
        else:
            observation = next_observation
            episode_timestep += 1   

        # Main DQN training loop
        if step >= config["learning_starts"]:
            # TODO(student): Sample config["batch_size"] samples from the replay buffer
            batch = replay_buffer.sample(config["batch_size"])

            # Convert to PyTorch tensors
            batch = ptu.from_numpy(batch)

            # TODO(student): Train the agent. `batch` is a dictionary of numpy arrays,
            update_info = agent.update(batch["observations"], batch["actions"], batch["rewards"], batch["next_observations"], batch["dones"], step)
            
            # Logging code
            update_info["epsilon"] = epsilon
            update_info["lr"] = agent.lr_scheduler.get_last_lr()[0]

            if step % args.log_interval == 0:
                for k, v in update_info.items():
                    logger.log_scalar(v, k, step)
                logger.flush()

        if step % args.eval_interval == 0:
            # Evaluate
            trajectories = utils.sample_n_trajectories(
                eval_env,
                agent,
                args.num_eval_trajectories,
                ep_len,
            )
            returns = [t["episode_statistics"]["r"] for t in trajectories]
            ep_lens = [t["episode_statistics"]["l"] for t in trajectories]

            logger.log_scalar(np.mean(returns), "eval_return", step)
            logger.log_scalar(np.mean(ep_lens), "eval_ep_len", step)

            if len(returns) > 1:
                logger.log_scalar(np.std(returns), "eval/return_std", step)
                logger.log_scalar(np.max(returns), "eval/return_max", step)
                logger.log_scalar(np.min(returns), "eval/return_min", step)
                logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", step)
                logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", step)
                logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", step)

    trajectories = utils.sample_n_trajectories(
                    eval_env,
                    agent,
                    args.num_eval_trajectories,
                    ep_len,
                )
    returns = [ t["episode_statistics"]["r"] for t in trajectories ]
    return np.mean(returns)

def run_experiment(sweep_name, config_fname, schedule_fname, eval_interval, num_seeds):    
    returns = []
    for seed in range(1, num_seeds+1):
        args = argparse.Namespace(
            config_file=config_fname,
            eval_interval=eval_interval, 
            num_eval_trajectories=10, 
            num_render_trajectories=0, 
            seed=np.random.randint(0, 1000000), 
            no_gpu=False, 
            which_gpu=0, 
            log_interval=1000, 
            exploration_schedule_file=schedule_fname
        )
        logdir_prefix = sweep_name + "_" # keep for autograder
        config = make_config(args.config_file)
        config["log_name"] += f"_seed_{seed}"    
        config["exploration_schedule"] = schedule.get_schedule(args.exploration_schedule_file)
        logger = make_logger(logdir_prefix, config)
        
        r = run_training_loop(config, logger, args)
        returns.append(r)
        logger.close()
                
    return np.mean(returns)

def run_sweep(sweep_name, params, n):
    """Executes `n` sweeps over the hyperparameters in `params`."""
    enviroment = params["env_name"]
    schedule = params["schedule_type"]
    
    # initialize configs 
    best_params = {k: v[0] for k, v in params.items()}
    environment_config = {
        "lr": best_params["lr"],
        "batch_size": best_params["batch_size"],
        "base_config": best_params["base_config"],
        "env_name": best_params["env_name"],
        "target_update_period": best_params["target_update_period"],
        "total_steps": best_params["total_steps"],
    }
    exploration_config = dict()
    for param_name in params:
        if param_name not in environment_config:
            exploration_config[param_name] = best_params[param_name]
    
    enviroment_config_fname = f"experiments/dqn/{enviroment}.yaml"
    exploration_config_fname = f"experiments/exploration/{schedule}.yaml"
    
    for i in range(n):
        print(f"Running sweep {i+1}/{n}")
        
        for param_name in params:

            # skip parameters that only have one value
            if len(params[param_name]) == 1:
                continue
            
            returns = []
            for param in params[param_name]:
                print(f"Sweeping over {param_name} = {param}")
                best_params[param_name] = param
                
                # update the configs files
                for k in environment_config:
                    environment_config[k] = best_params[k]
                for k in exploration_config:
                    exploration_config[k] = best_params[k]
                environment_config["exp_name"] = "_".join([k + "_" + str(v) for k, v in environment_config.items()]) + "_" + "_".join([k + "_" + str(v) for k, v in exploration_config.items()])
                
                # save the configs files
                with open(enviroment_config_fname, "w") as file:
                    yaml.dump(environment_config, file)
                with open(exploration_config_fname, "w") as file:
                    yaml.dump(exploration_config, file)
                del environment_config["exp_name"]
                
                # run the experiment
                r = run_experiment(
                    sweep_name,
                    enviroment_config_fname, 
                    exploration_config_fname, 
                    environment_config["total_steps"] // 10, 
                    num_seeds=3
                )
                returns.append(r)
            
            # update the best parameter based on the returns
            best_params[param_name] = params[param_name][np.argmax(returns)]
    
    # teardown
    with open(f"results/{sweep_name}.json", "w") as f:
        json.dump(best_params, f)
    os.remove(enviroment_config_fname)
    os.remove(exploration_config_fname)
    
    return best_params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", "-sn", type=str, required=True)
    parser.add_argument("--params_file", "-pf", type=str, required=True)
    parser.add_argument("--num_sweeps", "-ns", type=int, default=1)
    args = parser.parse_args()
    
    with open(args.params_file, "r") as f:
        params = json.load(f)
    best_params = run_sweep(args.name, params, args.num_sweeps)
    print(best_params)
    
if __name__ == "__main__":
    main()
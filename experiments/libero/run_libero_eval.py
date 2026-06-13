
import os
import sys
import tqdm
import time

import torch
import random
import wandb
import draccus

import numpy as np
from typing import Optional
from collections import deque
from dataclasses import dataclass
from wla_utils import WLA0

sys.path.append(os.path.join(os.getcwd(), "LIBERO"))
from libero.libero import benchmark


from libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_save_image,
    quat2axisangle,
    save_rollout_video,
)


@dataclass
class GenerateConfig:
    model_family: str = "wla"
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task

    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "experiments/libero_eval_logs"  
    norm_file_path: str = "configs/norm_stats.json"
    unnorm_key: str = "libero_spatial"
    resize_size: tuple = (256, 256)
    history_obs_step: int = 8
    chunk_size: int = 32

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)
    model_id: str = None
    checkpoints_dir: str = None
    save_video: bool = False


def set_seed_everywhere(seed: int):
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:

    set_seed_everywhere(cfg.seed)

    # Load model
    wla0 = WLA0(
        cfg.model_id,
        cfg.checkpoints_dir,
        cfg.norm_file_path,
        cfg.unnorm_key
    )

    # Initialize local logging
    DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")
    print(f"Checkpoints dir: {cfg.checkpoints_dir}")
    log_file.write(f"Checkpoints dir: {cfg.checkpoints_dir}\n")

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=512)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            
            t = 0
            history_obs = None
            action_queue = []
            action_counter = 0

            obs_deque = deque(maxlen=cfg.history_obs_step)

            while t < max_steps + cfg.num_steps_wait:
                if t < cfg.num_steps_wait:
                    obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                    t += 1
                    continue
                
                if cfg.save_video:
                    save_img = get_save_image(obs)
                    replay_images.append(save_img)
                
                if not action_queue:
                    history_obs = obs_deque[0] if obs_deque else None
                        
                    img = get_libero_image(history_obs, obs, cfg.resize_size)    
                    state = np.concatenate(
                        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                    )

                    observation = {
                        "full_image": img,
                        "state": state,
                    }

                    actions = wla0.inference(
                        observation, 
                        task_description,
                    )
                    actions = actions.clone()
                    actions[..., -1] = torch.where(actions[..., -1] >= 0.5, -1.0, 1.0)
                    # actions = actions[:4]
                    
                    action_queue = actions.tolist()
                    action_counter = 0

                current_action = action_queue.pop(0)
                obs_deque.append(obs)

                obs, reward, done, info = env.step(current_action)
                
                action_counter += 1
                t += 1

                if done:
                    task_successes += 1
                    total_successes += 1
                    break
    
            task_episodes += 1
            total_episodes += 1


            if cfg.save_video:
                save_rollout_video(
                    replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
                )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()

        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()

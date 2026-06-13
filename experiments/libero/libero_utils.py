import os
import math
import time
import imageio

import numpy as np
from torchvision.transforms import v2
from torchvision.transforms.functional import to_tensor

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

import torch
import torch.nn.functional as F
from torchvision.transforms import functional


DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


def resize_with_pad(img, size):
    h, w = functional.get_image_size(img)[::-1]
    s = size / max(h, w)
    nh, nw = int(h * s), int(w * s)

    img = v2.Resize((nh, nw))(img)
    return functional.pad(
        img,
        [(size - nw) // 2, (size - nh) // 2,
         (size - nw + 1) // 2, (size - nh + 1) // 2],
        fill=0
    )


def _make_transform(size):
    return v2.Compose([lambda img: resize_with_pad(img, size)])


def get_libero_env(task, model_family, resolution=512):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_image(history_obs, obs, resize_size):

    primary_image_transform = _make_transform(resize_size[0])
    auxiliary_image_transform = _make_transform(resize_size[1])

    img = obs["agentview_image"]
    img = img[::-1, ::-1]
    img_tensor = to_tensor(np.ascontiguousarray(img))
    img_tensor = primary_image_transform(img_tensor)

    wrist_img = obs['robot0_eye_in_hand_image']
    wrist_img = wrist_img[::-1, ::-1]
    wrist_img_tensor = to_tensor(np.ascontiguousarray(wrist_img))
    wrist_img_tensor = auxiliary_image_transform(wrist_img_tensor)
    
    if history_obs is not None:
        history_img = history_obs["agentview_image"]
        history_img = history_img[::-1, ::-1]
        history_img_tensor = to_tensor(np.ascontiguousarray(history_img))
        history_img_tensor = auxiliary_image_transform(history_img_tensor)
    else:
        history_img_tensor = img_tensor

    return [[history_img_tensor, img_tensor, wrist_img_tensor]]


def get_save_image(obs):
    img = obs["agentview_image"]
    img = img[::-1, ::-1]
    return img


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./experiments/rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

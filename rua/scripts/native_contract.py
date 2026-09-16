"""Pure, testable parts of WLA's native LIBERO evaluation contract."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def load_pilot(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    if cfg["suite"] != "libero_spatial" or cfg["task_ids"] != [0, 1, 2]:
        raise ValueError("The approved pilot is spatial tasks 0, 1, 2.")
    if cfg["debug_init_state_id"] in cfg["eval_init_state_ids"]:
        raise ValueError("Debug and evaluation initial states must be disjoint.")
    if cfg["eval_init_state_ids"] != [1, 2, 3, 4, 5]:
        raise ValueError("Do not replace pilot initial states based on results.")
    if (cfg["max_env_steps"], cfg["settle_steps"]) != (220, 10):
        raise ValueError("Keep the original spatial control budget and settling.")
    return cfg


def env_actions(actions) -> np.ndarray:
    """Exactly WLA's native gripper conversion; never clip or invent motion."""
    if hasattr(actions, "detach"):
        actions = actions.detach().cpu().numpy()
    result = np.array(actions, copy=True)
    if result.ndim != 2 or result.shape[1] != 7 or result.shape[0] == 0:
        raise ValueError(f"Expected a nonempty [chunk, 7] action array: {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("WLA returned non-finite actions; execute nothing.")
    result[..., -1] = np.where(result[..., -1] >= 0.5, -1.0, 1.0)
    return result


@dataclass
class Budget:
    max_steps: int = 220
    timeout_seconds: float = 900
    steps: int = 0
    started: float = 0

    def start(self) -> None:
        self.started = time.monotonic()

    def check(self) -> None:
        if self.steps >= self.max_steps:
            raise TimeoutError("environment_step_budget")
        if time.monotonic() - self.started >= self.timeout_seconds:
            raise TimeoutError("wall_clock_budget")

    def record_step(self) -> None:
        self.steps += 1

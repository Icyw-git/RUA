from __future__ import annotations

import time
from dataclasses import dataclass, field


class BudgetExhausted(Exception):
    pass


@dataclass
class Budget:
    max_steps: int = 220
    max_requests: int = 100
    timeout: float = 900
    steps: int = 0
    requests: int = 0
    started: float = field(default_factory=time.monotonic)

    def check_time(self):
        if time.monotonic() - self.started >= self.timeout:
            raise BudgetExhausted("wall_clock_budget")

    def before_step(self):
        self.check_time()
        if self.steps >= self.max_steps:
            raise BudgetExhausted("environment_step_budget")

    def before_request(self):
        self.check_time()
        if self.requests >= self.max_requests:
            raise BudgetExhausted("model_request_budget")
        self.requests += 1

    def describe(self):
        return (
            f"EPISODE BUDGET: {self.max_steps - self.steps} environment control steps "
            f"and {self.max_requests - self.requests} model requests remain; "
            f"{max(0, int(self.timeout - (time.monotonic() - self.started)))} seconds remain. "
            "Every repeated simulation action consumes a control step. DONE only claims completion; "
            "the simulator independently scores the task."
        )

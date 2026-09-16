"""Small action/context extension for the original Show-Harness controller role."""
from dataclasses import asdict
import json


class WLAActionPlugin:
    def __init__(self, executor, coordination=None):
        self.executor = executor
        self.coordination = coordination

    def action_tokens(self):
        return ("WLA_CHUNK",)

    def render_prompt(self):
        previous = self.executor.last_result
        result = "not yet called" if previous is None else json.dumps(asdict(previous), sort_keys=True)
        return (
            self.coordination.render_prompt() if self.coordination is not None else ""
        ) + (
            "WLA TOOL:\n"
            "You may choose WLA_CHUNK instead of an atomic action. It runs ONE native "
            "prediction chunk (at most 8 actual control steps), using the ORIGINAL COMPLETE "
            "task instruction, not your current stage or a rewritten instruction. "
            "It may progress beyond the current stage. After return, inspect the NEW "
            "images and proprioception. chunk_completed does NOT mean stage success. "
            "Do not infer holding from the last gripper command alone. "
            "Choose freely using the observations: WLA is available for learned manipulation; "
            "atomic actions are available for visually justified local corrections, lift, "
            "grasp or release. There is no fixed alternation schedule or required WLA quota. "
            "If progress is poor, reassess rather than blindly repeating either choice. "
            "The last WLA return is execution telemetry, not an additional task oracle.\n"
            f"Last WLA return: {result}"
        )

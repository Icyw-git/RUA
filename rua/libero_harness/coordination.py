"""Explicit, shared controller-v2 semantics for RUA-only and RUA+WLA.

The upstream planning loop stays in place. This boundary changes camera guidance
and scopes automated recovery to gripper commands issued by atomic GRASP.
"""
from dataclasses import replace

import numpy as np

REVISION = "front_view_handoff_v2"
HOLD_STEPS = 5


def controller_prompt(template):
    start = template.index("DIRECTION:")
    end = template.index("\nATTENTION:", start)
    return template[:start] + (
        "DIRECTION (front-view calibrated base-frame actions):\n"
        "Use AgentView as the ONLY directional guide for translations, including grasp alignment. "
        "The wrist camera moves/rotates with the robot: use it to inspect the grasp, "
        "not as a fixed left/right/forward/backward coordinate system.\n"
        "- MV_LEFT / MV_RIGHT move the end effector toward AgentView left / right.\n"
        "- MV_FWD / MV_BACK move toward AgentView bottom / top along the table plane.\n"
        "- MV_UP / MV_DOWN change world height; use them to lift / lower.\n"
        "If front-view alignment is occluded or uncertain, do not guess a wrist direction.\n"
        "{rotation}\n"
    ) + template[end:]


class CoordinationContext:
    """No new action tokens; identical context available to controller-v2 baselines."""
    def __init__(self, session):
        self.session = session

    def action_tokens(self):
        return ()

    def render_prompt(self):
        event = getattr(self.session, "last_handoff", None)
        text = (
            "CONTROL OWNERSHIP: A closed command is not proof of holding. A gripper "
            "last controlled by WLA is not automatically reopened by atomic-grasp recovery. "
            "Use both new images and measured width; explicitly RELEASE or GRASP when needed. "
            "When switching from WLA to a physical atomic action, the executor first spends "
            f"{HOLD_STEPS} counted control steps on HANDOFF_HOLD, preserving the gripper. "
            "Your proposed atomic action is NOT queued or executed in that iteration. "
            "Inspect the returned images and choose again. DONE never claims simulator success.\n"
        )
        if event:
            text += (
                f"Last handoff: requested={event['requested_token']}; "
                f"executed=HANDOFF_HOLD only; steps={event['executed_steps']}; "
                f"observation_sequence={event['observation_sequence']}. "
                "No pending atomic action remains.\n"
            )
        return text


class GripperOwnerRecovery:
    """Compose, rather than replace, the upstream physical-width recovery rules."""
    def __init__(self, original, session):
        self.original, self.session = original, session

    def render_prompt_context(self, note):
        return self.original.render_prompt_context(note)

    def phase_from_width(self, value):
        return self.original.phase_from_width(value)

    def before_decision(self, **kwargs):
        if self.session.gripper_authority == "wla":
            return None  # New model decision, no forced reopen of delegated commands.
        return self.original.before_decision(**kwargs)

    def after_step(self, **kwargs):
        decision = self.original.after_step(**kwargs)
        if self.session.gripper_authority != "wla" or decision is None:
            return decision
        # Keep unverified-GRASP DONE rejection but never infer WLA chunk/subgoal failure.
        reject_done = (str(kwargs["token"]).upper() == "DONE"
                       and decision.event == "unverified_grasp")
        return replace(
            decision, event="delegated_gripper_unverified", token=None, release=False,
            rollback_index=None, reset_history=False, block_done=reject_done,
            grasp_empty=False, grasp_unverified=True,
            prompt_note="WLA last controlled the gripper. Width alone does not verify holding; "
                        "inspect both images and choose the next action. No automatic RELEASE.",
        )


def validate_front_camera(session, reference):
    """Certify unchanged front-camera calibration, using robot/camera state only."""
    sim = session.env.sim
    index = sim.model.camera_name2id("agentview")
    current = np.r_[sim.data.cam_xpos[index], sim.data.cam_xmat[index],
                    sim.model.cam_fovy[index]]
    np.testing.assert_allclose(current, reference, atol=1e-10, rtol=0)
    return current.tolist()

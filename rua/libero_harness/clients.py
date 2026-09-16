"""Select only the model transport; planner/controller/environment are shared."""
from .claude import ClaudeClient


def make_client(cfg, budget, trace_dir):
    if cfg.get("backend", "claude") == "qwen":
        from .local_vlm import LocalVLMClient
        return LocalVLMClient(cfg, budget, trace_dir)
    return ClaudeClient(cfg, budget, trace_dir)


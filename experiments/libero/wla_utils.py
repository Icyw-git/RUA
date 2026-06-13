import os
import sys
import json
import torch

import numpy as np
from typing import Optional, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.wla import WLA
from utils.transforms import normalize_and_pad, unnormalize_and_unpad


class WLA0:
    cache_dir = "/data/yangyi/.cache"

    def __init__(
        self,
        model_id: str,
        checkpoints_dir: str,
        norm_file_path: str,
        unnorm_key: str,
        target_image_size: int = 512,
        vae_downsample_f: int = 32,
        max_state_dim: int = 8,
        original_action_dim: int = 7,
    ):        
        print(f"Loading model from: {model_id}")
        input_size = target_image_size // vae_downsample_f

        self.model = WLA.from_pretrained(
            model_id,
            input_size=input_size,
        )

        if checkpoints_dir: 
            print(f"Loading checkpoints from: {checkpoints_dir}")
            state_dict = torch.load(
                f"{checkpoints_dir}/model.pt",
                map_location='cpu'
            )
            self.model.load_state_dict(state_dict)

        if hasattr(self.model.model, 'world_expert'):
            del self.model.model.world_expert
        if hasattr(self.model.model, 'connector'):
            del self.model.model.connector
        if hasattr(self.model.model, 'vae'):
            del self.model.model.vae

        self.model.to("cuda")
        self.model.eval()

        with open(norm_file_path, "r") as f:
            norm_stats = json.load(f)
        self.norm_stats = norm_stats[unnorm_key]
        
        self.max_state_dim = max_state_dim
        self.original_action_dim = original_action_dim
  

    @torch.inference_mode()
    def inference(
        self, 
        observation: dict, 
        instruction: str, 
    ) -> np.ndarray:
        
        image = observation["full_image"]
        states = observation["state"]
        
        states = torch.tensor(states)
        states, _ = normalize_and_pad(states, self.norm_stats['observation.state'], self.max_state_dim)
        states = states.unsqueeze(0)
        
        model_dtype = next(self.model.model.action_expert.parameters()).dtype
        model_device = next(self.model.model.action_expert.parameters()).device
        states = states.to(dtype=model_dtype, device=model_device)

        model = self.model.module if hasattr(self.model, "module") else self.model
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                actions = model.sample_actions(
                    caption=instruction,
                    input_images=image,
                    num_images_per_prompt=1,
                    states=states
                )

        actions = torch.tensor(actions)
        actions = unnormalize_and_unpad(actions, self.norm_stats['action'], self.original_action_dim)

        return actions
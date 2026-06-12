import re
import gc
import math
import torch

from torch import nn
from typing import List
from qwen_vl_utils import process_vision_info

from diffusers.models.normalization import RMSNorm
from diffusers import SanaTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from transformers import PretrainedConfig, PreTrainedModel, AutoProcessor
from transformers import (
    Qwen3VLForConditionalGeneration,
    Qwen2Config,
)

from models.transformer_encoder import Qwen2Encoder
from .action_model.action_model import LayerwiseFlowmatchingActionHead


class MLLMInContextConfig(PretrainedConfig):
    def __init__(
        self,
        mllm_id: str = "Alibaba-DAMO-Academy/RynnBrain-2B",
        diffusion_model_id: str = "SJTU-DENG-Lab/Sana_600M_512px_diffusers_64channels",
        in_channels: int = 32,
        input_size: int = 32,
        num_metaqueries: int = 32,
        _gradient_checkpointing: bool = True,
        max_input_text_tokens: int = 256,
        connector_num_hidden_layers: int = 12,
        system_prompt: str = "",
        action_condition_type: str = "no_action_condition",
        **kwargs,
    ):
        super().__init__()
        self.mllm_id = mllm_id
        self.diffusion_model_id = diffusion_model_id
        self.in_channels = in_channels
        self.input_size = input_size
        self.num_metaqueries = num_metaqueries
        self._gradient_checkpointing = _gradient_checkpointing
        self.max_input_text_tokens = max_input_text_tokens
        self.connector_num_hidden_layers = connector_num_hidden_layers
        self.system_prompt = system_prompt
        self.action_condition_type = action_condition_type

        self.max_action_dim = kwargs.get("max_action_dim")
        self.max_state_dim = kwargs.get("max_state_dim")
        self.chunk_size = kwargs.get("chunk_size")
        self.use_history_obs = kwargs.get("use_history_obs")
        self.training_mode = kwargs.get("training_mode")


        self.num_inference_timesteps = kwargs.get("num_inference_timesteps")
        self.num_target_vision_tokens = kwargs.get("num_target_vision_tokens")
        self.add_pos_embed = kwargs.get("add_pos_embed")
        self.max_seq_len = kwargs.get("max_seq_len")
        self.noise_beta_alpha = kwargs.get("noise_beta_alpha")
        self.noise_beta_beta = kwargs.get("noise_beta_beta")
        self.num_timestep_buckets = kwargs.get("num_timestep_buckets")
        self.noise_s = kwargs.get("noise_s")        
        self.diffusion_model_cfg = kwargs.get("diffusion_model_cfg", {})

class MLLMInContext(PreTrainedModel):
    def __init__(
        self,
        config: MLLMInContextConfig,
    ) -> None:
        super().__init__(config)
        self._gradient_checkpointing = config._gradient_checkpointing
        self.config = config

        self.mllm_backbone = Qwen3VLForConditionalGeneration.from_pretrained(
            config.mllm_id, 
            # attn_implementation="flash_attention_2",
            attn_implementation="eager",
            dtype=torch.bfloat16,
        )
        self.mllm_backbone.model.config.use_sliding_window = False
        self.mllm_backbone.model.config.sliding_window = None
        self.num_embeddings = self.mllm_backbone.get_input_embeddings().num_embeddings

        new_vocab_size = (
            self.num_embeddings
            + config.num_metaqueries + 2
        )
        
        try:
            self.mllm_backbone.resize_token_embeddings(new_vocab_size)
        except:
            self.mllm_backbone.resize_token_embeddings(new_vocab_size, mean_resizing=False)

        self.mllm_hidden_size = self.mllm_backbone.config.text_config.hidden_size
        self.tokenizer = AutoProcessor.from_pretrained(
            config.mllm_id, min_pixels=224 * 224, max_pixels=960 * 24 * 24
        )
        self.tokenizer.tokenizer.padding_side = "left"
        self.tokenizer.resize_fn = None
        self.tokenizer.max_input_text_tokens = config.max_input_text_tokens
        self.tokenizer.num_metaqueries = config.num_metaqueries
        self.tokenizer.system_prompt = config.system_prompt
        self.pad_token_id = getattr(
            self.tokenizer, "tokenizer", self.tokenizer
        ).pad_token_id

        tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    f"<pad_token_{i}>"
                    for i in range(self.num_embeddings - len(tokenizer))
                ]
            }
        )

        if config.num_metaqueries > 0:
            tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": ["<begin_of_img>", "<end_of_img>"]
                    + [f"<img{i}>" for i in range(self.tokenizer.num_metaqueries)]
                }
            )
            self.boi_token_id = tokenizer.convert_tokens_to_ids("<begin_of_img>")
            self.eoi_token_id = tokenizer.convert_tokens_to_ids("<end_of_img>")

        self.vision_start_token_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.vision_end_token_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        self.im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

        self.world_expert = SanaTransformer2DModel.from_pretrained(
            config.diffusion_model_id,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,         
        )
        
        input_scale = math.sqrt(5.5)

        self.connector_in_dim = self.mllm_hidden_size
        self.connector_out_dim = (
            getattr(self.world_expert.config, "caption_channels", None)
            or getattr(self.world_expert.config, "encoder_hid_dim", None)
            or getattr(self.world_expert.config, "cross_attention_dim", None)
        )

        norm = RMSNorm(self.connector_out_dim, eps=1e-5, elementwise_affine=True)
        with torch.no_grad():
            norm.weight.fill_(input_scale)

        encoder = Qwen2Encoder(
            Qwen2Config(
                hidden_size=self.connector_in_dim,
                intermediate_size=self.connector_in_dim * 4,
                num_hidden_layers=config.connector_num_hidden_layers,
                num_attention_heads=self.connector_in_dim // 64,
                num_key_value_heads=self.connector_in_dim // 64,
                initializer_range=0.014,
                use_cache=False,
                rope=True,
                qk_norm=True,
            ),
        )
        self.connector = nn.Sequential(
            encoder,
            nn.Linear(self.connector_in_dim, self.connector_out_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.connector_out_dim, self.connector_out_dim),
            norm,
        )
        
        self.action_expert = LayerwiseFlowmatchingActionHead(
            config=self.config,
            num_vl_layers=self.mllm_backbone.config.text_config.num_hidden_layers,
            vl_hidden_dim=self.mllm_hidden_size,
        )

        if config._gradient_checkpointing:
            try:
                self.mllm_backbone.gradient_checkpointing_enable({"use_reentrant": False})
            except:
                pass
            if not isinstance(self.connector, nn.Identity):
                for module in self.connector:
                    if isinstance(module, Qwen2Encoder):
                        module.gradient_checkpointing_enable({"use_reentrant": False})
            self.world_expert.enable_gradient_checkpointing()
        
        if "raw_action" in self.config.action_condition_type:
            self.action_encoder = nn.Sequential(
                nn.Linear(self.config.max_action_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, self.mllm_hidden_size)
            )
        elif "learnable_action_token" in self.config.action_condition_type:
            self.learnable_action_token = nn.Parameter(torch.zeros(1, self.config.chunk_size, self.mllm_hidden_size))

    def get_tokenizer(self):
        return self.tokenizer

    def get_tokenize_fn(self):
        return self.tokenize

    def get_resize_fn(self):
        return self.resize_fn

    @staticmethod
    @torch.no_grad()
    def tokenize(
        tokenizer, caption, image=None, text_response=None, add_generation_prompt=True, language_data=None, training_mode="image"
    ):
        if not isinstance(caption, List):
            caption = [caption]

        prefix = (
            [
                {
                    "role": "system",
                    "content": (
                        [{"type": "text", "text": tokenizer.system_prompt}]
                    ),
                },
            ]
            if tokenizer.system_prompt is not None
            else []
        )

        if not add_generation_prompt or tokenizer.num_metaqueries <= 0:
            suffix = ""
        elif "action" in training_mode:
            suffix = (
                "\n<begin_of_img>"
                + "".join([f"<img{i}>" for i in range(tokenizer.num_metaqueries)])
                + "<end_of_img><|im_end|>"
            )
        elif "image" in training_mode:
            suffix = (
                "\n<begin_of_img>"
                + "".join([f"<img{i}>" for i in range(tokenizer.num_metaqueries)])
                + "<end_of_img><|im_end|>"
            )

        caption = [
            tokenizer.decode(
                tokenizer(text=cap, return_tensors="pt", padding=False).input_ids[
                    0, : tokenizer.max_input_text_tokens
                ]
            )
            for cap in caption
        ]
        if image is not None:
            if not isinstance(image, list):
                image = [image]
            for i, img in enumerate(image):
                if img and not isinstance(img, list):
                    image[i] = [img]
            if tokenizer.resize_fn is not None:
                image = [
                    [tokenizer.resize_fn(sub_img) for sub_img in imgs] if imgs else None
                    for imgs in image
                ]

            conversations = [
                prefix
                + [
                    {
                        "role": "user",
                        "content": (
                            [{"type": "image"} for _ in imgs]
                            + [{"type": "text", "text": cap}]
                            if imgs
                            else [{"type": "text", "text": cap}]
                        ),
                    },
                ]
                for cap, imgs in zip(caption, image)
            ]
            kwargs = {"images": [imgs for imgs in image if imgs]}
        else:
            conversations = [
                prefix
                + [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": cap}],
                    },
                ]
                for cap in caption
            ]
            kwargs = dict()

        prompts = [
            tokenizer.apply_chat_template(conv, add_generation_prompt=True)
            for conv in conversations
        ]
        if text_response is not None:
            prompts = [p + t.strip() for p, t in zip(prompts, text_response)]

        prompts = [p + suffix for p in prompts]

        text_inputs = tokenizer(
            text=prompts,
            return_tensors="pt",
            padding=True,
            do_rescale=False,
            # do_rescale=True,
            **kwargs,
        )

        if "pixel_values" in text_inputs:
            text_inputs["pixel_values"] = text_inputs["pixel_values"].unsqueeze(0)


        if language_data is not None:
            conversations = []

            for item in language_data:
                images = item.get("images")                
                messages = []
                
                for turn in item["messages"]:
                    role = turn.get("role")
                    text = turn.get("content")
                    
                    if role == "user":
                        content = []
                        
                        if not text:
                            content.append({"type": "text", "text": ""})
                        else:
                            for seg in re.split(r"(<image>)", text):
                                if seg == "<image>" and images:
                                    content.append({"type": "image", "image": images.pop(0)})
                                elif seg.strip():
                                    content.append({"type": "text", "text": seg.strip()})
                        messages.append({"role": role, "content": content})
                        
                    else:
                        if not text:
                            messages.append({"role": role, "content": [{"type": "text", "text": ""}]})
                        else:
                            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

                conversations.append(messages)

            prompts = [
                tokenizer.apply_chat_template(conv, tokenize=False)
                for conv in conversations
            ]
            
            image_inputs = [
                process_vision_info(conv)[0]
                for conv in conversations
            ]
            image_inputs = [img for img in image_inputs if img] or None
            
            language_data_inputs = tokenizer(
                text=prompts,
                images=image_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )

            start_token = 77091
            end_token = 151645
            prefix_token = 151644
            suffix_token = 198
            IGNORE_INDEX = -100

            input_ids = language_data_inputs["input_ids"]
            labels = torch.full_like(input_ids, IGNORE_INDEX)

            batch_size, seq_len = input_ids.shape
            for batch_idx in range(batch_size):
                input_ids_1d = input_ids[batch_idx]

                start_positions = (input_ids_1d == start_token).nonzero(as_tuple=True)[0]

                if len(start_positions) == 0:
                    continue
                
                end_positions = (input_ids_1d == end_token).nonzero(as_tuple=True)[0]

                for start_pos in start_positions:

                    if start_pos < 1 or input_ids_1d[start_pos - 1] != prefix_token:
                        continue
                    if start_pos + 1 >= seq_len or input_ids_1d[start_pos + 1] != suffix_token:
                        continue

                    ans_start = start_pos + 2
                    
                    if ans_start >= seq_len:
                        continue

                    valid_ends = end_positions[end_positions >= ans_start]
                    
                    if len(valid_ends) > 0:
                        ans_end = valid_ends[0].item()
                        copy_end = min(ans_end + 2, seq_len)
                        labels[batch_idx, ans_start:copy_end] = input_ids[batch_idx, ans_start:copy_end]

            language_data_inputs["labels"] = labels
                
            text_inputs["language_data"] = language_data_inputs
            
            del conversations
            torch.cuda.empty_cache()
            gc.collect()
        
        else:
            text_inputs["language_data"] = None

        return text_inputs.values()


    def encode_condition_action(
        self, input_ids, mllm_output, **kwargs
    ):
        prompt_embeds = mllm_output.hidden_states

        boi_pos = torch.where(input_ids == self.boi_token_id)[1]
        eoi_pos = torch.where(input_ids == self.eoi_token_id)[1]

        batch_size, seq_len = input_ids.shape
        indices = torch.arange(seq_len, device=input_ids.device)[None, :].expand(
            batch_size, -1
        )
        mask = (indices > boi_pos[:, None]) & (indices < eoi_pos[:, None])
        mask = mask.to(prompt_embeds[0].device)
        
        prompt_embeds_all_layer = []
        for layer in prompt_embeds:
            prompt_embeds_per_layer = layer[mask].view(batch_size, -1, layer.size(-1))
            prompt_embeds_all_layer.append(prompt_embeds_per_layer)
        
        expected_layers = len(self.action_expert.model.transformer_blocks)
        
        return prompt_embeds_all_layer[-expected_layers:]


    def encode_condition(
        self, input_ids, attention_mask, mllm_output, action_cond_features, **kwargs
    ):
        prompt_embeds = mllm_output.hidden_states
        embeddings = mllm_output.hidden_states[0]

        repeats = 1 if action_cond_features is None else action_cond_features.shape[0] // input_ids.shape[0]
        if repeats > 1:
            input_ids = input_ids.repeat_interleave(repeats, dim=0)
            attention_mask = attention_mask.repeat_interleave(repeats, dim=0)
            embeddings = embeddings.repeat_interleave(repeats, dim=0)
            prompt_embeds =[p.repeat_interleave(repeats, dim=0) for p in prompt_embeds]

        if self.tokenizer.num_metaqueries > 0:
            # Get positions for all sequences in batch at once
            boi_pos = torch.where(input_ids == self.boi_token_id)[1]
            eoi_pos = torch.where(input_ids == self.eoi_token_id)[1]
            
            def get_vision_positions(input_ids, token_id, use_history_obs):
                positions = torch.full((input_ids.size(0),), -1, dtype=torch.long, device=input_ids.device)
                rows, cols = torch.where(input_ids == token_id)
                for r in rows.unique():
                    cols_r = cols[rows == r]
                    positions[r] = cols_r[1] if use_history_obs else cols_r.min()
                return positions

            vision_start = get_vision_positions(input_ids, self.vision_start_token_id, use_history_obs=self.config.use_history_obs)
            vision_end = get_vision_positions(input_ids, self.vision_end_token_id, use_history_obs=self.config.use_history_obs)


            # Create mask for selecting tokens between BOI and EOI
            batch_size, seq_len = input_ids.shape
            indices = torch.arange(seq_len, device=input_ids.device)[None, :].expand(
                batch_size, -1
            )

            prompt_embeds_mask = (indices > boi_pos[:, None]) & (indices < eoi_pos[:, None])
            embeddings_mask = (indices > vision_start[:, None]) & (indices < vision_end[:, None])

            embeddings = embeddings[embeddings_mask].view(
                batch_size, -1, embeddings.size(-1)
            )
            
            prompt_embeds_all_layer = []
            for layer in prompt_embeds:
                prompt_embeds_per_layer = layer[prompt_embeds_mask].view(
                    batch_size, -1, layer.size(-1)
                )
                prompt_embeds_all_layer.append(prompt_embeds_per_layer)

            prefix_tensors =[embeddings, action_cond_features] if action_cond_features is not None else [embeddings]
            prompt_embeds_all_layer = [
                self.connector(torch.cat(prefix_tensors + [prompt_embeds_layer], dim=1))
                for prompt_embeds_layer in prompt_embeds_all_layer
            ]

            expected_layers = len(self.world_expert.transformer_blocks)
            
            mid_mask =[
                torch.ones(batch_size, action_cond_features.shape[1], dtype=attention_mask.dtype, device=attention_mask.device)
            ] if action_cond_features is not None else[]

            attention_mask = torch.cat([
                attention_mask[embeddings_mask].view(batch_size, -1), 
                *mid_mask,
                attention_mask[prompt_embeds_mask].view(batch_size, -1)
            ], dim=1)

        return prompt_embeds_all_layer[-expected_layers:], attention_mask


    def forward(self, hidden_states, timestep, encoder_hidden_states=None, encoder_attention_mask=None):

        attention_mask = None
        guidance = None
        controlnet_block_samples = None
        hidden_states_dtype = hidden_states[0].dtype

        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = (1 - attention_mask.to(hidden_states_dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states_dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        batch_size, num_channels, height, width = hidden_states.shape
        p = self.world_expert.config.patch_size
        post_patch_height, post_patch_width = height // p, width // p

        hidden_states = self.world_expert.patch_embed(hidden_states)

        if guidance is not None:
            timestep, embedded_timestep = self.world_expert.time_embed(
                timestep, guidance=guidance, hidden_dtype=hidden_states_dtype
            )
        else:
            timestep, embedded_timestep = self.world_expert.time_embed(
                timestep, batch_size=batch_size, hidden_dtype=hidden_states_dtype
            )
        
        encoder_hidden_states =[
            self.world_expert.caption_norm(
                self.world_expert.caption_projection(encoder_hidden_state).view(batch_size, -1, hidden_states.shape[-1])
            )
            for encoder_hidden_state in encoder_hidden_states
        ]

        # 2. Transformer blocks
        if torch.is_grad_enabled() and self.world_expert.gradient_checkpointing:
            for index_block, block in enumerate(self.world_expert.transformer_blocks):
                hidden_states = self.world_expert._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states[index_block],
                    encoder_attention_mask,
                    timestep,
                    post_patch_height,
                    post_patch_width,
                )
                if controlnet_block_samples is not None and 0 < index_block <= len(controlnet_block_samples):
                    hidden_states = hidden_states + controlnet_block_samples[index_block - 1]

        else:
            for index_block, block in enumerate(self.world_expert.transformer_blocks):
                hidden_states = block(
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states[index_block],
                    encoder_attention_mask,
                    timestep,
                    post_patch_height,
                    post_patch_width,
                )
                if controlnet_block_samples is not None and 0 < index_block <= len(controlnet_block_samples):
                    hidden_states = hidden_states + controlnet_block_samples[index_block - 1]

        # 3. Normalization
        hidden_states = self.world_expert.norm_out(hidden_states, embedded_timestep, self.world_expert.scale_shift_table)

        hidden_states = self.world_expert.proj_out(hidden_states)

        # 5. Unpatchify
        hidden_states = hidden_states.reshape(
            batch_size, post_patch_height, post_patch_width, self.world_expert.config.patch_size, self.world_expert.config.patch_size, -1
        )
        hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
        output = hidden_states.reshape(batch_size, -1, post_patch_height * p, post_patch_width * p)
        
        return Transformer2DModelOutput(sample=output).sample

import PIL
import torch
import numpy as np

from tqdm import tqdm
from typing import Optional, Union, List
from models.model import MLLMInContextConfig, MLLMInContext

from transformers import PreTrainedModel
from diffusers.models import AutoencoderDC
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import numpy_to_pil
from diffusers.schedulers import (
    DDPMScheduler,
    FlowMatchEulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)


class WLAConfig(MLLMInContextConfig):
    model_type = "wla"

    def __init__(
        self,
        vae_id: str = "SJTU-DENG-Lab/Sana_600M_512px_diffusers_64channels",
        input_size: int = 16,
        in_channels: int = 32,
        vae_downsample_f: int = 32,
        noise_scheduler_id: str = "SJTU-DENG-Lab/Sana_600M_512px_diffusers_64channels",
        scheduler_id: str = "SJTU-DENG-Lab/Sana_600M_512px_diffusers_64channels",
        _gradient_checkpointing: bool = True,
        num_metaqueries: int = 256,
        modules_to_freeze: tuple[str] = (),
        modules_to_unfreeze: tuple[str] = (),
        training_mode: str = "action",
        **kwargs,
    ):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.vae_id = vae_id
        self.input_size = input_size
        self.in_channels = in_channels
        self.vae_downsample_f = vae_downsample_f
        self.noise_scheduler_id = noise_scheduler_id
        self.scheduler_id = scheduler_id
        self._gradient_checkpointing = _gradient_checkpointing
        self.num_metaqueries = num_metaqueries
        self.modules_to_freeze = modules_to_freeze
        self.modules_to_unfreeze = modules_to_unfreeze
        self.training_mode = training_mode


class WLA(PreTrainedModel):
    config_class = WLAConfig

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config
        self.model = MLLMInContext(MLLMInContextConfig(**config.to_dict()))
        self.vae = AutoencoderDC.from_pretrained(config.vae_id, subfolder="vae", use_safetensors=True)
        self.training_mode = config.training_mode

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            config.noise_scheduler_id, subfolder="scheduler", use_safetensors=True
        )

        self.scheduler = DPMSolverMultistepScheduler.from_pretrained(
            config.scheduler_id, subfolder="scheduler", use_safetensors=True
        )

        for module_name in config.modules_to_freeze:
            if "." in module_name:
                module = self
                for sub_module_name in module_name.split("."):
                    module = getattr(module, sub_module_name, None)
                    if module is None:
                        break
                else:
                    module.requires_grad_(False)
            else:
                module = getattr(self, module_name, None)
                if module is not None:
                    module.requires_grad_(False)

        for module_name in config.modules_to_unfreeze:
            if "." in module_name:
                module = self
                for sub_module_name in module_name.split("."):
                    module = getattr(module, sub_module_name, None)
                    if module is None:
                        break
                else:
                    module.requires_grad_(True)
            else:
                module = getattr(self, module_name, None)
                if module is not None:
                    module.requires_grad_(True)
    
    def get_input_embeddings(self):
        return self.model.mllm_backbone.model.language_model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.model.mllm_backbone.model.language_model.embed_tokens = value

    def get_output_embeddings(self):
        return self.model.mllm_backbone.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.model.mllm_backbone.lm_head = new_embeddings

    def get_sigmas(self, timesteps, device, n_dim=4, dtype=torch.float32):
        sigmas = self.noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def get_tokenizer(self):
        return self.model.get_tokenizer()

    def get_tokenize_fn(self):
        return self.model.get_tokenize_fn()

    def forward(
        self, pixel_values=None, input_ids=None, attention_mask=None, **kwargs
    ):
        compute_image_loss = kwargs.get("compute_image_loss", "image" in self.training_mode)
        requested_action_loss = kwargs.get("compute_action_loss", "action" in self.training_mode)
        subtask_labels = kwargs.get("subtask_labels", None)
        mllm_labels = subtask_labels if "language" in self.training_mode else None

        mllm_output = self.model.mllm_backbone(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=kwargs.get("image_sizes", None),
            attention_mask=attention_mask,
            labels=mllm_labels,
            output_hidden_states=True,
        )

        image_loss = None
        action_loss = None
        language_loss = mllm_output.loss if mllm_labels is not None else None

        if "image" in self.training_mode and compute_image_loss:
            
            target_images = kwargs.get("target_images", None)
            target_depths = kwargs.get("target_depths", None)
            action_cond = kwargs.get("action_cond", None)

            latents = self.vae.encode(target_images).latent
            
            latent_depth = (
                self.vae.encode(target_depths).latent 
                if target_depths is not None and "depth" in self.training_mode
                else torch.zeros_like(latents)
            )
            latents = torch.cat([latents, latent_depth], dim=1)

            if (
                "shift_factor" in self.vae.config
                and self.vae.config.shift_factor is not None
            ):
                latents = latents - self.vae.config.shift_factor
            latents = latents * self.vae.config.scaling_factor

            bsz = latents.shape[0]

            if (
                pixel_values is not None
                and hasattr(self.model, "mllm_type")
                and self.model.mllm_type == "qwenvl"
            ):
                pixel_values = pixel_values.squeeze(0)

            noise = torch.randn_like(latents, device=latents.device)

            weighting_scheme = "uniform"
            u = compute_density_for_timestep_sampling(
                weighting_scheme=weighting_scheme,
                batch_size=bsz,
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=1.29,
            )
            indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
            timesteps = self.noise_scheduler.timesteps[indices].to(
                device=latents.device
            )

            sigmas = self.get_sigmas(
                timesteps, latents.device, n_dim=latents.ndim, dtype=latents.dtype
            )
            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
            
            if "raw_action" in self.config.action_condition_type:
                action_cond_features = self.model.action_encoder(action_cond)
            elif "learnable_action_token" in self.config.action_condition_type:
                action_cond_features = self.model.learnable_action_token * action_cond
            elif "no_action_condition" in self.config.action_condition_type:
                action_cond_features = None
    
            prompt_embeds, attention_mask = self.model.encode_condition(
                input_ids=input_ids,
                attention_mask=attention_mask,
                mllm_output=mllm_output,
                action_cond_features=action_cond_features,
                current_image_index=kwargs.get("current_image_index", None),
            )

            model_pred = self.model(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=attention_mask,
            )

            target = noise - latents
            weighting = compute_loss_weighting_for_sd3(
                weighting_scheme=weighting_scheme, sigmas=sigmas
            )
            
            diff = weighting.float() * (model_pred.float() - target.float()) ** 2
            if not (target_depths is not None and "depth" in self.training_mode):
                diff = diff[:, :self.config.in_channels // 2]

            image_loss = diff.mean()

        states = kwargs.get("states", None)
        actions = kwargs.get("actions", None)
        action_mask = kwargs.get("action_mask", None)
        has_action_targets = states is not None and actions is not None and action_mask is not None
        require_action_loss = kwargs.get("require_action_loss", False)

        if "action" in self.training_mode and requested_action_loss and not has_action_targets and require_action_loss:
            raise ValueError("Action loss was requested, but states/actions/action_mask are missing.")

        if "action" in self.training_mode and requested_action_loss and has_action_targets:
            states = states.unsqueeze(1)
            
            action_prompt_embeds = self.model.encode_condition_action(
                input_ids=input_ids,
                mllm_output=mllm_output, 
            )

            repeated_diffusion_steps = self.config.repeated_diffusion_steps
            actions_repeated = actions.repeat(repeated_diffusion_steps, 1, 1)
            action_mask_repeated = action_mask.repeat(repeated_diffusion_steps, 1, 1)
            action_prompt_embeds_repeated = [h.repeat(repeated_diffusion_steps, 1, 1) for h in action_prompt_embeds]
            state_repeated = states.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.model.action_expert(
                action_prompt_embeds_repeated, 
                actions_repeated,
                action_mask_repeated,
                state_repeated
            )

        if "image_action" in self.training_mode:
            loss_config = {
                "image_loss": (image_loss, 0.1),
                "action_loss": (action_loss, 1.0),
            }
        elif "image" in self.training_mode:
            loss_config = {
                "image_loss": (image_loss, 1.0),
            }
        else:
            loss_config = {
                "action_loss": (action_loss, 1.0),
            }
        if "language" in self.training_mode:
            loss_config["language_loss"] = (language_loss, 0.005)

        total_loss = None
        loss_dict = {}

        for loss_name, (loss_value, weight) in loss_config.items():
            if loss_value is not None:
                weighted_loss = weight * loss_value
                total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
                loss_dict[loss_name] = loss_value
        
        return {
            "loss": total_loss,
            **loss_dict
        }

    @torch.no_grad()
    def decode_latents(self, latents, normalize=True, return_tensor=False):

        latents = latents / self.vae.config.scaling_factor
        if (
            "shift_factor" in self.vae.config
            and self.vae.config.shift_factor is not None
        ):
            latents = latents + self.vae.config.shift_factor
        
        samples = self.vae.decode(latents[:, :32]).sample
        if normalize:
            samples = (samples / 2 + 0.5).clamp(0, 1)
        else:
            samples = samples.clamp(-1, 1)

        samples_depth = self.vae.decode(latents[:, 32:]).sample
        if normalize:
            samples_depth = (samples_depth / 2 + 0.5).clamp(0, 1)
        else:
            samples_depth = samples_depth.clamp(-1, 1)

        if return_tensor:
            return samples, samples_depth
        
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()
        samples = numpy_to_pil(samples)

        samples_depth = samples_depth.cpu().permute(0, 2, 3, 1).float().numpy()
        samples_depth = numpy_to_pil(samples_depth)

        return samples, samples_depth

    def _as_batch_list(self, value, batch_size, default=""):
        if value is None:
            return [default for _ in range(batch_size)]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().tolist()
        if isinstance(value, list):
            return value
        return [value for _ in range(batch_size)]

    def _normalize_input_images(self, input_images):
        if input_images is None:
            return None
        if isinstance(input_images, PIL.Image.Image):
            return [[input_images]]
        if isinstance(input_images, list) and (not input_images or not isinstance(input_images[0], list)):
            return [input_images]
        assert isinstance(input_images, list) and all(
            isinstance(sublist, list) for sublist in input_images
        ), "input_images needs to be a nested list"
        return input_images

    @torch.no_grad()
    def generate_subtask(
        self,
        caption="",
        input_images=None,
        history_subtask_text=None,
        max_new_tokens=None,
        **kwargs,
    ):
        device = next(self.parameters()).device
        single_caption = not isinstance(caption, list)
        if single_caption:
            caption = [caption]

        input_images = self._normalize_input_images(input_images)
        batch_size = len(caption)
        history_subtask_text = self._as_batch_list(history_subtask_text, batch_size)

        tokenize_func = self.get_tokenize_fn()
        tokenizer = self.get_tokenizer()

        if input_images is not None:
            input_ids, attention_mask, pixel_values, image_sizes = tokenize_func(
                tokenizer,
                caption,
                input_images,
                training_mode=self.training_mode,
                history_subtask_text=history_subtask_text,
                append_metaquery=False,
            )
        else:
            input_ids, attention_mask = tokenize_func(
                tokenizer,
                caption,
                training_mode=self.training_mode,
                history_subtask_text=history_subtask_text,
                append_metaquery=False,
            )
            pixel_values = None
            image_sizes = None

        input_ids = input_ids.to(device=device)
        attention_mask = attention_mask.to(device=device)
        pixel_values = pixel_values.to(device=device) if pixel_values is not None else None
        image_sizes = image_sizes.to(device=device) if image_sizes is not None else None

        generated_ids = self.model.mllm_backbone.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_sizes,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens or 256,
            do_sample=False,
            pad_token_id=self.model.pad_token_id,
            eos_token_id=self.model.im_end_token_id,
        )
        generated_ids = generated_ids[:, input_ids.shape[1]:]
        decoded = tokenizer.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        decoded = [text.split("<|im_end|>")[0].strip() for text in decoded]
        return decoded[0] if single_caption else decoded

    def sample_images(
        self,
        caption="",
        input_images=None,
        guidance_scale: float = 3.0,
        image_guidance_scale: float = 1.5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 30,
        num_images_per_prompt: int = 1,
        return_tensor=False,
        negative_prompt="",
        enable_progress_bar=False,
        **kwargs,
    ):
        device = next(self.parameters()).device
        action_cond = kwargs.get("action_cond", None)
        current_image_index = kwargs.get("current_image_index", None)
        if action_cond is not None:
            action_cond = action_cond.unsqueeze(0)

        if not isinstance(caption, list):
            caption = [caption]

        input_images = self._normalize_input_images(input_images)

        bsz = len(caption)
        if current_image_index is not None:
            if not isinstance(current_image_index, torch.Tensor):
                current_image_index = torch.tensor(current_image_index, dtype=torch.long)
            else:
                current_image_index = current_image_index.to(dtype=torch.long)
            current_image_index = current_image_index.reshape(-1)
        do_image_classifier_free_guidance = image_guidance_scale > 1.0  # True
        do_image_classifier_free_guidance = False

        tokenize_func = self.get_tokenize_fn()
        tokenizer = self.get_tokenizer()

        if input_images is not None:
            if do_image_classifier_free_guidance:
                if action_cond is not None:
                    action_cond = action_cond.to(device=device).repeat_interleave(
                        3, dim=0
                    )
                caption = [negative_prompt] * bsz * 2 + caption
                input_images_null = [
                    (
                        [
                            PIL.Image.new("RGB", (img.size[0], img.size[1]))
                            for img in images
                        ]
                        if images
                        else None
                    )
                    for images in input_images
                ]
                input_images = input_images_null + input_images * 2
                if current_image_index is not None:
                    current_image_index = current_image_index.repeat(3)
            else:
                if action_cond is not None:
                    action_cond = action_cond.to(device=device).repeat_interleave(
                        2, dim=0
                    )
                caption = [negative_prompt] * bsz + caption
                input_images = input_images * 2
                if current_image_index is not None:
                    current_image_index = current_image_index.repeat(2)

            input_ids, attention_mask, pixel_values, image_sizes = tokenize_func(
                tokenizer, caption, input_images, training_mode="image"
            )
        else:
            do_image_classifier_free_guidance = False
            caption = [negative_prompt] * bsz + caption
            if current_image_index is not None:
                current_image_index = current_image_index.repeat(2)
            input_ids, attention_mask = tokenize_func(tokenizer, caption, training_mode="image")
            pixel_values = None
            image_sizes = None

        latent_size = self.config.input_size
        latent_channels = self.config.in_channels

        latents = randn_tensor(
            shape=(
                bsz * num_images_per_prompt,
                latent_channels,
                latent_size,
                latent_size,
            ),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )

        # set step values
        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            self.scheduler.set_timesteps(num_inference_steps, sigmas=sigmas)
        else:
            self.scheduler.set_timesteps(num_inference_steps)

        # Repeat pixel_values and conditions for each image per prompt
        input_ids = input_ids.to(device=device).repeat_interleave(
            num_images_per_prompt, dim=0
        )

        attention_mask = attention_mask.to(device=device).repeat_interleave(
            num_images_per_prompt, dim=0
        )
        if current_image_index is not None:
            current_image_index = current_image_index.to(device=device).repeat_interleave(
                num_images_per_prompt, dim=0
            )

        if action_cond is not None:
            action_cond = action_cond.to(device=device).repeat_interleave(
                num_images_per_prompt, dim=0
            )

        pixel_values = (
            pixel_values.to(device=device)
            .reshape(bsz, -1, *pixel_values.shape[1:])
            .repeat_interleave(num_images_per_prompt, dim=0)
            .flatten(0, 1)
            if pixel_values is not None
            else None
        )

        image_sizes = (
            image_sizes.to(device=device).repeat_interleave(
                num_images_per_prompt, dim=0
            )
            if image_sizes is not None
            else None
        )

        mllm_output = self.model.mllm_backbone(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_sizes,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        if "raw_action" in self.config.action_condition_type:
            action_cond_features = self.model.action_encoder(action_cond) if action_cond is not None else None
        elif "learnable_action_token" in self.config.action_condition_type:
            action_cond_features = action_cond * self.model.learnable_action_token if action_cond is not None else None
        elif "no_action_condition" in self.config.action_condition_type:
            action_cond_features = None
        else:
            action_cond_features = None

        prompt_embeds, attention_mask = self.model.encode_condition(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mllm_output=mllm_output,
            action_cond_features=action_cond_features,
            current_image_index=current_image_index,
        )
        # Convert to float32 before saving
        for t in tqdm(
            self.scheduler.timesteps,
            desc="Sampling images",
            disable=not enable_progress_bar,
        ):
            latent_model_input = torch.cat([latents] * (len(input_ids) // len(latents)))

            ###### Test All Layer ######
            latent_model_input = latent_model_input.to(prompt_embeds[0].dtype if isinstance(prompt_embeds, list) else prompt_embeds.dtype)
            if hasattr(self.scheduler, "scale_model_input"):
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

            # predict noise model_output
            noise_pred = self.model(
                hidden_states=latent_model_input,
                timestep=t.unsqueeze(0)
                .expand(latent_model_input.shape[0])
                .to(latents.device),
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=attention_mask,
            )

            # perform guidance
            if do_image_classifier_free_guidance:
                noise_pred_uncond, noise_pred_uncond_text, noise_pred = (
                    noise_pred.chunk(3)
                )
                noise_pred = (
                    noise_pred_uncond
                    + image_guidance_scale
                    * (noise_pred_uncond_text - noise_pred_uncond)
                    + guidance_scale * (noise_pred - noise_pred_uncond_text)
                )
            else:
                noise_pred_uncond, noise_pred = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred - noise_pred_uncond
                )
                # pass

            # compute previous image: x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        samples, samples_depth = self.decode_latents(
            latents.to(self.vae.dtype) if self.vae is not None else latents,
            return_tensor=return_tensor,
        )
        return samples, samples_depth

    def sample_actions(
        self,
        caption="",
        input_images=None,
        num_images_per_prompt: int = 1,
        **kwargs,
    ):
        # eval_mode = kwargs.get("eval_mode")
        device = next(self.parameters()).device
        history_subtask_text = kwargs.get("history_subtask_text", None)
        provided_subtask_text = kwargs.get("subtask_text", None)

        if not isinstance(caption, list):
            caption = [caption]

        input_images = self._normalize_input_images(input_images)
        bsz = len(caption)
        history_subtask_text = self._as_batch_list(history_subtask_text, bsz)
        language_mode = "language" in str(self.config.training_mode)

        if language_mode:
            if provided_subtask_text is None:
                subtask_text = self.generate_subtask(
                    caption=caption,
                    input_images=input_images,
                    history_subtask_text=history_subtask_text,
                )
            else:
                subtask_text = provided_subtask_text
            subtask_text = self._as_batch_list(subtask_text, bsz)
        else:
            subtask_text = None

        tokenize_func = self.get_tokenize_fn()
        tokenizer = self.get_tokenizer()

        input_ids, attention_mask, pixel_values, image_sizes = tokenize_func(
            tokenizer,
            caption,
            input_images,
            training_mode="action",
            history_subtask_text=history_subtask_text if language_mode else None,
            target_subtask_text=subtask_text if language_mode else None,
        )

        # Repeat pixel_values and conditions for each image per prompt
        input_ids = input_ids.to(device=device).repeat_interleave(
            num_images_per_prompt, dim=0
        )
        attention_mask = attention_mask.to(device=device).repeat_interleave(
            num_images_per_prompt, dim=0
        )
        pixel_values = (
            pixel_values.to(device=device)
            .reshape(bsz, -1, *pixel_values.shape[1:])
            .repeat_interleave(num_images_per_prompt, dim=0)
            .flatten(0, 1)
            if pixel_values is not None
            else None
        )
        image_sizes = (
            image_sizes.to(device=device).repeat_interleave(
                num_images_per_prompt, dim=0
            )
            if image_sizes is not None
            else None
        )
        
        states = kwargs.get("states", None)
        states = states.unsqueeze(1)

        mllm_output = self.model.mllm_backbone(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_sizes,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        action_prompt_embeds = self.model.encode_condition_action(
            input_ids=input_ids,
            mllm_output=mllm_output, 
        )
        
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.model.action_expert.predict_action(action_prompt_embeds, states)
        
        return pred_actions.detach().cpu().to(torch.float32).numpy()[0]

import os
import yaml
import torch
import wandb
import shutil
import datasets
import transformers
import PIL.Image

from PIL import PngImagePlugin
from dataclasses import dataclass, field
from typing import Optional
from accelerate.utils import release_memory
from transformers.trainer_utils import get_last_checkpoint
from torchvision.transforms.functional import to_pil_image

from dataset import get_train_datasets
from models.wla import WLAConfig, WLA
from trainer import WLATrainer, WLACallback, SaveCallback
from trainer_utils import possible_override_args, find_newest_checkpoint, get_full_dirs


datasets.disable_caching()
os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["WANDB_PROJECT"] = "WLA"
os.environ["WANDB_MODE"] = "offline"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

PIL.Image.MAX_IMAGE_PIXELS = None
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class OverrideArguments:
    config_file: str = None


@dataclass
class DiffusionModelConfig:
    cross_attention_dim: int = 2048
    dropout: float = 0.0
    final_dropout: bool = True
    interleave_self_attention: bool = True
    norm_type: str = "ada_norm"
    num_layers: int = 16
    output_dim: int = 1024
    positional_embeddings: Optional[str] = None


@dataclass
class ModelArguments:
    _gradient_checkpointing: bool = True
    vae_id: str = "SJTU-DENG-Lab/Sana_600M_512px_diffusers_64channels"
    in_channels: int = 32
    vae_downsample_f: int = 32
    noise_scheduler_id: str = "SJTU-DENG-Lab/Sana_600M_512px_diffusers_64channels"
    scheduler_id: str = "SJTU-DENG-Lab/Sana_600M_512px_diffusers_64channels"
    mllm_id: str = "Alibaba-DAMO-Academy/RynnBrain-2B"
    diffusion_model_id: str = "SJTU-DENG-Lab/Sana_600M_512px_diffusers_64channels"
    num_metaqueries: int = 9
    modules_to_freeze: tuple[str] = ()
    modules_to_unfreeze: tuple[str] = ()
    max_input_text_tokens: int = 256
    connector_num_hidden_layers: int = 12
    system_prompt: str = (
        ""
    )
    chunk_size: int = 32
    sample_num: int = 4
    use_history_obs: bool = True
    history_obs_step: int = 8
    use_begin_frame_context: bool = False
    auxiliary_drop_thresh: float = 0.1
    max_action_dim: int = 14
    max_state_dim: int = 14

    training_mode: str = "action"

    num_inference_timesteps: int = 4
    num_target_vision_tokens: int = 32
    add_pos_embed: bool = True
    max_seq_len: int = 1024
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    num_timestep_buckets: int = 1000
    noise_s: float = 0.999
    repeated_diffusion_steps: int = 4

    diffusion_model_cfg: DiffusionModelConfig = field(
        default_factory=lambda: DiffusionModelConfig()
    )
    
    instruction_folder_path: str = ""
    action_condition_type: str = "learnable_action_token"


@dataclass
class DataArguments:
    train_datasets: dict[str, float] = field(
        default_factory=lambda: {
            "default_dataset": -1,
        }
    )
    eval_dataset: str = "libero"
    target_image_size: int = 512
    primary_image_size: int = 512
    auxiliary_image_size: int = 256
    dataset_root_dir: str = "LIBERO_LeRobot"
    quality_set: str = "auto"  # auto, nominal, or recovery for RUA quality-starts datasets
    robotwin_image_only_dataset_root_dir: Optional[str] = None
    norm_stats_path: str = "configs/norm_stats.json"
    camera_keys_path: str = "configs/robocoin_camera_keys.json"
    unnorm_key: str = "libero_all"


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    ddp_timeout: int = 36000
    base_dir: str = "."
    output_dir: str = "output"
    save_dir: str = "checkpoints"
    save_part_checkpoints: bool = True
    data_dir: str = ".cache"
    eval_on_start: bool = True
    eval_strategy: str = "steps"
    eval_steps: int = 5000
    eval_delay: int = 0
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    optim: str = "adamw_torch"
    learning_rate: float = 1e-4
    weight_decay: float = 1.0e-08
    # weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    gradient_clipping: float = 1.0
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine_with_min_lr"
    lr_scheduler_kwargs: dict = field(default_factory=lambda: {"min_lr": 1e-5})
    logging_steps: int = 10
    warmup_steps: int = 5000
    save_strategy: str = "steps"
    save_steps: int = 5000
    save_total_limit: int = 1
    restore_callback_states_from_checkpoint: bool = True
    seed: int = 42
    data_seed: int = 42
    bf16: bool = True
    tf32: bool = True
    dataloader_num_workers: int = 32
    datasets_num_proc: int = os.getenv("OMP_NUM_THREADS", 12)
    dataloader_persistent_workers: bool = False
    dataloader_pin_memory: bool = True
    dataloader_drop_last: bool = True
    remove_unused_columns: bool = False
    run_name: str = "test"
    report_to: str = "wandb"
    ddp_find_unused_parameters: bool = False
    overwrite_output_dir: bool = False
    resume_from_checkpoint: str = None
    disable_tqdm: bool = True

    def __post_init__(self):
        try:
            self = possible_override_args(override_args, self)
            self = get_full_dirs(self)
        except (FileNotFoundError, yaml.YAMLError) as exc:
            print(f"Failed to load override config: {exc}")
        super().__post_init__()


if __name__ == "__main__":
    override_parser = transformers.HfArgumentParser((OverrideArguments))
    override_args = override_parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )[0]
    parser = transformers.HfArgumentParser(
        (OverrideArguments, ModelArguments, DataArguments, TrainingArguments)
    )
    _, model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args, data_args = possible_override_args(override_args, model_args, data_args)

    assert (
        data_args.target_image_size % model_args.vae_downsample_f == 0
    ), f"Image size must be divisible by {model_args.vae_downsample_f}"
    input_size = data_args.target_image_size // model_args.vae_downsample_f

    if training_args.resume_from_checkpoint is not None:
        training_args.resume_from_checkpoint = find_newest_checkpoint(
            training_args.resume_from_checkpoint
        )
        model = WLA.from_pretrained(
            training_args.resume_from_checkpoint,
            input_size=input_size,
            ignore_mismatched_sizes=True,
            torch_dtype=torch.bfloat16,
            dtype=torch.bfloat16,
            **model_args.__dict__,
        )
    else:
        model = WLA(
            config=WLAConfig(
                input_size=input_size,
                **model_args.__dict__,
            ),
        )

    training_mode = model.config.training_mode
    num_embeddings = model.model.num_embeddings
    num_metaqueries = model.config.num_metaqueries

    def freeze_hook(grad, mode):
        if mode in ["image", "action", "image_action"]:
            grad[:num_embeddings].zero_()
        return grad
    
    # model.model.mllm_backbone.model.embed_tokens.weight.register_hook(lambda grad: freeze_hook(grad, training_mode))
    # model.model.mllm_backbone.model.language_model.embed_tokens.weight.register_hook(lambda grad: freeze_hook(grad, training_mode))

    with training_args.main_process_first(local=False):
        train_dataset, eval_dataset, gt_images, gt_depths, src_images, collate_fn = get_train_datasets(
            data_args,
            training_args,
            model_args,
            model.get_tokenize_fn(),
            model.get_tokenizer(),
        )

    trainer = WLATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        callbacks=[WLACallback(), SaveCallback()],
    )
    
    def log_image_collection(log_key, images):
        if not images:
            return
        if isinstance(images, dict):
            for name, image_list in images.items():
                log_image_collection(f"{log_key}/{name}", image_list)
            return
        trainer.log_images({
            log_key: [
                wandb.Image(image) if isinstance(image, PIL.Image.Image) else wandb.Image(to_pil_image(image))
                for image in images
            ]
        })

    log_image_collection("src_images", src_images)
    log_image_collection("gt_images", gt_images)
    log_image_collection("gt_depths", gt_depths)

    training_args.output_dir = str(
        os.path.join(training_args.output_dir, training_args.run_name)
    )
    if trainer.is_world_process_zero():
        if training_args.overwrite_output_dir and os.path.exists(
            training_args.output_dir
        ):
            shutil.rmtree(training_args.output_dir)
        print(f"Training dataset size: {len(train_dataset)}")

    while (
        trainer.state.global_step < training_args.max_steps if training_args.max_steps > 0 
        else (
            trainer.state.epoch is None
            or (training_args.num_train_epochs - trainer.state.epoch) > 0.01
        )
    ):
        if trainer.state.epoch is not None:
            trainer.control.should_training_stop = False
            trainer.args.eval_on_start = False
            trainer.model = model
            (trainer.model_wrapped,) = release_memory(trainer.model_wrapped)
            trainer.model_wrapped = trainer.model
        last_checkpoint = None
        if (
            os.path.isdir(training_args.output_dir)
            and not training_args.overwrite_output_dir
        ):
            last_checkpoint = get_last_checkpoint(training_args.output_dir)


        if training_args.resume_from_checkpoint is not None:
            trainer.train(resume_from_checkpoint=False)
        else:
            trainer.train(resume_from_checkpoint=last_checkpoint)

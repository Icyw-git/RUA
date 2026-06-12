import os
import io
import re
import PIL
import json
import h5py
import torch
import random
import numpy as np

from tqdm import tqdm
from functools import partial
from torch.utils.data import ConcatDataset
from torchvision.transforms import v2
from torchcodec.decoders import VideoDecoder
from utils.data_utils import index_episodes, index_episodes_egodex, get_robocoin_list
from utils.transforms import _make_transform, resize_with_pad, normalize_and_pad, pad_to_dim
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata, MultiLeRobotDataset


class LeRobotTrainDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        base_dataset,
        target_transform,
        primary_image_size,
        auxiliary_image_size,
        primary_image_key,
        auxiliary_image_key,
        norm_stats,
        model_args,
        language_dataset=None,
        instruction_dict=None,
        primary_depth_key=None,
        include_actions=True,
    ):
        self.dataset = base_dataset
        self.target_transform = target_transform
        self.primary_image_key = primary_image_key
        self.primary_depth_key = primary_depth_key
        self.auxiliary_image_key = auxiliary_image_key
        self.norm_stats = norm_stats
        self.model_args = model_args
        self.language_dataset = language_dataset
        self.instruction_dict = instruction_dict
        self.include_actions = include_actions
        self.primary_image_transform = _make_transform(primary_image_size)
        self.auxiliary_image_transform = _make_transform(auxiliary_image_size)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        task = item.get("task", "")
        caption = "" if random.random() < 0.05 else (
            random.choice(self.instruction_dict[task])
            if self.instruction_dict and task in self.instruction_dict else task
        )
        
        result = {'caption': caption}
        
        images = item[self.primary_image_key]
        target_images = images[1:-1] if self.model_args.use_history_obs else images[1:]
        result["target_images"] = [self.target_transform(img) for img in target_images]
        
        if "depth" in self.model_args.training_mode and self.primary_depth_key:
            target_depths = item[self.primary_depth_key][1:]
            result["target_depths"] = [self.target_transform(img) for img in target_depths]
      
        if self.include_actions:
            result["states"], _ = normalize_and_pad(item['observation.state'], self.norm_stats['observation.state'], self.model_args.max_state_dim)
            normalized_actions, action_mask = normalize_and_pad(item['action'], self.norm_stats["action"], self.model_args.max_action_dim)
            padded_actions = pad_to_dim(item['action'], self.model_args.max_action_dim)[0]
            result["actions"], result["action_mask"] = normalized_actions, action_mask

            sample_num, chunk_size = self.model_args.sample_num, self.model_args.chunk_size
            
            if "raw_action" in self.model_args.action_condition_type:
                limits = torch.arange(0, sample_num + 1, device=padded_actions.device).unsqueeze(1) * chunk_size // max(1, sample_num)
                result["action_cond"] = padded_actions.unsqueeze(0) * (
                    torch.arange(padded_actions.shape[0], device=padded_actions.device) < limits
                ).unsqueeze(-1)

            elif "learnable_action_token" in self.model_args.action_condition_type:
                limits = torch.arange(1, sample_num + 1, dtype=torch.float32).unsqueeze(1) * chunk_size // sample_num
                result["action_cond"] = (torch.arange(chunk_size, dtype=torch.float32) < limits).unsqueeze(-1)

        input_imgs = [(item[self.primary_image_key][0], self.primary_image_transform)]
        if self.model_args.use_history_obs:
            input_imgs.insert(0, (item[self.primary_image_key][-1], self.auxiliary_image_transform))

        if random.random() >= self.model_args.auxiliary_drop_thresh:
            input_imgs +=[(item[key], self.auxiliary_image_transform) for key in self.auxiliary_image_key if key is not None]

        null_mask = torch.rand(len(input_imgs)) < 0
        result["input_images"] = [
            transform(torch.zeros_like(img) if mask and img is not None else img) 
            if img is not None else None
            for (img, transform), mask in zip(input_imgs, null_mask)
        ]
        
        if self.language_dataset is not None:
            random_idx = random.randint(0, len(self.language_dataset) - 1)
            result["language_data"] = self.language_dataset[random_idx]

        return result

class LeRobotEvalDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        base_dataset,
        primary_image_size, 
        auxiliary_image_size,
        primary_image_key,
        auxiliary_image_key,
        norm_stats,
        model_args,
        instruction_dict,
        primary_depth_key=None,
        include_actions=True,
    ):
        self.dataset = base_dataset
        self.primary_image_key = primary_image_key
        self.primary_depth_key = primary_depth_key
        self.auxiliary_image_key = auxiliary_image_key
        self.norm_stats = norm_stats
        self.model_args = model_args
        self.instruction_dict = instruction_dict
        self.include_actions = include_actions
        self.primary_image_transform = _make_transform(primary_image_size)
        self.auxiliary_image_transform = _make_transform(auxiliary_image_size)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        task = item.get("task", "")
        caption = random.choice(self.instruction_dict[task]) if self.instruction_dict and task in self.instruction_dict else task
        
        result = {'caption': caption}

        images = item[self.primary_image_key]
        result["target_images"] = images[1:-1] if self.model_args.use_history_obs else images[1:]
        
        if "depth" in self.model_args.training_mode and self.primary_depth_key:
            result["target_depths"] = item[self.primary_depth_key][1:]

        if self.include_actions:
            result["states"], _ = normalize_and_pad(item['observation.state'], self.norm_stats['observation.state'], self.model_args.max_state_dim)
            normalized_actions, action_mask = normalize_and_pad(item['action'], self.norm_stats["action"], self.model_args.max_action_dim)
            padded_actions = pad_to_dim(item['action'], self.model_args.max_action_dim)[0]
            result["actions"], result["action_mask"] = normalized_actions, action_mask
            
            sample_num, chunk_size = self.model_args.sample_num, self.model_args.chunk_size
            if "raw_action" in self.model_args.action_condition_type:
                limits = torch.arange(0, sample_num + 1, device=padded_actions.device).unsqueeze(1) * chunk_size // max(1, sample_num)
                result["action_cond"] = padded_actions.unsqueeze(0) * (
                    torch.arange(padded_actions.shape[0], device=padded_actions.device) < limits
                ).unsqueeze(-1)

            elif "learnable_action_token" in self.model_args.action_condition_type:
                limits = torch.arange(1, sample_num + 1, dtype=torch.float32).unsqueeze(1) * chunk_size // sample_num
                result["action_cond"] = (torch.arange(chunk_size, dtype=torch.float32) < limits).unsqueeze(-1)

        input_imgs = [(item[self.primary_image_key][0], self.primary_image_transform)]
        if self.model_args.use_history_obs:
            input_imgs.insert(0, (item[self.primary_image_key][-1], self.auxiliary_image_transform))

        if random.random() > self.model_args.auxiliary_drop_thresh:
            input_imgs +=[(item[key], self.auxiliary_image_transform) for key in self.auxiliary_image_key if key is not None]

        result["input_images"] = [
            transform(img) if img is not None else None
            for (img, transform) in input_imgs
        ]
        return result


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, primary_dataset, secondary_dataset):
        self.primary_dataset = primary_dataset
        self.secondary_dataset = secondary_dataset
        self.primary_len = len(primary_dataset)
        self.secondary_len = len(secondary_dataset)
        if self.primary_len == 0 or self.secondary_len == 0:
            raise ValueError("PairedDataset requires both datasets to be non-empty.")
        self.length = max(self.primary_len, self.secondary_len)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "primary": self.primary_dataset[idx % self.primary_len],
            "secondary": self.secondary_dataset[idx % self.secondary_len],
        }


class EgodexRawDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, model_args):
        self.chunk_size = model_args.chunk_size
        self.sample_num = model_args.sample_num
        self.history_obs_step = model_args.history_obs_step if model_args.use_history_obs else None
        
        self.dataset_path_list, self.raw_episode_lens = index_episodes_egodex(dataset_path)
        self.effective_episode_lens = [max(0, len - self.chunk_size) for len in self.raw_episode_lens]
        self.cumulative_len = np.cumsum(self.effective_episode_lens)

    def __len__(self):
        return sum(self.effective_episode_lens)
    
    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1], f"Index {index} out of bounds"
        episode_index = np.argmax(self.cumulative_len > index)
        start_ts = index - (self.cumulative_len[episode_index] - self.effective_episode_lens[episode_index])
        return int(episode_index), int(start_ts)

    def __getitem__(self, idx):
        
        episode_id, frame_id = self._locate_transition(idx)
        hdf5_file = self.dataset_path_list[episode_id]
        mp4_file = hdf5_file[:-5] + '.mp4' 
        
        with h5py.File(hdf5_file, "r") as root:
            if root.attrs['llm_type'] == 'reversible':
                direction = root.attrs['which_llm_description']
                lang_instruct = root.attrs['llm_description' if direction == '1' else 'llm_description2'] 
            else:
                lang_instruct = root.attrs['llm_description']
        
        frames = VideoDecoder(mp4_file, device='cpu')
            
        history_idx = max(0, frame_id - self.history_obs_step) if self.history_obs_step else None
        target_idxs = [frame_id + self.chunk_size] +[
            frame_id + int(self.chunk_size * j / max(1, self.sample_num)) 
            for j in range(1, self.sample_num + 1)
        ]
        
        return {
            "history_image": frames[history_idx] if history_idx is not None else None,
            "current_image": frames[frame_id],
            "target_images": [frames[i] for i in target_idxs],
            "caption": lang_instruct
        }


class EgodexDatasetWrapper(torch.utils.data.Dataset):
    def __init__(
        self, 
        raw_dataset, 
        target_transform, 
        primary_image_size,
        auxiliary_image_size,
        model_args, 
        is_train=True
    ):
        self.dataset = raw_dataset
        self.target_transform = target_transform
        self.model_args = model_args
        self.is_train = is_train
        self.primary_image_transform = _make_transform(primary_image_size)
        self.auxiliary_image_transform = _make_transform(auxiliary_image_size)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        caption = "" if (self.is_train and random.random() < 0.05) else item["caption"]
        input_images = [(item["current_image"], self.primary_image_transform)]
        if self.model_args.use_history_obs:
            input_images.insert(0, (item["history_image"], self.auxiliary_image_transform))
        
        null_mask = torch.rand(len(input_images)) < 0
        input_images = [
            transform(torch.zeros_like(img) if mask and img is not None else img) 
            if img is not None else None
            for (img, transform), mask in zip(input_images, null_mask)
        ]
        
        target_images = item["target_images"]
        if self.is_train:
            target_images = [self.target_transform(img) for img in target_images]

        action_cond = torch.zeros(
            self.model_args.sample_num + 1, 
            self.model_args.chunk_size, 
            self.model_args.max_action_dim
        )

        return {
            "caption": caption,
            "input_images": input_images,
            "target_images": target_images,
            "action_cond": action_cond
        }


class VideoRawDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, model_args):
        self.chunk_size = model_args.chunk_size
        self.sample_num = model_args.sample_num
        self.action_condition_type = model_args.action_condition_type
        self.history_obs_step = model_args.history_obs_step if model_args.use_history_obs else None
        
        self.dataset_path_list, self.raw_episode_lens, self.captions = index_episodes(dataset_path)
        self.effective_episode_lens = [max(0, len - self.chunk_size) for len in self.raw_episode_lens]
        self.cumulative_len = np.cumsum(self.effective_episode_lens)

    def __len__(self):
        return sum(self.effective_episode_lens)
    
    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1], f"Index {index} out of bounds"
        episode_index = np.argmax(self.cumulative_len > index)
        start_ts = index - (self.cumulative_len[episode_index] - self.effective_episode_lens[episode_index])
        return int(episode_index), int(start_ts)

    def __getitem__(self, idx):
        
        episode_id, frame_id = self._locate_transition(idx)
        mp4_file = self.dataset_path_list[episode_id]
        caption = self.captions[episode_id]

        frames = VideoDecoder(mp4_file, device='cpu')
            
        history_idx = max(0, frame_id - self.history_obs_step) if self.history_obs_step else None
        if "raw_action" in self.action_condition_type:
            target_idxs = [frame_id + self.chunk_size] +[
                frame_id + int(self.chunk_size * j / max(1, self.sample_num)) 
                for j in range(1, self.sample_num + 1)
            ]
        elif "learnable_action_token" in self.action_condition_type:
            target_idxs = [
                frame_id + int(self.chunk_size * j / self.sample_num) 
                for j in range(1, self.sample_num + 1)
            ]
        elif "no_action_condition" in self.action_condition_type:
            target_idxs = [frame_id + self.chunk_size]

        return {
            "history_image": frames[history_idx] if history_idx is not None else None,
            "current_image": frames[frame_id],
            "target_images": [frames[i] for i in target_idxs],
            "caption": caption
        }
        

class VideoDatasetWrapper(torch.utils.data.Dataset):
    def __init__(
        self, 
        raw_dataset, 
        target_transform, 
        primary_image_size,
        auxiliary_image_size,
        instruction_dict,
        model_args, 
        is_train=True
    ):
        self.dataset = raw_dataset
        self.target_transform = target_transform
        self.model_args = model_args
        self.is_train = is_train
        self.instruction_dict = instruction_dict
        self.primary_image_transform = _make_transform(primary_image_size)
        self.auxiliary_image_transform = _make_transform(auxiliary_image_size)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        caption_options = self.instruction_dict.get(item["caption"]) if self.instruction_dict else None
        caption = random.choice(caption_options) if caption_options else item["caption"]
        caption = "" if (self.is_train and random.random() < 0.05) else caption
            
        input_images = [(item["current_image"], self.primary_image_transform)]
        if self.model_args.use_history_obs:
            input_images.insert(0, (item["history_image"], self.auxiliary_image_transform))
        
        null_mask = torch.rand(len(input_images)) < 0
        input_images = [
            transform(torch.zeros_like(img) if mask and img is not None else img) 
            if img is not None else None
            for (img, transform), mask in zip(input_images, null_mask)
        ]
        
        target_images = item["target_images"]
        if self.is_train:
            target_images = [self.target_transform(img) for img in target_images]

        action_cond = torch.zeros(
            self.model_args.sample_num + 1, 
            self.model_args.chunk_size, 
            self.model_args.max_action_dim
        )

        return {
            "caption": caption,
            "input_images": input_images,
            "target_images": target_images,
            "action_cond": action_cond
        }


def _collate_fn(batch, tokenize_func, tokenizer, model_args):
    if batch and "primary" in batch[0] and "secondary" in batch[0]:
        return {
            "primary": _collate_fn(
                [example["primary"] for example in batch],
                tokenize_func=tokenize_func,
                tokenizer=tokenizer,
                model_args=model_args,
            ),
            "secondary": _collate_fn(
                [example["secondary"] for example in batch],
                tokenize_func=tokenize_func,
                tokenizer=tokenizer,
                model_args=model_args,
            ),
        }

    input_images = [example.get("input_images") for example in batch]
    return_dict = {"input_images": input_images}
    
    if "image" in model_args.training_mode:
        return_dict["target_images"] = torch.stack([item for example in batch for item in example["target_images"]])
        if  model_args.action_condition_type in ["raw_action", "learnable_action_token"]:
            return_dict["action_cond"] = torch.cat([torch.as_tensor(example["action_cond"]) for example in batch], dim=0)
        if "depth" in model_args.training_mode:
            return_dict["target_depths"] = torch.stack([item for example in batch for item in example["target_depths"]])
    
    has_actions = all("actions" in example and "action_mask" in example and "states" in example for example in batch)
    if "action" in model_args.training_mode and has_actions:
        return_dict["actions"] = torch.stack([
            example["actions"] if isinstance(example["actions"], torch.Tensor) 
            else torch.tensor(example["actions"])
            for example in batch
        ])
        return_dict["action_mask"] = torch.stack([
            example["action_mask"] if isinstance(example["action_mask"], torch.Tensor) 
            else torch.tensor(example["action_mask"])
            for example in batch
        ])
        return_dict["states"] = torch.stack([
            example["states"] if isinstance(example["states"], torch.Tensor) 
            else torch.tensor(example["states"])
            for example in batch
        ])

    captions = [example["caption"] for example in batch]
    language_data = [example.get("language_data") for example in batch] if "language" in model_args.training_mode else None

    if any(imgs is not None for imgs in input_images):
        (
            return_dict["input_ids"],
            return_dict["attention_mask"],
            return_dict["pixel_values"],
            return_dict["image_sizes"],
            return_dict["language_data"],
        ) = tokenize_func(tokenizer, captions, input_images, language_data=language_data, training_mode=model_args.training_mode)
    else:
        return_dict["input_ids"], return_dict["attention_mask"] = tokenize_func(
            tokenizer, captions, training_mode=model_args.training_mode
        )
    return return_dict


def get_train_datasets(data_args, training_args, model_args, tokenize_func, tokenizer):
    target_transform = v2.Compose([
        lambda img: resize_with_pad(img, data_args.target_image_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5], [0.5]),
    ])
    ground_truth_transform = _make_transform(data_args.primary_image_size)

    collate_fn = partial(
        _collate_fn, 
        tokenize_func=tokenize_func, 
        tokenizer=tokenizer, 
        model_args=model_args,
    )

    if "egodex" in data_args.train_datasets:
        train_dataset = EgodexRawDataset(data_args.dataset_root_dir, model_args)        
        random.seed(training_args.data_seed)
        eval_dataset = torch.utils.data.Subset(
            train_dataset,
            random.sample(range(len(train_dataset)), training_args.world_size)
        )
        
        train_dataset = EgodexDatasetWrapper(
            train_dataset,
            target_transform=target_transform,
            primary_image_size=data_args.primary_image_size,
            auxiliary_image_size=data_args.auxiliary_image_size,
            model_args=model_args,
            is_train=True
        )
        
        eval_dataset = EgodexDatasetWrapper(
            eval_dataset,
            target_transform=target_transform,
            primary_image_size=data_args.primary_image_size,
            auxiliary_image_size=data_args.auxiliary_image_size,
            model_args=model_args,
            is_train=False
        )
        
        src_idx = 1 if model_args.use_history_obs else 0
        src_images = [ground_truth_transform(sample["input_images"][src_idx]) for sample in eval_dataset]
        gt_images = [ground_truth_transform(sample["target_images"][0]) for sample in eval_dataset]
        gt_depths = None
    
    elif "robotwin_video" in data_args.train_datasets:
        train_dataset = VideoRawDataset(data_args.dataset_root_dir, model_args)        
        random.seed(training_args.data_seed)
        eval_dataset = torch.utils.data.Subset(
            train_dataset,
            random.sample(range(len(train_dataset)), training_args.world_size)
        )
        
        with open(model_args.instruction_folder_path, 'r', encoding='utf-8') as f:
            instruction_dict = json.load(f)
        
        train_dataset = VideoDatasetWrapper(
            train_dataset,
            target_transform=target_transform,
            primary_image_size=data_args.primary_image_size,
            auxiliary_image_size=data_args.auxiliary_image_size,
            instruction_dict=instruction_dict,
            model_args=model_args,
            is_train=True
        )
        
        eval_dataset = VideoDatasetWrapper(
            eval_dataset,
            target_transform=target_transform,
            primary_image_size=data_args.primary_image_size,
            auxiliary_image_size=data_args.auxiliary_image_size,
            instruction_dict=instruction_dict,
            model_args=model_args,
            is_train=False
        )
        
        src_idx = 1 if model_args.use_history_obs else 0
        src_images = [ground_truth_transform(sample["input_images"][src_idx]) for sample in eval_dataset]
        gt_images = [ground_truth_transform(sample["target_images"][0]) for sample in eval_dataset]
        gt_depths = None
        

    elif "libero" in data_args.train_datasets:
        train_dataset, norm_stats, primary_image_key, auxiliary_image_key = load_libero_dataset(data_args, model_args, training_args)
        random.seed(training_args.data_seed)
        eval_dataset = torch.utils.data.Subset(
            train_dataset,
            random.sample(range(len(train_dataset)), training_args.world_size)
        )
        
        train_dataset = LeRobotTrainDataset(
            base_dataset=train_dataset,
            target_transform=target_transform,
            primary_image_size=data_args.primary_image_size,
            auxiliary_image_size=data_args.auxiliary_image_size,
            primary_image_key=primary_image_key,
            primary_depth_key=None,
            auxiliary_image_key=auxiliary_image_key,
            norm_stats=norm_stats,
            model_args=model_args,
            language_dataset=None,
            instruction_dict=None
        )
        
        eval_dataset = LeRobotEvalDataset(
            base_dataset=eval_dataset,
            primary_image_size=data_args.primary_image_size,
            auxiliary_image_size=data_args.auxiliary_image_size,
            primary_image_key=primary_image_key,
            primary_depth_key=None,
            auxiliary_image_key=auxiliary_image_key,
            norm_stats=norm_stats,
            model_args=model_args,
            instruction_dict=None
        )

        src_idx = 1 if model_args.use_history_obs else 0
        src_images = [ground_truth_transform(sample["input_images"][src_idx]) for sample in eval_dataset]
        gt_images = [ground_truth_transform(sample["target_images"][0]) for sample in eval_dataset]
        gt_depths = (
            [ground_truth_transform(sample["target_depths"][0]) for sample in eval_dataset]
            if "depth" in model_args.training_mode else None
        )


    elif "piper" in data_args.train_datasets:
        train_dataset, norm_stats, primary_image_key, auxiliary_image_key = load_piper_dataset(data_args, model_args, training_args)
        random.seed(training_args.data_seed)
        eval_dataset = torch.utils.data.Subset(
            train_dataset,
            random.sample(range(len(train_dataset)), training_args.world_size)
        )
        
        train_dataset = LeRobotTrainDataset(
            base_dataset=train_dataset,
            target_transform=target_transform,
            primary_image_size=data_args.primary_image_size,
            auxiliary_image_size=data_args.auxiliary_image_size,
            primary_image_key=primary_image_key,
            primary_depth_key=None,
            auxiliary_image_key=auxiliary_image_key,
            norm_stats=norm_stats,
            model_args=model_args,
            language_dataset=None,
            instruction_dict=None
        )
        
        eval_dataset = LeRobotEvalDataset(
            base_dataset=eval_dataset,
            primary_image_size=data_args.primary_image_size,
            auxiliary_image_size=data_args.auxiliary_image_size,
            primary_image_key=primary_image_key,
            primary_depth_key=None,
            auxiliary_image_key=auxiliary_image_key,
            norm_stats=norm_stats,
            model_args=model_args,
            instruction_dict=None
        )

        src_idx = 1 if model_args.use_history_obs else 0
        src_images = [ground_truth_transform(sample["input_images"][src_idx]) for sample in eval_dataset]
        gt_images = [ground_truth_transform(sample["target_images"][0]) for sample in eval_dataset]
        gt_depths = (
            [ground_truth_transform(sample["target_depths"][0]) for sample in eval_dataset]
            if "depth" in model_args.training_mode else None
        )

    elif "robotwin" in data_args.train_datasets:
        primary_base_dataset, norm_stats, instruction_dict, primary_image_key, primary_depth_key, auxiliary_image_key = \
            load_robotwin_dataset(data_args, model_args, training_args, include_actions=True)

        random.seed(training_args.data_seed)
        primary_eval_dataset = torch.utils.data.Subset(
            primary_base_dataset,
            random.sample(range(len(primary_base_dataset)), min(len(primary_base_dataset), training_args.world_size))
        )
        
        language_dataset = load_language_dataset(data_args, training_args) if "language" in model_args.training_mode else None

        primary_train_dataset = LeRobotTrainDataset(
            base_dataset=primary_base_dataset,
            target_transform=target_transform,
            primary_image_size=data_args.primary_image_size,
            auxiliary_image_size=data_args.auxiliary_image_size,
            primary_image_key=primary_image_key,
            primary_depth_key=primary_depth_key,
            auxiliary_image_key=auxiliary_image_key,
            norm_stats=norm_stats,
            model_args=model_args,
            language_dataset=language_dataset,
            instruction_dict=instruction_dict,
            include_actions=True,
        )
        primary_eval_dataset = LeRobotEvalDataset(
            base_dataset=primary_eval_dataset,
            primary_image_size=data_args.primary_image_size,
            auxiliary_image_size=data_args.auxiliary_image_size,
            primary_image_key=primary_image_key,
            primary_depth_key=primary_depth_key,
            auxiliary_image_key=auxiliary_image_key,
            norm_stats=norm_stats,
            model_args=model_args,
            instruction_dict=instruction_dict,
            include_actions=True,
        )

        src_idx = 1 if model_args.use_history_obs else 0
        collect_src_images = lambda dataset: [ground_truth_transform(sample["input_images"][src_idx]) for sample in dataset]
        collect_gt_images = lambda dataset: [ground_truth_transform(sample["target_images"][0]) for sample in dataset]
        collect_gt_depths = lambda dataset: (
            [ground_truth_transform(sample["target_depths"][0]) for sample in dataset]
            if "depth" in model_args.training_mode else None
        )

        if data_args.robotwin_image_only_dataset_root_dir:
            if "no_action_condition" not in model_args.action_condition_type:
                raise ValueError("Paired robotwin image training requires action_condition_type=no_action_condition.")

            secondary_dataset_root_dir = data_args.robotwin_image_only_dataset_root_dir
            if _is_video_raw_dataset(secondary_dataset_root_dir):
                secondary_base_dataset = VideoRawDataset(secondary_dataset_root_dir, model_args)

                random.seed(training_args.data_seed)
                secondary_eval_dataset = torch.utils.data.Subset(
                    secondary_base_dataset,
                    random.sample(range(len(secondary_base_dataset)), min(len(secondary_base_dataset), training_args.world_size))
                )

                secondary_train_dataset = VideoDatasetWrapper(
                    secondary_base_dataset,
                    target_transform=target_transform,
                    primary_image_size=data_args.primary_image_size,
                    auxiliary_image_size=data_args.auxiliary_image_size,
                    instruction_dict=None,
                    model_args=model_args,
                    is_train=True
                )
                secondary_eval_dataset = VideoDatasetWrapper(
                    secondary_eval_dataset,
                    target_transform=target_transform,
                    primary_image_size=data_args.primary_image_size,
                    auxiliary_image_size=data_args.auxiliary_image_size,
                    instruction_dict=None,
                    model_args=model_args,
                    is_train=False
                )
            else:
                secondary_base_dataset, secondary_norm_stats, secondary_instruction_dict, secondary_primary_image_key, secondary_primary_depth_key, secondary_auxiliary_image_key = \
                    load_robotwin_dataset(
                        data_args,
                        model_args,
                        training_args,
                        dataset_root_dir=secondary_dataset_root_dir,
                        include_actions=False,
                    )

                random.seed(training_args.data_seed)
                secondary_eval_dataset = torch.utils.data.Subset(
                    secondary_base_dataset,
                    random.sample(range(len(secondary_base_dataset)), min(len(secondary_base_dataset), training_args.world_size))
                )

                secondary_train_dataset = LeRobotTrainDataset(
                    base_dataset=secondary_base_dataset,
                    target_transform=target_transform,
                    primary_image_size=data_args.primary_image_size,
                    auxiliary_image_size=data_args.auxiliary_image_size,
                    primary_image_key=secondary_primary_image_key,
                    primary_depth_key=secondary_primary_depth_key,
                    auxiliary_image_key=secondary_auxiliary_image_key,
                    norm_stats=secondary_norm_stats,
                    model_args=model_args,
                    language_dataset=None,
                    instruction_dict=secondary_instruction_dict,
                    include_actions=False,
                )
                secondary_eval_dataset = LeRobotEvalDataset(
                    base_dataset=secondary_eval_dataset,
                    primary_image_size=data_args.primary_image_size,
                    auxiliary_image_size=data_args.auxiliary_image_size,
                    primary_image_key=secondary_primary_image_key,
                    primary_depth_key=secondary_primary_depth_key,
                    auxiliary_image_key=secondary_auxiliary_image_key,
                    norm_stats=secondary_norm_stats,
                    model_args=model_args,
                    instruction_dict=secondary_instruction_dict,
                    include_actions=False,
                )

            train_dataset = PairedDataset(primary_train_dataset, secondary_train_dataset)
            eval_dataset = {
                "primary": primary_eval_dataset,
                "secondary": secondary_eval_dataset,
            }
            src_images = {
                "primary": collect_src_images(primary_eval_dataset),
                "secondary": collect_src_images(secondary_eval_dataset),
            }
            gt_images = {
                "primary": collect_gt_images(primary_eval_dataset),
                "secondary": collect_gt_images(secondary_eval_dataset),
            }
            gt_depths = {
                "primary": collect_gt_depths(primary_eval_dataset),
                "secondary": collect_gt_depths(secondary_eval_dataset),
            } if "depth" in model_args.training_mode else None
        else:
            train_dataset = primary_train_dataset
            eval_dataset = primary_eval_dataset
            src_images = collect_src_images(eval_dataset)
            gt_images = collect_gt_images(eval_dataset)
            gt_depths = collect_gt_depths(eval_dataset)
    
    elif "robocoin" in data_args.train_datasets:
        target_robots = [
            "AgiBot-g1", "AIRBOT_MMK2", "alpha_bot_2", "Cobot_Magic", 
            "G1edu-u3", "Galbot_g1", "R1_Lite", "Split_aloha", "Tianqin_A2"
        ]
        robocoin_list = get_robocoin_list(data_args.dataset_root_dir, target_robots)
        
        language_dataset = load_language_dataset(data_args, training_args) if "language" in model_args.training_mode else None
        train_list, eval_list = [], []
        
        random.seed(training_args.data_seed)        
        eval_indices = set(
            random.sample(
                range(len(robocoin_list)), 
                min(len(robocoin_list), training_args.world_size)
            )
        )

        for idx, robot_dataset in enumerate(robocoin_list):
            train_dataset, norm_stats, primary_image_key, auxiliary_image_key = \
                load_robocoin_dataset(robot_dataset, data_args, model_args, training_args)
        
            train_list.append(
                LeRobotTrainDataset(
                    base_dataset=train_dataset,
                    target_transform=target_transform,
                    primary_image_size=data_args.primary_image_size,
                    auxiliary_image_size=data_args.auxiliary_image_size,
                    primary_image_key=primary_image_key,
                    primary_depth_key=None,
                    auxiliary_image_key=auxiliary_image_key,
                    norm_stats=norm_stats,
                    model_args=model_args,
                    language_dataset=language_dataset,
                    instruction_dict=None
                )
            )
            
            if idx in eval_indices:
                eval_dataset = torch.utils.data.Subset(
                    train_dataset,
                    random.sample(range(len(train_dataset)), 1)
                )
                eval_list.append(
                    LeRobotEvalDataset(
                        base_dataset=eval_dataset,
                        primary_image_size=data_args.primary_image_size,
                        auxiliary_image_size=data_args.auxiliary_image_size,
                        primary_image_key=primary_image_key,
                        primary_depth_key=None,
                        auxiliary_image_key=auxiliary_image_key,
                        norm_stats=norm_stats,
                        model_args=model_args,
                        instruction_dict=None
                    )
                )
        
        train_dataset = ConcatDataset(train_list)
        eval_dataset = ConcatDataset(eval_list)

        src_idx = 1 if model_args.use_history_obs else 0
        src_images = [ground_truth_transform(sample["input_images"][src_idx]) for sample in eval_dataset]
        gt_images = [ground_truth_transform(sample["target_images"][0]) for sample in eval_dataset]
        gt_depths = (
            [ground_truth_transform(sample["target_depths"][0]) for sample in eval_dataset]
            if "depth" in model_args.training_mode else None
        )

    return train_dataset, eval_dataset, gt_images, gt_depths, src_images, collate_fn


def load_language_dataset(data_args, training_args):
    language_dataset_dir = data_args.language_dataset_dir
    language_dataset = []
    
    json_dir = os.path.join(language_dataset_dir, "json_files")
    images_dir = os.path.join(language_dataset_dir, 'images')
    
    for json_file in os.listdir(json_dir):
        if not json_file.endswith('.json'):
            continue
        
        json_name = os.path.splitext(json_file)[0]
        json_path = os.path.join(json_dir, json_file)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        for item in json_data:
            messages = item.get('messages')
            images = item.get('images')

            images = [os.path.join(images_dir, json_name, img) for img in images] if images else None

            language_dataset.append({
                'messages': messages,
                'images': images
            })
    random.seed(training_args.data_seed)
    random.shuffle(language_dataset)
    return language_dataset


def load_libero_dataset(data_args, model_args, training_args):
    ds_meta = LeRobotDatasetMetadata(
        os.path.join(data_args.dataset_root_dir, "libero_spatial_no_noops_lerobot")
    )
    primary_image_key, wrist_image_key = ds_meta.camera_keys[0], ds_meta.camera_keys[1]
    
    with open(data_args.norm_stats_path, 'r') as f:
        norm_stats = json.load(f)
    norm_stats = norm_stats[data_args.unnorm_key]

    common_ts = [0] + [
        model_args.chunk_size * i / (model_args.sample_num * ds_meta.fps) 
        for i in range(1, model_args.sample_num + 1)
    ]

    if "raw_action" in model_args.action_condition_type:
        common_ts.insert(1, model_args.chunk_size / ds_meta.fps)
    if model_args.use_history_obs:
        common_ts.append(-model_args.history_obs_step / ds_meta.fps)

    delta_timestamps = {
        primary_image_key: list(common_ts),
        "action": [t / ds_meta.fps for t in range(model_args.chunk_size)],
    }

    dataset = MultiLeRobotDataset(
        repo_ids = [
            os.path.join(data_args.dataset_root_dir, item) 
                for item in os.listdir(data_args.dataset_root_dir) 
            if os.path.isdir(os.path.join(data_args.dataset_root_dir, item))
        ],
        delta_timestamps=delta_timestamps,
        video_backend="pyav"
    )

    return dataset, norm_stats, primary_image_key, [wrist_image_key]


ROBOTWIN_VARIANT_FOLDERS = ("aloha-agilex_clean_50", "aloha-agilex_randomized_500")


def _is_lerobot_repo(path):
    return os.path.isdir(os.path.join(path, "meta"))


def _is_video_raw_dataset(path):
    return os.path.isfile(os.path.join(path, "video_info.json"))


def _discover_lerobot_repo_ids(root_dir):
    if _is_lerobot_repo(root_dir):
        return [root_dir]

    repo_ids = []
    for task_name in sorted(os.listdir(root_dir)):
        task_path = os.path.join(root_dir, task_name)
        if not os.path.isdir(task_path):
            continue
        for folder_name in ROBOTWIN_VARIANT_FOLDERS:
            repo_path = os.path.join(task_path, folder_name)
            if _is_lerobot_repo(repo_path):
                repo_ids.append(repo_path)

    return sorted(set(repo_ids))


def load_robotwin_dataset(
    data_args,
    model_args,
    training_args,
    dataset_root_dir=None,
    include_actions=True,
):
    dataset_root_dir = dataset_root_dir or data_args.dataset_root_dir
    repo_ids = _discover_lerobot_repo_ids(dataset_root_dir)
    if not repo_ids:
        variant_folders = ", ".join(ROBOTWIN_VARIANT_FOLDERS)
        raise ValueError(
            f"No LeRobot repositories found under {dataset_root_dir}. "
            f"Expected directories like <root>/<task>/{{{variant_folders}}}."
        )

    ds_meta = LeRobotDatasetMetadata(repo_ids[0])
    primary_image_key, left_wrist_image_key, right_wrist_image_key = (*ds_meta.camera_keys, None, None, None)[:3]
        
    primary_depth_key = None

    with open(data_args.norm_stats_path, 'r') as f:
        norm_stats = json.load(f)
    norm_stats = norm_stats[data_args.unnorm_key]

    common_ts = [0] + [
        model_args.chunk_size * i / (model_args.sample_num * ds_meta.fps) 
        for i in range(1, model_args.sample_num + 1)
    ]

    if "raw_action" in model_args.action_condition_type:
        common_ts.insert(1, model_args.chunk_size / ds_meta.fps)
    if model_args.use_history_obs:
        common_ts.append(-model_args.history_obs_step / ds_meta.fps)

    delta_timestamps = {
        primary_image_key: list(common_ts),
    }
    if include_actions:
        delta_timestamps["action"] = [t / ds_meta.fps for t in range(model_args.chunk_size)]

    dataset = MultiLeRobotDataset(
        repo_ids=repo_ids,
        delta_timestamps=delta_timestamps,
        video_backend="pyav"
    )

    with open(model_args.instruction_folder_path, 'r', encoding='utf-8') as f:
        instruction_dict = json.load(f)

    return dataset, norm_stats, instruction_dict, primary_image_key, primary_depth_key, [left_wrist_image_key, right_wrist_image_key]



def load_piper_dataset(data_args, model_args, training_args):
    ds_meta = LeRobotDatasetMetadata(data_args.dataset_root_dir)
    primary_image_key, left_wrist_image_key, right_wrist_image_key = \
        ds_meta.camera_keys[0], ds_meta.camera_keys[1], ds_meta.camera_keys[2]

    with open(data_args.norm_stats_path, 'r') as f:
        norm_stats = json.load(f)
    norm_stats = norm_stats[data_args.unnorm_key]

    common_ts = [0] + [
        model_args.chunk_size * i / (model_args.sample_num * ds_meta.fps) 
        for i in range(1, model_args.sample_num + 1)
    ]

    if "raw_action" in model_args.action_condition_type:
        common_ts.insert(1, model_args.chunk_size / ds_meta.fps)
    if model_args.use_history_obs:
        common_ts.append(-model_args.history_obs_step / ds_meta.fps)

    delta_timestamps = {
        primary_image_key: list(common_ts),
        "action": [t / ds_meta.fps for t in range(model_args.chunk_size)],
    }

    dataset = LeRobotDataset(
        data_args.dataset_root_dir,
        delta_timestamps=delta_timestamps,
        video_backend="pyav"
    )

    return dataset, norm_stats, primary_image_key, [left_wrist_image_key, right_wrist_image_key]



def load_robocoin_dataset(robot_dataset, data_args, model_args, training_args):
    ds_meta = LeRobotDatasetMetadata(robot_dataset)

    with open(data_args.camera_keys_path, 'r') as f:
        camera_keys = json.load(f)
    camera_keys = camera_keys[os.path.basename(robot_dataset)]
    primary_image_key, left_wrist_image_key, right_wrist_image_key = (*camera_keys, None, None, None)[:3]

    with open(data_args.norm_stats_path, 'r') as f:
        norm_stats = json.load(f)
    norm_stats = norm_stats[os.path.basename(robot_dataset)]

    common_ts = [0] + [
        model_args.chunk_size * i / (model_args.sample_num * ds_meta.fps) 
        for i in range(1, model_args.sample_num + 1)
    ]

    if "raw_action" in model_args.action_condition_type:
        common_ts.insert(1, model_args.chunk_size / ds_meta.fps)
    if model_args.use_history_obs:
        common_ts.append(-model_args.history_obs_step / ds_meta.fps)

    delta_timestamps = {
        primary_image_key: list(common_ts),
        "action": [t / ds_meta.fps for t in range(model_args.chunk_size)],
    }

    dataset = LeRobotDataset(
        robot_dataset,
        delta_timestamps=delta_timestamps,
    )

    return dataset, norm_stats, primary_image_key, [left_wrist_image_key, right_wrist_image_key]

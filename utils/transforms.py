import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision.transforms import functional


def resize_with_pad(img, size):
    h, w = functional.get_image_size(img)[::-1]
    s = size / max(h, w)
    nh, nw = int(h * s), int(w * s)

    img = v2.Resize((nh, nw))(img)
    return functional.pad(
        img,
        [(size - nw) // 2, (size - nh) // 2,
         (size - nw + 1) // 2, (size - nh + 1) // 2],
        fill=0
    )


def _make_transform(size):
    return v2.Compose([lambda img: resize_with_pad(img, size)])


def pad_to_dim(x, target_dim, value: float = 0.0):
    current_dim = x.shape[-1]
    mask = torch.ones_like(x)
    if current_dim < target_dim:
        pad_dim = target_dim - current_dim
        pad_config = (0, pad_dim) 
        x = F.pad(x, pad_config, mode='constant', value=value)
        mask = F.pad(mask, pad_config, mode='constant', value=0.0)
    return x, mask


# def normalize_and_pad(data, norm_stats_key, max_dim):
#     data_max = torch.tensor(norm_stats_key["max"])
#     data_min = torch.tensor(norm_stats_key["min"])
#     normalized_data = 2 * (data - data_min) / (data_max - data_min + 1e-6) - 1
#     return pad_to_dim(normalized_data, max_dim)


# def unnormalize_and_unpad(data, norm_stats_key, original_dim):
#     data_max = torch.tensor(norm_stats_key["max"]).to(data.device)
#     data_min = torch.tensor(norm_stats_key["min"]).to(data.device)
#     unpadded_data = data[..., :original_dim]
#     denormalized_data = (unpadded_data + 1) * (data_max - data_min + 1e-6) / 2 + data_min
#     return denormalized_data


def normalize_and_pad(data, norm_stats_key, max_dim, add_eps=True):
    data_max = torch.tensor(norm_stats_key["max"])
    data_min = torch.tensor(norm_stats_key["min"])
    eps = 1e-6 if add_eps else 0.0
    normalized_data = 2 * (data - data_min) / (data_max - data_min + eps) - 1
    return pad_to_dim(normalized_data, max_dim)


def unnormalize_and_unpad(data, norm_stats_key, original_dim, add_eps=True):
    data_max = torch.tensor(norm_stats_key["max"]).to(data.device)
    data_min = torch.tensor(norm_stats_key["min"]).to(data.device)
    unpadded_data = data[..., :original_dim]
    eps = 1e-6 if add_eps else 0.0
    denormalized_data = (unpadded_data + 1) * (data_max - data_min + eps) / 2 + data_min
    return denormalized_data

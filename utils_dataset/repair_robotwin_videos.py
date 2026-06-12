import os
import re
import cv2
import torch
import torchvision
import numpy as np
from torchcodec.decoders import VideoDecoder

log_path = "video_check_robotwin_lerobot.log"
video_paths = set()

print(f"[*] Parsing log file: {log_path}")
if not os.path.exists(log_path):
    raise FileNotFoundError(f"Log file not found: {log_path}")

with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        match = re.search(r"Path:\s*(.*?)\s*\|\s*Frame:", line)
        if match:
            video_path = match.group(1).strip()
            video_paths.add(video_path)

print(f"[*] Log parsing complete. Found {len(video_paths)} unique videos to repair.")


for idx_vid, video_path in enumerate(video_paths, 1):
    print(f"\n========================================================")
    print(f"[*] Processing video ({idx_vid}/{len(video_paths)}): {video_path}")
    
    if not os.path.exists(video_path):
        print(f"[!] Video file does not exist, skipping: {video_path}")
        continue

    base_name, ext = os.path.splitext(video_path)
    temp_output_path = f"{base_name}_temp_fixed{ext}"
    
    try:
        decoder = VideoDecoder(video_path, device="cpu", seek_mode="approximate")
        num_frames = decoder.metadata.num_frames

        fps = 50.0
        
        frames_list =[]
        reference_frame = None
        
        print(f"[*] Starting video decoding, total frames: {num_frames}...")

        for idx in range(num_frames):
            try:
                frame_obj = decoder.get_frames_at(indices=[idx])

                if hasattr(frame_obj, 'data') and isinstance(frame_obj.data, torch.Tensor):
                    frame = frame_obj.data
                elif hasattr(frame_obj, 'tensor') and isinstance(frame_obj.tensor, torch.Tensor):
                    frame = frame_obj.tensor
                else:
                    frame = frame_obj
                    
                frames_list.append(frame)

                if reference_frame is None:
                    reference_frame = frame
                    
            except Exception as frame_err:
                frames_list.append(None)
        
        if reference_frame is None:
            print(f"[!] Error: The entire video could not be decoded, so the reference frame shape cannot be obtained. Skipping this video.")
            continue

        empty_count = sum(1 for f in frames_list if f is None)
        if empty_count > 0:
            print(f"[*] Found {empty_count} corrupted frames. Replacing them with black frames...")
            for i in range(num_frames):
                if frames_list[i] is None:
                    if isinstance(reference_frame, torch.Tensor):
                        frames_list[i] = torch.zeros_like(reference_frame)
                    elif isinstance(reference_frame, np.ndarray):
                        frames_list[i] = np.zeros_like(reference_frame)
                    else:
                        raise TypeError(f"Unknown frame data type, unable to generate black frame: {type(reference_frame)}")
        else:
            print(f"[*] Video decoded successfully. No corrupted frames found. Re-encoding will still be performed because errors may occur only in specific reading modes.")

        print(f"[*] Starting re-encoding to temporary file: {temp_output_path}")
        if isinstance(reference_frame, torch.Tensor):
            video_tensor = torch.cat(frames_list, dim=0).cpu()

            if video_tensor.dtype in[torch.float32, torch.float64]:
                video_tensor = (video_tensor * 255).clamp(0, 255).to(torch.uint8)

            if video_tensor.ndim == 4 and video_tensor.shape[1] == 3:
                video_tensor = video_tensor.permute(0, 2, 3, 1)
                
            torchvision.io.write_video(temp_output_path, video_tensor, fps=fps)

        else:
            sample = frames_list[0]
            if sample.ndim == 4:
                sample = sample[0] 
                
            if sample.shape[0] == 3:
                c, h, w = sample.shape
            else:
                h, w, c = sample.shape

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_output_path, fourcc, fps, (w, h))
            
            for f in frames_list:
                img = f[0] if f.ndim == 4 else f
                if img.shape[0] == 3: 
                    img = np.transpose(img, (1, 2, 0))

                if img.dtype in [np.float32, np.float64]:
                    img = np.clip(img * 255, 0, 255).astype(np.uint8)

                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                out.write(img_bgr)
                
            out.release()

        if os.path.exists(temp_output_path):
            os.remove(video_path)
            os.rename(temp_output_path, video_path)
            print(f"[*] Success! The original video has been deleted and replaced with the repaired video: {video_path}")
        else:
            print(f"[!] Warning: Temporary output file not found: {temp_output_path}. Replacement failed.")
            
    except Exception as e:
        print(f"[!] Critical exception! Failed to process video {video_path}. Error message: {e}")
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)

print("\n[*] Global processing complete!")
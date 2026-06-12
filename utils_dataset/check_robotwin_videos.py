import os
import logging
from pathlib import Path
import ray
from tqdm import tqdm
from torchcodec.decoders import VideoDecoder


ROOT_DIR = "datasets/RoboTwin-LeRobot"
LOG_FILE = "video_check_robotwin_lerobot.log"
NUM_CPUS = 55


logging.basicConfig(
    filename=LOG_FILE,
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@ray.remote
def check_single_video(video_path: str):
    try:
        decoder = VideoDecoder(video_path, device="cpu", seek_mode="approximate")
        
        num_frames = decoder.metadata.num_frames
        if num_frames is None or num_frames == 0:
            return f"SKIP: {video_path} (No frames found or metadata error)"

        batch_size = 64
        all_indices = list(range(num_frames))
        
        for i in range(0, num_frames, batch_size):
            current_batch_indices = all_indices[i : i + batch_size]
            try:
                decoder.get_frames_at(indices=current_batch_indices)
            except Exception:
                for idx in current_batch_indices:
                    try:
                        decoder.get_frames_at(indices=[idx])
                    except Exception as frame_err:
                        error_msg = f"FAILED: Path: {video_path} | Frame: {idx} | Error: {str(frame_err)}"
                        return error_msg
        
        return f"OK: {video_path}"

    except Exception as e:
        return f"CRITICAL_ERROR: {video_path} | Error: {str(e)}"

def main():
    if not ray.is_initialized():
        ray.init(num_cpus=NUM_CPUS)

    print(f"Scanning folder: {ROOT_DIR} ...")
    root_path = Path(ROOT_DIR)

    video_files = list(root_path.glob("**/videos/observation.images.*/chunk-*/*.mp4"))
    total_videos = len(video_files)
    print(f"Found {total_videos} video files. Starting parallel check...")

    video_paths = [str(v) for v in video_files]
    result_ids = [check_single_video.remote(p) for p in video_paths]

    pbar = tqdm(total=total_videos, desc="Video check progress")

    results = []
    ready_ids, remaining_ids = ray.wait(result_ids, num_returns=1, timeout=None)
    
    while ready_ids:
        for result_id in ready_ids:
            res = ray.get(result_id)
            results.append(res)

            if "OK" in res:
                pass
            else:
                logging.error(res)
            
            pbar.update(1)
        
        if not remaining_ids:
            break
            
        ready_ids, remaining_ids = ray.wait(remaining_ids, num_returns=1, timeout=None)

    pbar.close()

    failed_count = sum(1 for r in results if "FAILED" in r or "CRITICAL" in r)
    print(f"\nCheck completed!")
    print(f"Total videos: {total_videos}")
    print(f"Corrupted videos: {failed_count}")
    print(f"Detailed results saved to: {LOG_FILE}")

    ray.shutdown()

if __name__ == "__main__":
    main()
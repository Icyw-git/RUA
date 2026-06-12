import os
import h5py
import json
import fnmatch
from tqdm import tqdm

def index_episodes_egodex(dataset_path): 
    # find all hdf5 files
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_path):
        for filename in fnmatch.filter(files, "*.hdf5"):
            hdf5_files.append(os.path.join(root, filename))
    print(f"Found {len(hdf5_files)} hdf5 files")

    # get lengths of all hdf5 files
    all_episode_len = []
    for dataset_path in tqdm(hdf5_files, desc='iterating dataset_path to get all episode lengths...'):
        try:
            with h5py.File(dataset_path, "r") as root:
                action = root['/transforms/leftHand'][()]
        except Exception as e:
            print(f"Error loading {dataset_path}")
        all_episode_len.append(len(action))
    
    return hdf5_files, all_episode_len


def index_episodes(root_dir):        
    json_path = os.path.join(root_dir, "video_info.json")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    abs_paths_list = [
        os.path.abspath(os.path.join(root_dir, item['relative_path'])) 
        for item in data
    ]
    frames_list = [item['frame_count'] for item in data]
    captions_list = [item['caption'] for item in data]
    
    return abs_paths_list, frames_list, captions_list

def get_robocoin_list(base_dir, target_robots):
    folders = [
        f 
        for f in os.listdir(base_dir) 
        if os.path.isdir(os.path.join(base_dir, f))
    ]

    return [
        os.path.join(base_dir, f) 
        for f in folders 
        if any(robot in f for robot in target_robots)
    ]
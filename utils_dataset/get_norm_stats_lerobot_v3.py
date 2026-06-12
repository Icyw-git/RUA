import json
import numpy as np
from pathlib import Path

ACTION_DIM = 16
STATE_DIM = 16

def compute_global_stats(root_dir):
    root_path = Path(root_dir)
 
    global_stats = {
        "action": {
            "min": np.full(ACTION_DIM, np.inf),
            "max": np.full(ACTION_DIM, -np.inf)
        },
        "observation.state": {
            "min": np.full(STATE_DIM, np.inf),
            "max": np.full(STATE_DIM, -np.inf)
        }
    }

    json_files = list(root_path.glob("*/*/meta/stats.json"))
    
    print(json_files)
    
    if not json_files:
        print(f"No stats.json files found under '{root_dir}'. Please check the path.")
        return
    
    print(f"Found {len(json_files)} stats.json files. Computing global min/max values...")

    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: File {file_path} is not valid JSON.")
                continue

            for key in ["action", "observation.state"]:
                if key in data:
                    local_min = np.array(data[key]["min"])
                    local_max = np.array(data[key]["max"])

                    expected_dim = ACTION_DIM if key == "action" else STATE_DIM

                    if len(local_min) == expected_dim and len(local_max) == expected_dim:
                        global_stats[key]["min"] = np.minimum(global_stats[key]["min"], local_min)
                        global_stats[key]["max"] = np.maximum(global_stats[key]["max"], local_max)
                    else:
                        print(f"Warning: '{key}' in {file_path} is not {expected_dim}-dimensional.")
                else:
                    print(f"Warning: File {file_path} is missing the '{key}' key.")

    result = {
        "action": {
            "min": global_stats["action"]["min"].tolist(),
            "max": global_stats["action"]["max"].tolist()
        },
        "observation.state": {
            "min": global_stats["observation.state"]["min"].tolist(),
            "max": global_stats["observation.state"]["max"].tolist()
        }
    }

    print("\n=== Global statistics results ===")
    print(json.dumps(result, indent=4))

    out_file = root_path / "global_stats.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)
    print(f"\nComputation completed! Global results saved to: {out_file.absolute()}")

if __name__ == "__main__":
    TARGET_DIR = "datasets/RoboTwin-LeRobot"
    compute_global_stats(TARGET_DIR)
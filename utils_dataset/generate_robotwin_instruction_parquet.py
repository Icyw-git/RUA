import os
import pandas as pd
from pathlib import Path

ROOT_DIR = Path("datasets/RoboTwin-LeRobot")

CATEGORY_CONFIG = {
    "aloha-agilex_clean_50": {
        "num_rows": 50,
        "suffix": "demo_clean"
    },
    "aloha-agilex_randomized_500": {
        "num_rows": 500,
        "suffix": "demo_randomized"
    }
}

def main():
    if not ROOT_DIR.exists():
        print(f"Error: Root directory not found: {ROOT_DIR}")
        return

    for task_dir in ROOT_DIR.iterdir():
        if not task_dir.is_dir():
            continue
            
        task_name = task_dir.name

        for category_name, config in CATEGORY_CONFIG.items():
            category_dir = task_dir / category_name
            meta_dir = category_dir / "meta"

            if not meta_dir.exists():
                meta_dir.mkdir(parents=True, exist_ok=True)
                
            num_rows = config["num_rows"]

            instruction_str = f"{task_name}-{config['suffix']}"
            instructions = [instruction_str] * num_rows
            task_indices = list(range(num_rows)) 

            df = pd.DataFrame(
                {"task_index": task_indices},
                index=instructions
            )

            output_path = meta_dir / "tasks.parquet"
            df.to_parquet(output_path)
            
            print(f"Successfully updated: {task_name}/{category_name}/meta/tasks.parquet")
            print(f"  -> Rows: {num_rows}, instruction: {instruction_str}")

    print("\nAll files have been replaced!")

if __name__ == "__main__":
    main()
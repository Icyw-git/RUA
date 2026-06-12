import os
import json
from tqdm import tqdm

def generate_seen_tasks_json():
    data_dir = "datasets/RoboTwin2.0/dataset"
    output_file = "configs/robotwin_unseen_instruction.json"
    
    if not os.path.exists(data_dir):
        print(f"Error: Could not find the '{data_dir}' folder! Please make sure the script is in the same directory as the '{data_dir}' folder.")
        return

    result_dict = {}
    categories = ["aloha-agilex_clean_50", "aloha-agilex_randomized_500"]
    task_names =[d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    tasks_to_process =[(task, cat) for task in task_names for cat in categories]
    
    for task_name, category in tqdm(tasks_to_process, desc="Extraction progress", ncols=100):
        instructions_dir = os.path.join(data_dir, task_name, category, "instructions")
        if category == "aloha-agilex_clean_50":
            dict_key = f"{task_name}-demo_clean"
        elif category == "aloha-agilex_randomized_500":
            dict_key = f"{task_name}-demo_randomized"
        unique_seen_instructions = set()
        
        if os.path.exists(instructions_dir):
            json_files = [f for f in os.listdir(instructions_dir) if f.endswith(".json")]

            for filename in json_files:
                file_path = os.path.join(instructions_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                        seen_list = json_data.get("seen",[])
                        unique_seen_instructions.update(seen_list)
                except Exception as e:
                    tqdm.write(f"An error occurred while reading or parsing {file_path}: {e}")
        else:
            tqdm.write(f"Warning: Could not find path {instructions_dir}")

        result_dict[dict_key] = list(unique_seen_instructions)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)
        
    print(f"\n🎉 Processing complete! Successfully generated file: {output_file}, containing {len(result_dict)} keys.")


if __name__ == "__main__":
    generate_seen_tasks_json()
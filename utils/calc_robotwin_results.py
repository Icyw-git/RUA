import os

def get_last_line_as_float(file_path):
    try:
        if not os.path.exists(file_path):
            return None
        with open(file_path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
            if not lines:
                return None
            return float(lines[-1])
    except:
        return None

def main():
    root_dir = "RoboTwin/eval_result"
    output_log = "eval_results_table.log"
    modes = ['demo_clean', 'demo_randomized']
    
    table_data = {}
    validation_log = []

    if not os.path.exists(root_dir):
        print(f"Error: Directory not found: {root_dir}")
        return
    
    all_task_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    
    if len(all_task_dirs) != 50:
        validation_log.append(f"Note: Expected 50 tasks, but found {len(all_task_dirs)}.")

    for task in all_task_dirs:
        table_data[task] = {mode: None for mode in modes}
        task_path = os.path.join(root_dir, task)
        
        algos = [d for d in os.listdir(task_path) if os.path.isdir(os.path.join(task_path, d))]
        if not algos:
            validation_log.append(f"Task {task}: Missing algorithm folder")
            continue
        
        algo_name = algos[0]
        
        for mode in modes:
            mode_path = os.path.join(task_path, algo_name, mode, algo_name)
            
            if not os.path.exists(mode_path):
                continue
                
            timestamps = sorted([d for d in os.listdir(mode_path) if os.path.isdir(os.path.join(mode_path, d))])
            if not timestamps:
                continue
                
            latest_ts = timestamps[-1]
            result_file = os.path.join(mode_path, latest_ts, "_result.txt")
            
            score = get_last_line_as_float(result_file)
            table_data[task][mode] = score

    averages = {}
    for mode in modes:
        valid_scores = [table_data[t][mode] for t in table_data if table_data[t][mode] is not None]
        averages[mode] = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    with open(output_log, 'w', encoding='utf-8') as f:
        f.write("RoboTwin Evaluation Summary Report\n")
        f.write("=" * 85 + "\n")
        
        header = f"{'Task Name':<45} | {'demo_clean':<15} | {'demo_randomized':<15}"
        f.write(header + "\n")
        f.write("-" * 85 + "\n")
        
        for task in all_task_dirs:
            score_clean = table_data[task]['demo_clean']
            score_rand = table_data[task]['demo_randomized']
            
            str_clean = f"{score_clean:.4f}" if score_clean is not None else "N/A"
            str_rand = f"{score_rand:.4f}" if score_rand is not None else "N/A"
            
            f.write(f"{task:<45} | {str_clean:<15} | {str_rand:<15}\n")
            
        f.write("-" * 85 + "\n")
        
        avg_clean = f"{averages['demo_clean']:.4f}"
        avg_rand = f"{averages['demo_randomized']:.4f}"
        f.write(f"{'AVERAGE (Calculated from valid entries)':<45} | {avg_clean:<15} | {avg_rand:<15}\n")
        f.write("=" * 85 + "\n\n")

        if validation_log:
            f.write("Check Logs:\n")
            for log in validation_log:
                f.write(f"- {log}\n")

    print(f"Statistics completed! Table saved to: {output_log}")
    print(f"Total number of tasks: {len(all_task_dirs)}")
    print(f"Clean average: {averages['demo_clean']:.4f} | Randomized average: {averages['demo_randomized']:.4f}")

if __name__ == "__main__":
    main()
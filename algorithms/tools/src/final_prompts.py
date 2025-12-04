import pandas as pd
import os
from pathlib import Path

def create_best_prompts_table(source_dirs, save_folder, algo_labels):
    """
    Extract final best prompts from each method and create an Excel table.
    
    Args:
        source_dirs (list): List of source directories containing results
        save_folder (str): Folder to save the output Excel file
        algo_labels (list): List of tuples with (prefix, label) for algorithm names
    """
    
    all_prompts = {}
    baseline_prompts = {}
    
    # Iterate through each source directory (method)
    for source_dir in source_dirs:
        source_path = Path(source_dir)
        
        if not source_path.is_dir():
            print(f"Warning: {source_dir} is not a valid directory")
            continue
        
        # Find the algorithm label for this directory
        algo_name = source_path.name
        label = next((label for prefix, label in algo_labels if algo_name.startswith(prefix)), algo_name)
        
        all_prompts[label] = {}
        
        # Find all results folders (results_MODEL_SEED_PROMPT_NUMBER)
        for results_folder in sorted(source_path.glob("results_*")):
            if not results_folder.is_dir():
                continue
            
            # Extract prompt number from folder name
            parts = results_folder.name.split("_")
            if len(parts) >= 2:
                prompt_number = parts[-1]  # Last part is prompt number
            else:
                continue
            
            # Read the CSV file
            csv_path = results_folder / "fitness_results.csv"
            if not csv_path.exists():
                print(f"Warning: {csv_path} not found")
                continue
            
            try:
                df = pd.read_csv(csv_path)
                
                # Get the baseline prompt from the first row
                first_row = df.iloc[0]
                baseline_prompt = first_row['prompt']
                baseline_prompts[int(prompt_number)] = baseline_prompt
                
                # Get the best prompt (last row typically has the final best)
                last_row = df.iloc[-1]
                best_prompt = last_row['best_prompt']
                
                all_prompts[label][int(prompt_number)] = best_prompt
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")
                continue
    
    # Create output DataFrame
    # Find all unique prompt numbers across all methods
    all_prompt_numbers = set()
    for prompts_dict in all_prompts.values():
        all_prompt_numbers.update(prompts_dict.keys())
    
    all_prompt_numbers = sorted(all_prompt_numbers)
    
    # Build rows with baseline prompt as first column
    rows = []
    for prompt_num in all_prompt_numbers:
        row = {
            "Prompt #": prompt_num,
            "Initial Prompt": baseline_prompts.get(prompt_num, "")
        }
        for method_name in sorted(all_prompts.keys()):
            row[method_name] = all_prompts[method_name].get(prompt_num, "")
        rows.append(row)
    
    output_df = pd.DataFrame(rows)
    
    # Save to Excel
    output_path = os.path.join(save_folder, "best_prompts_comparison.xlsx")
    output_df.to_excel(output_path, index=False, sheet_name="Best Prompts")
    print(f"Best prompts table saved to {output_path}")
    
    return output_df
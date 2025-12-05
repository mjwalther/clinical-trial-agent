import json
import os
from pathlib import Path

def extract_trial_profiles(base_path):
    """
    Extracts trial profiles from rank folders for each patient.
    """
    base_path = Path(base_path)
    all_trial_profiles = {}
    
    # Loop through all patient folders
    for patient_folder in sorted(base_path.iterdir()):
        if not patient_folder.is_dir():
            continue
            
        patient_id = patient_folder.name
        print(f"\nProcessing patient: {patient_id}")
        
        # Find all rank folders (rank1_, rank2_, rank3_, etc.)
        rank_folders = sorted([f for f in patient_folder.iterdir() 
                              if f.is_dir() and f.name.startswith("rank")])
        
        if not rank_folders:
            print(f"  Warning: No rank folders found")
            continue
        
        patient_trials = []
        
        # Process each rank folder
        for rank_folder in rank_folders:
            rank_name = rank_folder.name
            print(f"  Processing {rank_name}")
            
            # Initialize trial data
            trial_data = {
                "patient_id": patient_id,
                "rank_folder": rank_name,
                "trial_info": None,
                "inclusion_criteria": [],
                "exclusion_criteria": []
            }
            
            # Path to 1trial folder
            trial_folder = rank_folder / "1trial"
            
            if not trial_folder.exists():
                print(f"    Warning: 1trial folder not found")
                patient_trials.append(trial_data)
                continue
            
            # Extract corpus.json (trial overview)
            corpus_path = trial_folder / "corpus" / "corpus.json"
            if corpus_path.exists():
                try:
                    with open(corpus_path, 'r', encoding='utf-8') as f:
                        corpus_data = json.load(f)
                        trial_data["trial_info"] = {
                            "trial_id": corpus_data.get("_id"),
                            "title": corpus_data.get("title"),
                            "brief_summary": corpus_data.get("metadata", {}).get("brief_summary"),
                            "phase": corpus_data.get("metadata", {}).get("phase"),
                            "drugs": corpus_data.get("metadata", {}).get("drugs_list", []),
                            "diseases": corpus_data.get("metadata", {}).get("diseases_list", []),
                            "enrollment": corpus_data.get("metadata", {}).get("enrollment")
                        }
                        print(f"    Found trial info: {trial_data['trial_info']['trial_id']}")
                except Exception as e:
                    print(f"    Warning: Failed to read corpus.json: {e}")
            else:
                print(f"    Warning: corpus.json not found")
            
            # Extract inclusion and exclusion criteria from minified_canon folder
            minified_canon_folder = trial_folder / "minified_canon"
            
            if minified_canon_folder.exists():
                # Find inclusion and exclusion files
                for file in minified_canon_folder.iterdir():
                    if not file.name.endswith('.json'):
                        continue
                    
                    try:
                        with open(file, 'r', encoding='utf-8') as f:
                            criteria_data = json.load(f)
                        
                        # Check if it's inclusion or exclusion based on filename
                        if 'inclusion' in file.name.lower():
                            trial_data["inclusion_criteria"] = criteria_data
                            print(f"    Found {len(criteria_data)} inclusion criteria")
                        elif 'exclusion' in file.name.lower():
                            trial_data["exclusion_criteria"] = criteria_data
                            print(f"    Found {len(criteria_data)} exclusion criteria")
                    
                    except Exception as e:
                        print(f"    Warning: Failed to read {file.name}: {e}")
            else:
                print(f"    Warning: minified_canon folder not found")
            
            patient_trials.append(trial_data)
        
        all_trial_profiles[patient_id] = patient_trials
        print(f"  Extracted {len(patient_trials)} trials for {patient_id}")
    
    return all_trial_profiles


def save_trial_profiles(all_profiles, output_base_dir):
    """
    Save trial profiles to individual JSON files organized by patient.
    """
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    for patient_id, trials in all_profiles.items():
        # Create patient subfolder
        patient_folder = output_base_dir / patient_id
        patient_folder.mkdir(parents=True, exist_ok=True)
        
        # Save each trial as a separate JSON file
        for idx, trial_data in enumerate(trials, 1):
            # Extract trial ID from rank folder name or use index
            rank_name = trial_data.get('rank_folder', f'rank{idx}')
            
            # Create filename based on rank
            filename = f"{rank_name}.json"
            output_path = patient_folder / filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(trial_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(trials)} trials for {patient_id}")
    
    print(f"\n✓ Saved all trial profiles to: {output_base_dir}")


def print_trial_summary(all_profiles):
    """Print a summary of extracted trial profiles."""
    print("TRIAL PROFILES SUMMARY")
    
    for patient_id, trials in all_profiles.items():
        print(f"\n{patient_id}: {len(trials)} trials")
        
        for trial in trials[:3]:  # Show first 3 trials
            rank = trial.get('rank_folder', 'unknown')
            trial_id = "N/A"
            if trial.get('trial_info'):
                trial_id = trial['trial_info'].get('trial_id', 'N/A')
            
            inclusion_count = len(trial.get('inclusion_criteria', []))
            exclusion_count = len(trial.get('exclusion_criteria', []))
            
            print(f"  {rank}: {trial_id}")
            print(f"    Inclusion: {inclusion_count}, Exclusion: {exclusion_count}")


if __name__ == "__main__":
    base_path = "/Users/jiro/Desktop/trial-matching-agent/dataset-conversation"
    
    # Extract trial profiles
    all_profiles = extract_trial_profiles(base_path)
    
    # Save trial profiles organized by patient
    output_dir = "/Users/jiro/Desktop/trial-matching-agent/trial_profiles"
    save_trial_profiles(all_profiles, output_dir)
    
    print_trial_summary(all_profiles)
    
    total_trials = sum(len(trials) for trials in all_profiles.values())
    print(f"Processing complete. Extracted {total_trials} total trial profiles across {len(all_profiles)} patients.")
import json
import os
from pathlib import Path
from collections import defaultdict

def extract_patient_profiles(base_path):
    """
    Extract patient profiles from canonical.jsonl files in the dataset.
    """
    base_path = Path(base_path)
    patient_profiles = []
    
    # Loop through all subdirectories in the base path
    for patient_folder in sorted(base_path.iterdir()):
        if not patient_folder.is_dir():
            continue
            
        patient_id = patient_folder.name
        print(f"Processing patient: {patient_id}")
        
        # Find the folder that starts with "rank1_"
        # Only need to extract patient profile once
        rank1_folders = [f for f in patient_folder.iterdir() 
                        if f.is_dir() and f.name.startswith("rank1_")]
        
        if not rank1_folders:
            print(f"  Warning: No folder starting with 'rank1_' found in {patient_folder}")
            continue
        
        rank1_folder = rank1_folders[0]
        
        patient_data = {
            "patient_id": patient_id,
            "patient_note": None,
            "conditions": []
        }
        
        # Patient note
        patient_note_path = rank1_folder / "0patient_note" / "patient_note.json"
        if patient_note_path.exists():
            try:
                with open(patient_note_path, 'r', encoding='utf-8') as note_file:
                    note_data = json.load(note_file)
                    patient_data["patient_note"] = {
                        "text": note_data.get("text", ""),
                        "note_id": note_data.get("_id", patient_id)
                    }
                    print(f"  Found patient note")
            except Exception as e:
                print(f"  Warning: Failed to read patient note: {e}")
        else:
            print(f"  Warning: patient_note.json not found")
        
        # canonical.jsonl file
        canonical_path = rank1_folder / "0patient_coded_results" / "canonical.jsonl"
        
        if not canonical_path.exists():
            print(f"  Warning: canonical.jsonl not found at {canonical_path}")
            # Still add the patient even if canonical.jsonl is missing
            print(f"  Extracted 0 conditions (patient note only)")
            patient_profiles.append(patient_data)
            continue
        
        with open(canonical_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())
                    
                    # Only process records where extracted_value is true
                    if record.get("extracted_value") != True:
                        continue
                    
                    # Extract relevant information in JSON format
                    condition_info = {
                        "conceptId": record.get("conceptId"),
                        "preferred_term": record.get("preferred_term"),
                        "fully_specified_name": record.get("fully_specified_name"),
                        "span_match": record.get("span_match"),
                        "entity_variable_name": record.get("entity_variable_name"),
                        "type": record.get("type"),
                        "template": record.get("template"),
                        "fact_id": record.get("fact_id"),
                        "start_time_hours": record.get("start_time_in_hours"),
                        "end_time_hours": record.get("end_time_in_hours")
                    }
                    
                    patient_data["conditions"].append(condition_info)
                
                except json.JSONDecodeError as e:
                    print(f"  Warning: Failed to parse line {line_num}: {e}")
                except Exception as e:
                    print(f"  Warning: Error processing line {line_num}: {e}")
        
        print(f"  Extracted {len(patient_data['conditions'])} conditions")
        patient_profiles.append(patient_data)
    
    return patient_profiles


def save_patient_profiles(profiles, output_dir):
    """Save each patient profile to a separate JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for patient_data in profiles:
        patient_id = patient_data['patient_id']
        
        # Extract the number after "sigir-"
        if 'sigir-' in patient_id:
            file_number = patient_id.split('sigir-')[1]
            filename = f"{file_number}.json"
        else:
            # Fallback if the naming pattern is different
            filename = f"{patient_id}.json"
        
        output_path = output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(patient_data, f, indent=2, ensure_ascii=False)
        
        print(f"  Saved: {filename}")
    
    print(f"\nSaved {len(profiles)} patient profiles to: {output_dir}")


def print_patient_summary(profiles):
    """Print a summary of all patients."""
    print("PATIENT PROFILES SUMMARY")
    
    for patient_data in profiles:
        patient_id = patient_data['patient_id']
        total_conditions = len(patient_data['conditions'])
        
        print(f"\n{patient_id}:")
        print(f"  Total conditions: {total_conditions}")
        
        # Count by template type
        template_counts = {}
        for condition in patient_data['conditions']:
            template = condition.get('template', 'unknown')
            template_counts[template] = template_counts.get(template, 0) + 1
        
        for template, count in sorted(template_counts.items()):
            print(f"    {template}: {count}")
        
        # Show a few example conditions
        if patient_data['conditions']:
            print(f"  Sample conditions:")
            for condition in patient_data['conditions'][:3]:
                term = condition.get('preferred_term', 'N/A')
                print(f"    - {term}")


if __name__ == "__main__":
    base_path = "/Users/jiro/Desktop/trial-matching-agent/dataset-conversation"
    
    print("Starting patient profile extraction...")
    profiles = extract_patient_profiles(base_path)
    
    # Save each patient to a separate JSON file in the same folder
    output_dir = "/Users/jiro/Desktop/trial-matching-agent/patient_profiles"
    save_patient_profiles(profiles, output_dir)
    
    print_patient_summary(profiles)
    
    print(f"Processing complete. Extracted profiles for {len(profiles)} patients.")
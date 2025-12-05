import json
import re
from pathlib import Path
from collections import defaultdict

def load_patient_profile(patient_id, patient_profiles_dir):
    """Load a patient's profile."""
    patient_profiles_dir = Path(patient_profiles_dir)
    
    # Extract number from patient_id (e.g., sigir-20141 -> 20141)
    if 'sigir-' in patient_id:
        file_number = patient_id.split('sigir-')[1]
        filename = f"{file_number}.json"
    else:
        filename = f"{patient_id}.json"
    
    profile_path = patient_profiles_dir / filename
    
    if not profile_path.exists():
        return None
    
    with open(profile_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_variable_name(variable_name):
    """
    Normalize variable names by removing temporal suffixes like _now, _inthehistory, etc.
    """
    normalized = variable_name.lower()
    
    # Remove any suffix starting with _inthe (e.g., _inthepast30days, _inthefuture, etc.)
    normalized = re.sub(r'_inthe[a-z0-9]+$', '', normalized)
    
    # Check for other basic suffixes
    other_suffixes = [
        '_now',
        '_currently',
        '_present',
        '_active'
    ]
    
    for suffix in other_suffixes:
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)]
            break
    
    return normalized


def build_patient_variable_set(patient_profile):
    """
    Build a set of normalized variable names from patient's conditions.
    Returns both the set and a mapping for detailed info.
    """
    variable_set = set()
    variable_details = {}
    
    for condition in patient_profile.get('conditions', []):
        entity_var = condition.get('entity_variable_name')
        if entity_var:
            normalized = normalize_variable_name(entity_var)
            variable_set.add(normalized)
            
            # Store details for reasoning
            if normalized not in variable_details:
                variable_details[normalized] = []
            variable_details[normalized].append({
                'original_name': entity_var,
                'preferred_term': condition.get('preferred_term'),
                'conceptId': condition.get('conceptId'),
                'fact_id': condition.get('fact_id')
            })
    
    return variable_set, variable_details


def check_trial_eligibility(patient_profile, trial_profile):
    """
    Check if a patient is eligible for a trial based on exclusion criteria only.
    Patient is eligible if they do NOT have any of the exclusion criteria.

    * Will implement inclusion criteria later on * 
    """
    patient_variables, patient_details = build_patient_variable_set(patient_profile)
    
    exclusion_criteria = trial_profile.get('exclusion_criteria', [])
    
    # Check exclusion criteria
    exclusion_violated = []
    exclusion_satisfied = []
    
    for criterion in exclusion_criteria:
        normalized_criterion = normalize_variable_name(criterion)
        
        if normalized_criterion in patient_variables:
            # Patient HAS the excluded condition --> violation
            exclusion_violated.append({
                'criterion': criterion,
                'normalized': normalized_criterion,
                'patient_has': True,
                'details': patient_details.get(normalized_criterion, [])
            })
        else:
            # Patient does not have the excluded condition
            exclusion_satisfied.append({
                'criterion': criterion,
                'normalized': normalized_criterion,
                'patient_has': False
            })
    
    # Determine eligibility based on exclusions
    no_exclusions_violated = len(exclusion_violated) == 0
    is_eligible = no_exclusions_violated
    
    reasoning = {
        'eligible': is_eligible,
        'exclusion_criteria': {
            'total': len(exclusion_criteria),
            'satisfied': len(exclusion_satisfied),
            'violated': len(exclusion_violated),
            'details': {
                'satisfied': exclusion_satisfied,
                'violated': exclusion_violated
            }
        }
    }
    
    # Summary message
    if is_eligible:
        reasoning['summary'] = f"Patient is ELIGIBLE. None of the {len(exclusion_criteria)} exclusion criteria violated."
    else:
        reasoning['summary'] = f"Patient is INELIGIBLE. {len(exclusion_violated)} exclusion criteria violated."
    
    return reasoning


def evaluate_patient_trials(patient_id, patient_profiles_dir, trial_profiles_dir):
    """
    Evaluate all trials for a specific patient.
    """
    # Load patient profile
    patient_profile = load_patient_profile(patient_id, patient_profiles_dir)
    
    if not patient_profile:
        print(f"Patient profile not found for {patient_id}")
        return None
    
    # Load all trial profiles for this patient
    trial_profiles_dir = Path(trial_profiles_dir)
    patient_trial_dir = trial_profiles_dir / patient_id
    
    if not patient_trial_dir.exists():
        print(f"Trial profiles not found for {patient_id}")
        return None
    
    results = {
        'patient_id': patient_id,
        'patient_summary': patient_profile.get('patient_note', {}).get('text', 'N/A'),
        'trials_evaluated': []
    }
    
    # Process each trial
    trial_files = sorted(patient_trial_dir.glob('*.json'))
    
    for trial_file in trial_files:
        with open(trial_file, 'r', encoding='utf-8') as f:
            trial_profile = json.load(f)
        
        # Check eligibility
        eligibility = check_trial_eligibility(patient_profile, trial_profile)
        
        # Build trial result
        trial_info = trial_profile.get('trial_info', {})
        trial_result = {
            'rank': trial_profile.get('rank_folder'),
            'trial_id': trial_info.get('trial_id'),
            'title': trial_info.get('title'),
            'phase': trial_info.get('phase'),
            'drugs': trial_info.get('drugs', []),
            'diseases': trial_info.get('diseases', []),
            'eligibility': eligibility
        }
        
        results['trials_evaluated'].append(trial_result)
    
    # Add summary statistics
    eligible_count = sum(1 for t in results['trials_evaluated'] if t['eligibility']['eligible'])
    results['summary'] = {
        'total_trials': len(results['trials_evaluated']),
        'eligible_trials': eligible_count,
        'ineligible_trials': len(results['trials_evaluated']) - eligible_count
    }
    
    return results


def evaluate_all_patients(patient_profiles_dir, trial_profiles_dir, output_dir):
    """
    Evaluate eligibility for all patients across all their trials.
    """
    patient_profiles_dir = Path(patient_profiles_dir)
    trial_profiles_dir = Path(trial_profiles_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all patient directories from trial_profiles
    patient_dirs = [d for d in trial_profiles_dir.iterdir() if d.is_dir()]
    
    all_results = []
    
    for patient_dir in sorted(patient_dirs):
        patient_id = patient_dir.name
        print(f"\nEvaluating trials for {patient_id}...")
        
        results = evaluate_patient_trials(patient_id, patient_profiles_dir, trial_profiles_dir)
        
        if results:
            all_results.append(results)
            
            # Save individual patient results
            output_file = output_dir / f"{patient_id}_eligibility.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"  Eligible for {results['summary']['eligible_trials']}/{results['summary']['total_trials']} trials")
    
    # Save combined results
    combined_output = output_dir / "all_patients_eligibility.json"
    with open(combined_output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    return all_results


def print_eligibility_summary(all_results):
    """Print a summary of eligibility results."""
    print("ELIGIBILITY SUMMARY")
    
    for result in all_results:
        patient_id = result['patient_id']
        summary = result['summary']
        
        print(f"\n{patient_id}:")
        print(f"  Eligible: {summary['eligible_trials']}/{summary['total_trials']} trials")
        
        # Show eligible trials
        eligible_trials = [t for t in result['trials_evaluated'] if t['eligibility']['eligible']]
        if eligible_trials:
            print(f"  Eligible trials:")
            for trial in eligible_trials:
                print(f"    - {trial['rank']}: {trial['trial_id']}")


if __name__ == "__main__":
    patient_profiles_dir = "/Users/jiro/Desktop/trial-matching-agent/patient_profiles"
    trial_profiles_dir = "/Users/jiro/Desktop/trial-matching-agent/trial_profiles"
    output_dir = "/Users/jiro/Desktop/trial-matching-agent/eligibility_results"
        
    # Evaluate all patients
    all_results = evaluate_all_patients(patient_profiles_dir, trial_profiles_dir, output_dir)
    print_eligibility_summary(all_results)
    
    print(f"Evaluation done. Results saved to: {output_dir}")
    print(f"  - Individual patient files: {len(all_results)} patients")
    print(f"  - Combined results: all_patients_eligibility.json")
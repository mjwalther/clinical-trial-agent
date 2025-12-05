import json
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory


class ClinicalTrialMatchingAgent:
    """
    Agent that matches patients to eligible clinical trials through thoughtful conversation.
    """
    
    def __init__(self, patient_profiles_dir: str, trial_profiles_dir: str, openai_api_key: str):
        self.patient_profiles_dir = Path(patient_profiles_dir)
        self.trial_profiles_dir = Path(trial_profiles_dir)
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.7,
            openai_api_key=openai_api_key
        )
        
        # Memory for conversation
        self.chat_history = InMemoryChatMessageHistory()
        
        # Current patient data
        self.current_patient_id = None
        self.current_patient_profile = None
        self.current_patient_name = None
        self.eligible_trials = []
        self.all_trials_with_reasoning = []
        
        # Recommended trial profile for detailed Q&A
        self.recommended_trial_profile = None
        
        # Initialize SQL database for preference matching ONLY
        self.db_path = "trialogue_preferences.db"
        self.init_preference_database()
    

    # SQL database method for preference-based trial narrowing (if >1 eligible trial is found)
    def init_preference_database(self):
        """
        SQLite database for storing user preferences and trial characteristics.
        """

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # User preferences table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                session_id TEXT,
                question_number INTEGER,
                question TEXT,
                answer TEXT,
                preference_type TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (session_id, question_number)
            )
        ''')
        
        # Trial characteristics table for eligible trials
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trial_characteristics (
                session_id TEXT,
                trial_id TEXT,
                trial_index INTEGER,
                title TEXT,
                phase TEXT,
                phase_numeric INTEGER,
                diseases TEXT,
                interventions TEXT,
                brief_summary TEXT,
                is_early_phase INTEGER,
                is_late_phase INTEGER,
                is_invasive INTEGER,
                PRIMARY KEY (session_id, trial_id)
            )
        ''')
        
        # Preference-trial scores table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS preference_scores (
                session_id TEXT,
                trial_id TEXT,
                preference_type TEXT,
                score REAL,
                reasoning TEXT,
                PRIMARY KEY (session_id, trial_id, preference_type)
            )
        ''')
        
        conn.commit()
        conn.close()
        
        print(f"SQL Preferences Database initialized at {self.db_path}")
    
    def store_user_preference(self, session_id: str, question_number: int, question: str, answer: str, preference_type: str):
        """
        Storing user preference answers in database.
        """

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO user_preferences 
            (session_id, question_number, question, answer, preference_type)
            VALUES (?, ?, ?, ?, ?)
        ''', (session_id, question_number, question, answer, preference_type))
        
        conn.commit()
        conn.close()
        
        print(f"Stored preference #{question_number} in SQL database")
    
    def store_trial_characteristics(self, session_id: str, eligible_trials: List[Dict]):
        """
        Store characteristics of eligible trials in database for preference matching.
        """

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clear existing trials for this session
        cursor.execute('DELETE FROM trial_characteristics WHERE session_id = ?', (session_id,))
        
        for i, trial_data in enumerate(eligible_trials):
            trial = trial_data.get('trial', {})
            trial_info = trial.get('trial_info', {})
            
            trial_id = trial_info.get('trial_id', f'trial_{i}')
            title = trial_info.get('title', 'Unknown')
            phase = trial_info.get('phase', 'Not listed')
            diseases = json.dumps(trial_info.get('diseases', []))
            interventions = json.dumps(trial_info.get('interventions', []))
            brief_summary = trial_info.get('brief_summary', '')
            
            # Classify trial characteristics
            phase_numeric = self._parse_phase_number(phase)
            is_early_phase = 1 if phase_numeric <= 2 else 0
            is_late_phase = 1 if phase_numeric >= 3 else 0
            is_invasive = self._is_invasive_trial(interventions, brief_summary)
            
            cursor.execute('''
                INSERT INTO trial_characteristics
                (session_id, trial_id, trial_index, title, phase, phase_numeric, diseases, 
                 interventions, brief_summary, is_early_phase, is_late_phase, is_invasive)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (session_id, trial_id, i, title, phase, phase_numeric, diseases, 
                  interventions, brief_summary, is_early_phase, is_late_phase, is_invasive))
        
        conn.commit()
        conn.close()
        
        print(f"Stored {len(eligible_trials)} trial characteristics in SQL database")
    
    def _parse_phase_number(self, phase: str) -> int:
        """
        Extract numeric phase from phase string.
        """
        if not phase or phase == 'Not listed' or phase == 'N/A':
            return 0
        phase_lower = phase.lower()
        if 'phase 1' in phase_lower or 'phase i' in phase_lower:
            return 1
        elif 'phase 2' in phase_lower or 'phase ii' in phase_lower:
            return 2
        elif 'phase 3' in phase_lower or 'phase iii' in phase_lower:
            return 3
        elif 'phase 4' in phase_lower or 'phase iv' in phase_lower:
            return 4
        return 0
    
    def _is_invasive_trial(self, interventions_json: str, summary: str) -> int:
        """
        Determine if trial involves invasive procedures.
        """
        invasive_keywords = ['surgery', 'surgical', 'invasive', 'injection', 'biopsy', 
                            'catheter', 'endoscopy', 'procedure', 'operation']
        
        try:
            interventions = json.loads(interventions_json)
            text_to_check = ' '.join(interventions).lower() + ' ' + summary.lower()
        except:
            text_to_check = summary.lower()
        
        for keyword in invasive_keywords:
            if keyword in text_to_check:
                return 1
        return 0
    
    def _classify_preference_type(self, question: str) -> str:
        """
        Classify what type of preference a question is asking about.
        """
        question_lower = question.lower()
        if 'phase' in question_lower or 'early' in question_lower or 'late' in question_lower:
            return 'phase'
        elif 'invasive' in question_lower or 'treatment approach' in question_lower:
            return 'invasiveness'
        elif 'matter' in question_lower or 'priority' in question_lower or 'important' in question_lower:
            return 'priority'
        return 'general'
    
    def narrow_trials_by_preferences_sql(self, eligible_trials: List[Dict], preference_qa: List[Dict], session_id: str = "default") -> Dict:
        """
        Using SQL queries to match user preferences with trial characteristics.
        """
        # Store trial characteristics in database
        self.store_trial_characteristics(session_id, eligible_trials)
        
        # Store user preferences in database
        for i, qa in enumerate(preference_qa, 1):
            preference_type = self._classify_preference_type(qa['question'])
            self.store_user_preference(session_id, i, qa['question'], qa['answer'], preference_type)
        
        # SQL QUERY: Match preferences to trial characteristics
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Query to get all preferences for this session
        cursor.execute('''
            SELECT question_number, question, answer, preference_type
            FROM user_preferences
            WHERE session_id = ?
            ORDER BY question_number
        ''', (session_id,))
        preferences = cursor.fetchall()
        
        # Analyze each preference against trials using SQL queries
        trial_scores = {}
        
        for pref in preferences:
            question_num, question, answer, pref_type = pref
            answer_lower = answer.lower()
            
            if pref_type == 'phase':
                # Find trials matching phase preference
                if 'early' in answer_lower or 'experimental' in answer_lower or 'cutting' in answer_lower:
                    query = '''
                        SELECT trial_id, trial_index, title, phase
                        FROM trial_characteristics
                        WHERE session_id = ? AND is_early_phase = 1
                    '''
                else:
                    query = '''
                        SELECT trial_id, trial_index, title, phase
                        FROM trial_characteristics
                        WHERE session_id = ? AND is_late_phase = 1
                    '''
                
                cursor.execute(query, (session_id,))
                matching_trials = cursor.fetchall()
                
                for trial in matching_trials:
                    trial_id = trial[0]
                    if trial_id not in trial_scores:
                        trial_scores[trial_id] = {'score': 0, 'reasons': [], 'title': trial[2]}
                    trial_scores[trial_id]['score'] += 10
                    trial_scores[trial_id]['reasons'].append(f"Matches phase preference (Phase: {trial[3]})")
            
            elif pref_type == 'invasiveness':
                # Find non-invasive trials if preferred
                if 'avoid' in answer_lower or 'non-invasive' in answer_lower or 'not invasive' in answer_lower:
                    query = '''
                        SELECT trial_id, trial_index, title
                        FROM trial_characteristics
                        WHERE session_id = ? AND is_invasive = 0
                    '''
                    cursor.execute(query, (session_id,))
                    matching_trials = cursor.fetchall()
                    
                    for trial in matching_trials:
                        trial_id = trial[0]
                        if trial_id not in trial_scores:
                            trial_scores[trial_id] = {'score': 0, 'reasons': [], 'title': trial[2]}
                        trial_scores[trial_id]['score'] += 15
                        trial_scores[trial_id]['reasons'].append("Non-invasive approach matches preference")
            
            elif pref_type == 'priority':
                # Score based on stated priorities
                if 'safety' in answer_lower:
                    # Prefer later phase trials for safety
                    query = '''
                        SELECT trial_id, trial_index, title, phase_numeric
                        FROM trial_characteristics
                        WHERE session_id = ?
                        ORDER BY phase_numeric DESC
                    '''
                    cursor.execute(query, (session_id,))
                    trials = cursor.fetchall()
                    
                    # Give higher scores to higher phase trials
                    for trial in trials:
                        trial_id = trial[0]
                        phase_num = trial[3]
                        if trial_id not in trial_scores:
                            trial_scores[trial_id] = {'score': 0, 'reasons': [], 'title': trial[2]}
                        if phase_num >= 3:
                            trial_scores[trial_id]['score'] += 8
                            trial_scores[trial_id]['reasons'].append("Later phase supports safety priority")
        
        conn.close()
        
        # Find trial with highest score
        if trial_scores:
            # Sort trials by score
            sorted_trials = sorted(trial_scores.items(), key=lambda x: x[1]['score'], reverse=True)
            best_trial_id = sorted_trials[0][0]
            best_score = sorted_trials[0][1]['score']
            
            # Build detailed scoring breakdown
            all_scores = []
            for trial_id, score_data in sorted_trials:
                all_scores.append({
                    'trial_id': trial_id,
                    'title': score_data['title'],
                    'score': score_data['score'],
                    'reasons': score_data['reasons']
                })
            
            # Find the best trial in eligible_trials
            best_trial_data = None
            for trial_data in eligible_trials:
                trial_id = trial_data.get('trial', {}).get('trial_info', {}).get('trial_id', '')
                if trial_id == best_trial_id:
                    best_trial_data = trial_data
                    break
            
            if best_trial_data:
                reasoning = f"SQL-based matching score: {best_score} points. " + " ".join(trial_scores[best_trial_id]['reasons'])
                print(f"SQL Preference Matching: Best trial = {best_trial_id} (Score: {best_score})")
                
                return {
                    'trial': best_trial_data,
                    'reasoning': reasoning,
                    'sql_scores': all_scores,  
                    'all_eligible_trials': eligible_trials 
                }
        
        return {
            'trial': eligible_trials[0],
            'reasoning': 'Based on your preferences, this trial appears to be a good match for you.',
            'sql_scores': [],
            'all_eligible_trials': eligible_trials
        }
    
    
    def generate_flexible_recommendation_message(self, recommendation_data: Dict) -> str:
        """
        Generate a flexible recommendation message that includes SQL scoring info
        and invites discussion of other eligible trials.
        Also stores complete trial profile for detailed Q&A.
        """
        trial_data = recommendation_data.get('trial', {})
        reasoning = recommendation_data.get('reasoning', '')
        sql_scores = recommendation_data.get('sql_scores', [])
        all_eligible = recommendation_data.get('all_eligible_trials', [])
        
        trial = trial_data.get('trial', {})
        trial_info = trial.get('trial_info', {})
        
        title = trial_info.get('title', 'Unknown Trial')
        trial_id = trial_info.get('trial_id', 'N/A')
        
        # Store complete trial profile for Q&A
        self.recommended_trial_profile = {
            'trial_data': trial_data,
            'trial': trial,
            'trial_info': trial_info,
            'inclusion_criteria': trial.get('inclusion_criteria', []),
            'exclusion_criteria': trial.get('exclusion_criteria', []),
            'eligibility_reasoning': trial_data.get('reasoning', {}),
            'sql_scores': sql_scores,
            'all_eligible_trials': all_eligible
        }
        
        # Start with recommendation
        message = f"Based on your preferences, I recommend: **{title}**\n\n"
        
        # Add trial details
        message += f"**Trial Details:**\n"
        message += f"• Trial ID: {trial_id}\n"
        message += f"• Phase: {trial_info.get('phase', 'Not listed')}\n"
        
        diseases = trial_info.get('diseases', [])
        if diseases:
            message += f"• Focus: {', '.join(diseases[:3])}\n"
        
        interventions = trial_info.get('interventions', [])
        if interventions:
            message += f"• Interventions: {', '.join(interventions[:3])}\n"
        
        message += f"\n**Learn More:**\n"
        message += f"Visit ClinicalTrials.gov and search for trial ID: **{trial_id}**\n\n"
        message += f"Direct link: https://clinicaltrials.gov/study/{trial_id}\n\n"
        
        return message
        
    def normalize_variable_name(self, variable_name: str) -> str:
        """
        Normalize variable names for comparison.
        """
        normalized = variable_name.lower().strip()
        
        # Remove any suffix starting with _inthe
        normalized = re.sub(r'_inthe[a-z0-9]+$', '', normalized)
        
        # Remove _now_in pattern (e.g., _now_in_years -> _in_years)
        normalized = re.sub(r'_now_in', '_in', normalized)
        
        # Check for other basic suffixes at the end
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
    
    def is_gender_criterion(self, criterion: str) -> bool:
        """
        Check if a criterion is a gender/sex criterion.
        """
        normalized = criterion.lower()
        return 'patient_sex_is_' in normalized or 'patient_gender_is_' in normalized
    
    def should_ignore_criterion(self, criterion: str) -> bool:
        """
        Check if a criterion should be ignored during eligibility checking.
        """
        normalized = criterion.lower()
        
        # Ignore age recorded in months or days since we have age in years
        if 'patient_age_value_recorded' in normalized and ('in_months' in normalized or 'in_days' in normalized):
            return True
        
        return False
    
    def get_mutually_exclusive_gender_criteria(self, criteria: List[str]) -> List[List[str]]:
        """
        Group gender criteria that are mutually exclusive (patient can only be one gender).
        Returns a list of gender criterion groups, where each group represents an OR condition.
        """
        gender_criteria = [c for c in criteria if self.is_gender_criterion(c)]
        
        if len(gender_criteria) <= 1:
            return []
        
        # All gender criteria form one mutually exclusive group
        return [gender_criteria] if gender_criteria else []
    
    def build_patient_variable_set(self, patient_profile: Dict) -> Tuple[Set[str], Dict[str, List[Dict]]]:
        """
        Build a set of normalized variable names from patient conditions (including demographics).
        Also return details for each variable.
        """
        variables = set()
        details = {}
        
        conditions = patient_profile.get('conditions', [])
        
        for condition in conditions:
            entity_var = condition.get('entity_variable_name')
            if entity_var:
                normalized = self.normalize_variable_name(entity_var)
                variables.add(normalized)
                
                if normalized not in details:
                    details[normalized] = []
                
                detail_entry = {
                    'preferred_term': condition.get('preferred_term'),
                    'conceptId': condition.get('conceptId'),
                    'span_match': condition.get('span_match')
                }
                
                # Include extracted value if present (for demographics like age, sex)
                if 'extracted_value' in condition:
                    detail_entry['extracted_value'] = condition.get('extracted_value')
                
                # Include type information for better formatting
                if 'type' in condition:
                    detail_entry['type'] = condition.get('type')
                
                details[normalized].append(detail_entry)
        
        return variables, details
    
    def load_patient_profile(self, patient_id: str) -> Optional[Dict]:
        """
        Load a patient's profile.
        """
        # Extract number from patient_id
        if 'sigir-' in patient_id:
            file_number = patient_id.split('sigir-')[1]
            filename = f"{file_number}.json"
        else:
            filename = f"{patient_id}.json"
        
        profile_path = self.patient_profiles_dir / filename
        
        if not profile_path.exists():
            return None
        
        with open(profile_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_trial_profiles(self, patient_id: str) -> List[Dict]:
        """
        Load all trial profiles for a patient.
        """
        trial_folder = self.trial_profiles_dir / patient_id
        
        if not trial_folder.exists():
            return []
        
        trials = []
        for trial_file in sorted(trial_folder.glob('*.json')):
            with open(trial_file, 'r', encoding='utf-8') as f:
                trial_data = json.load(f)
                trial_data['_file_name'] = trial_file.name
                trials.append(trial_data)
        
        return trials
    
    def format_criterion_name(self, criterion: str, details: List[Dict] = None) -> str:
        """
        Convert a criterion variable name to a human-readable format.
        """
        # Start with lowercase version
        readable = criterion.lower()
        
        # Remove common prefixes
        prefixes_to_remove = [
            'patient_has_',
            'patient_can_',
            'patients_',
            'patient_'
        ]
        
        for prefix in prefixes_to_remove:
            if readable.startswith(prefix):
                readable = readable[len(prefix):]
                break
        
        # Special handling for demographics with extracted values
        if details:
            for detail in details:
                if 'extracted_value' in detail:
                    value = detail['extracted_value']
                    var_type = detail.get('type')
                    
                    # For age
                    if 'age' in criterion.lower() and var_type == 'Int':
                        return f"age of {value} years"
                    
                    # For sex/gender
                    if 'sex' in criterion.lower() and var_type == 'Bool':
                        if 'female' in criterion.lower() and value:
                            return "female gender"
                        elif 'male' in criterion.lower() and value:
                            return "male gender"
        
        readable = readable.replace('_', ' ')
        
        # Handle common medical terminology patterns
        replacements = {
            'inthehistory': 'in the past',
            'inthe history': 'in the past',
            'in thehistory': 'in the past',
            'in the history': 'in the past',
            ' now': ' currently',
            ' hx': ' history',
            ' dx': ' diagnosis',
            ' tx': ' treatment',
            'undergone ': '',
            'underwent ': '',
            'diagnosis of ': '',
            'finding of ': '',
            'symptoms of ': '',
        }
        
        for old, new in replacements.items():
            readable = readable.replace(old, new)
        
        readable = ' '.join(readable.split())
        readable = readable[0].upper() + readable[1:] if readable else readable
        
        return readable
    
    def check_trial_eligibility(self, patient_profile: Dict, trial_profile: Dict) -> Dict:
        """
        Check if a patient is eligible for a trial based on BOTH inclusion and exclusion criteria.
        Patient must have all inclusion criteria and none of the exclusion criteria.
        Special case: Gender criteria are treated as OR (patient needs to match at least one).
        Some criteria are automatically ignored (e.g., age in months when we have age in years).
        """
        patient_variables, patient_details = self.build_patient_variable_set(patient_profile)
        
        inclusion_criteria = trial_profile.get('inclusion_criteria', [])
        exclusion_criteria = trial_profile.get('exclusion_criteria', [])
        
        # Filter out criteria that should be ignored
        inclusion_criteria = [c for c in inclusion_criteria if not self.should_ignore_criterion(c)]
        exclusion_criteria = [c for c in exclusion_criteria if not self.should_ignore_criterion(c)]
        
        # Identify mutually exclusive gender criteria in inclusion
        gender_groups = self.get_mutually_exclusive_gender_criteria(inclusion_criteria)
        gender_criteria_in_groups = set()
        for group in gender_groups:
            gender_criteria_in_groups.update(group)
        
        # Check inclusion criteria
        inclusion_met = []
        inclusion_missing = []
        
        for criterion in inclusion_criteria:
            # Skip gender criteria, handle them separately
            if criterion in gender_criteria_in_groups:
                continue
                
            normalized_criterion = self.normalize_variable_name(criterion)
            
            if normalized_criterion in patient_variables:
                # Patient has the required condition
                criterion_details = patient_details.get(normalized_criterion, [])
                inclusion_met.append({
                    'criterion': criterion,
                    'normalized': normalized_criterion,
                    'patient_has': True,
                    'details': criterion_details,
                    'readable_name': self.format_criterion_name(criterion, criterion_details)
                })
            else:
                # Patient does not have the required condition
                inclusion_missing.append({
                    'criterion': criterion,
                    'normalized': normalized_criterion,
                    'patient_has': False,
                    'readable_name': self.format_criterion_name(criterion, None)
                })
        
        # Handle gender criteria
        for gender_group in gender_groups:
            # Check if patient matches any of the gender criteria in this group
            matched_any = False
            matched_criterion = None
            
            for criterion in gender_group:
                normalized_criterion = self.normalize_variable_name(criterion)
                if normalized_criterion in patient_variables:
                    matched_any = True
                    matched_criterion = criterion
                    break
            
            if matched_any:
                # Patient matches at least one gender criterion
                criterion_details = patient_details.get(self.normalize_variable_name(matched_criterion), [])
                inclusion_met.append({
                    'criterion': matched_criterion,
                    'normalized': self.normalize_variable_name(matched_criterion),
                    'patient_has': True,
                    'details': criterion_details,
                    'readable_name': self.format_criterion_name(matched_criterion, criterion_details),
                    'is_gender_group': True,
                    'gender_group_options': gender_group
                })
            else:
                # Patient doesn't match any gender criterion in the group
                inclusion_missing.append({
                    'criterion': ' OR '.join(gender_group),
                    'normalized': 'gender_criterion_group',
                    'patient_has': False,
                    'readable_name': 'Gender: ' + ' OR '.join([self.format_criterion_name(c, None) for c in gender_group]),
                    'is_gender_group': True
                })
        
        # Check exclusion criteria
        exclusion_violated = []
        exclusion_satisfied = []
        
        for criterion in exclusion_criteria:
            normalized_criterion = self.normalize_variable_name(criterion)
            
            if normalized_criterion in patient_variables:
                # Patient has the excluded condition
                criterion_details = patient_details.get(normalized_criterion, [])
                exclusion_violated.append({
                    'criterion': criterion,
                    'normalized': normalized_criterion,
                    'patient_has': True,
                    'details': criterion_details,
                    'readable_name': self.format_criterion_name(criterion, criterion_details)
                })
            else:
                # Patient does not have the excluded condition
                exclusion_satisfied.append({
                    'criterion': criterion,
                    'normalized': normalized_criterion,
                    'patient_has': False,
                    'readable_name': self.format_criterion_name(criterion, None)
                })
        
        # Determine eligibility: must meet all inclusions and no exclusions
        all_inclusions_met = len(inclusion_missing) == 0
        no_exclusions_violated = len(exclusion_violated) == 0
        is_eligible = all_inclusions_met and no_exclusions_violated
        
        reasoning = {
            'eligible': is_eligible,
            'inclusion_criteria': {
                'total': len(inclusion_criteria),
                'met': len(inclusion_met),
                'missing': len(inclusion_missing),
                'details': {
                    'met': inclusion_met,
                    'missing': inclusion_missing
                }
            },
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
        
        return reasoning
    
    def analyze_all_trials(self, patient_id: str) -> List[Dict]:
        """
        Analyze patient against all trials and return with eligibility reasoning.
        """
        patient_profile = self.load_patient_profile(patient_id)
        if not patient_profile:
            return []
        
        trials = self.load_trial_profiles(patient_id)
        all_trials = []
        
        for trial in trials:
            eligibility = self.check_trial_eligibility(patient_profile, trial)
            all_trials.append({
                'trial': trial,
                'reasoning': eligibility,
                'eligible': eligibility['eligible']
            })
        
        return all_trials
    
    def generate_patient_intro(self, patient_profile: Dict) -> str:
        """
        Generate a first-person introduction from the patient profile - only what patient would naturally disclose.
        """
        patient_note = patient_profile.get('patient_note', {})
        note_text = patient_note.get('text', '')
        
        # LLM to convert to first person, focusing on presenting complaint/symptoms only
        prompt = f"""Convert the following medical note into a brief, natural first-person introduction for a patient seeking clinical trial matches.

IMPORTANT CONTEXT: The patient ({self.current_patient_name}) is talking to an AI assistant that helps match patients with clinical trials. They are NOT talking to a doctor or healthcare provider.

The patient should:
- Introduce themselves briefly by name
- Mention they're looking for clinical trial opportunities
- Briefly describe their main condition or symptoms (2-3 sentences)
- Keep it conversational and concise

Do NOT include: age, gender, detailed medical history, test results, specific diagnoses, procedures, medications, or other detailed information. Just the basics of what brought them to seek trials.

Medical Note: {note_text}

First-Person Introduction:"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def ask_for_additional_info(self, patient_profile: Dict, initial_intro: str) -> str:
        """
        Generate agent's request for additional information not disclosed in intro.
        """
        patient_note = patient_profile.get('patient_note', {})
        note_text = patient_note.get('text', '')
        
        prompt = f"""The patient said: "{initial_intro}"

Based on the complete medical record below, identify what KEY information the patient did NOT mention that would be important for clinical trial matching.

Complete Medical Record: {note_text}

Generate a warm, conversational request asking the patient to provide the missing information needed for trial matching. Ask about:
- Age and gender (if not mentioned)
- Relevant medical history
- Current medications
- Other conditions or diagnoses
- Any procedures or treatments

Keep it friendly and conversational. Make it clear you're an AI assistant helping them find clinical trials. Start with something like "Thank you for sharing that. To find the best clinical trial matches for you, I'll need to know a bit more about your medical background..."

Agent's Request:"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def generate_complete_patient_response(self, patient_profile: Dict) -> str:
        """
        Generate patient's complete response with all missing information.
        """
        patient_note = patient_profile.get('patient_note', {})
        note_text = patient_note.get('text', '')
        
        prompt = f"""Based on this complete medical record, generate a first-person response from the patient ({self.current_patient_name}) providing all their relevant medical information for clinical trial matching.

IMPORTANT: 
- The patient is responding to a request for more information to help match them with clinical trials
- Do NOT restate their name (they already introduced themselves)
- Keep it conversational but comprehensive
- Include: age, gender, medical history, current conditions, medications, relevant procedures/treatments

Make it sound like someone filling out their medical background for a trial matching service, not talking to a doctor.

Medical Record: {note_text}

Patient's Response:"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def generate_preference_questions(self, eligible_trials: List[Dict], question_number: int = 1, previous_qa: List[Dict] = None) -> Dict:
        """
        Generate one preference question at a time for conversational flow.
        
        Args:
            eligible_trials: List of eligible trial data
            question_number: Which question to ask (1, 2, or 3)
            previous_qa: List of previous questions and answers [{'question': '...', 'answer': '...'}]
        
        Returns:
            Dict with 'question', 'is_final' flag
        """
        if previous_qa is None:
            previous_qa = []
        
        # Extract unique characteristics from eligible trials
        phases = set()
        diseases = set()
        interventions = set()
        
        for trial_data in eligible_trials:
            trial = trial_data.get('trial', {})
            trial_info = trial.get('trial_info', {})
            
            phase = trial_info.get('phase', '')
            if phase and phase != 'N/A':
                phases.add(phase)
            
            trial_diseases = trial_info.get('diseases', [])
            diseases.update(trial_diseases)
            
            trial_interventions = trial_info.get('interventions', [])
            interventions.update(trial_interventions)
        
        context = f"""
Available trial characteristics:
- Phases: {', '.join(sorted(phases)) if phases else 'Not specified'}
- Focus areas/diseases: {', '.join(sorted(diseases)[:5]) if diseases else 'Various'}
- Number of trials: {len(eligible_trials)}
"""
        
        # Build previous Q&A context
        previous_context = ""
        if previous_qa:
            previous_context = "\n\nPrevious questions and answers:\n"
            for i, qa in enumerate(previous_qa, 1):
                previous_context += f"Q{i}: {qa['question']}\nA{i}: {qa['answer']}\n"
        
        # Determine which question to ask
        if question_number == 1:
            prompt = f"""You are helping a patient narrow down {len(eligible_trials)} eligible clinical trials.

{context}

Generate ONE conversational question to understand the patient's preference regarding trial phase (early vs. later phase trials).

Make it warm, empathetic, and focused on helping them understand the choice. Keep it to 2-3 sentences maximum.

Your question:"""
        
        elif question_number == 2:
            prompt = f"""You are helping a patient narrow down {len(eligible_trials)} eligible clinical trials.

{context}{previous_context}

Based on their previous answer, generate ONE follow-up question about treatment approaches or specific aspects they're interested in or want to avoid.

Make it warm, conversational, and build on their previous response. Keep it to 2-3 sentences maximum.

Your question:"""
        
        elif question_number == 3:
            prompt = f"""You are helping a patient narrow down {len(eligible_trials)} eligible clinical trials.

{context}{previous_context}

Based on their previous answers, generate ONE final question to understand what matters most to them (innovation, safety, convenience, duration, etc.).

Make it warm, conversational, and help them prioritize. Keep it to 2-3 sentences maximum.

Your question:"""
        
        else:
            # Done asking questions
            return {
                'question': None,
                'is_final': True
            }
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        return {
            'question': response.content,
            'is_final': question_number >= 3
        }
    
    def context_aware_chat(self, user_message: str, conversation_context: Dict) -> str:
        """
        Handle user questions intelligently based on conversation context.
        Provides accurate information about trials, treatments, diseases, SQL scores, etc.
        """
        # Build context based on conversation state
        context_info = []
        
        # Add current patient info if available
        if self.current_patient_profile:
            patient_summary = self.extract_key_patient_info(self.current_patient_profile)
            context_info.append(f"Patient Information:\n{patient_summary}")
        
        # Add trial information based on what stage we're at
        current_state = conversation_context.get('state', 'unknown')
        
        if current_state == 'reviewing_trials':
            # Currently reviewing trials - provide info about current trial
            current_trial_index = conversation_context.get('current_trial_index', 0)
            all_trials = conversation_context.get('all_trials', [])
            
            if current_trial_index < len(all_trials):
                trial = all_trials[current_trial_index]
                trial_info = trial.get('trial', {}).get('trial_info', {})
                context_info.append(f"""
Current Trial Being Reviewed:
- Title: {trial_info.get('title', 'N/A')}
- Trial ID: {trial_info.get('trial_id', 'N/A')}
- Phase: {trial_info.get('phase', 'Not listed')}
- Focus Areas: {', '.join(trial_info.get('diseases', []))}
- Brief Summary: {trial_info.get('brief_summary', 'N/A')}
- Eligibility: {'ELIGIBLE' if trial.get('eligible') else 'NOT ELIGIBLE'}
- Explanation: {trial.get('explanation', 'N/A')}
""")
        
        elif current_state in ['post_recommendation', 'gathering_preferences']:
            # User is in preference/recommendation phase - provide comprehensive info
            
            # If we have a stored recommended trial profile, use it for detailed Q&A
            if self.recommended_trial_profile:
                profile = self.recommended_trial_profile
                trial_info = profile['trial_info']
                
                context_info.append(f"""
Recommended Trial (Complete Profile):
- Title: {trial_info.get('title', 'N/A')}
- Trial ID: {trial_info.get('trial_id', 'N/A')}
- Phase: {trial_info.get('phase', 'Not listed')}
- Focus Areas: {', '.join(trial_info.get('diseases', []))}
- Interventions: {', '.join(trial_info.get('interventions', []))}
- Brief Summary: {trial_info.get('brief_summary', 'N/A')}
- Detailed Summary: {trial_info.get('detailed_description', 'N/A')}

Inclusion Criteria:
{chr(10).join(['- ' + str(c) for c in profile['inclusion_criteria'][:10]])}

Exclusion Criteria:
{chr(10).join(['- ' + str(c) for c in profile['exclusion_criteria'][:10]])}

Eligibility Reasoning:
- Patient meets {profile['eligibility_reasoning'].get('inclusion_criteria', {}).get('met', 0)} of {profile['eligibility_reasoning'].get('inclusion_criteria', {}).get('total', 0)} inclusion criteria
- Patient violates {profile['eligibility_reasoning'].get('exclusion_criteria', {}).get('violated', 0)} of {profile['eligibility_reasoning'].get('exclusion_criteria', {}).get('total', 0)} exclusion criteria
""")
            
            # Also include SQL scores if available
            sql_scores = conversation_context.get('sql_scores', [])
            if sql_scores:
                scores_text = "\nSQL Preference Matching Scores:\n"
                for i, score_info in enumerate(sql_scores, 1):
                    scores_text += f"{i}. {score_info['title'][:60]} - {score_info['score']} points\n"
                    if score_info['reasons']:
                        scores_text += f"   Reasons: {', '.join(score_info['reasons'])}\n"
                context_info.append(scores_text)
            
            # Include all eligible trials for comparison
            eligible_trials = conversation_context.get('eligible_trials', [])
            if eligible_trials and len(eligible_trials) > 1:
                trials_summary = "\nAll Eligible Trials for Comparison:\n"
                for i, trial_data in enumerate(eligible_trials, 1):
                    trial = trial_data.get('trial', {})
                    trial_info = trial.get('trial_info', {})
                    trials_summary += f"""
{i}. {trial_info.get('title', 'N/A')}
   - Trial ID: {trial_info.get('trial_id', 'N/A')}
   - Phase: {trial_info.get('phase', 'Not listed')}
   - Focus: {', '.join(trial_info.get('diseases', []))}
"""
                context_info.append(trials_summary)
        
        elif current_state == 'interactive':
            # Post-recommendation - user may have questions about final trial or alternatives
            recommended_trial = conversation_context.get('recommended_trial')
            if recommended_trial:
                trial = recommended_trial.get('trial', {})
                trial_info = trial.get('trial_info', {})
                context_info.append(f"""
Recommended Trial:
- Title: {trial_info.get('title', 'N/A')}
- Trial ID: {trial_info.get('trial_id', 'N/A')}
- Phase: {trial_info.get('phase', 'Not listed')}
- Focus Areas: {', '.join(trial_info.get('diseases', []))}
- Brief Summary: {trial_info.get('brief_summary', 'N/A')}
""")
            
            # Also include other eligible trials for comparison
            eligible_trials = conversation_context.get('eligible_trials', [])
            if eligible_trials and len(eligible_trials) > 1:
                context_info.append(f"\nYou also have {len(eligible_trials) - 1} other eligible trial(s) available.")
        
        # Build the full context
        full_context = "\n\n".join(context_info) if context_info else "No specific trial context available yet."
        
        # Create prompt for LLM
        prompt = f"""You are Trialogue, a compassionate clinical trial matching agent helping a patient explore their clinical trial options.

CONTEXT:
{full_context}

PATIENT QUESTION:
{user_message}

INSTRUCTIONS:
- Answer the patient's question accurately using the context provided
- For questions about inclusion/exclusion criteria, list them clearly and explain why the patient meets or doesn't meet them
- For questions about treatments/interventions, describe them from the trial information
- For questions about trial details (phase, summary, diseases), provide comprehensive information
- For questions comparing trials or asking about SQL scores, explain the scoring rationale
- If asked "why am I eligible?", explain which inclusion criteria they meet and which exclusions they don't violate
- If the information isn't in the context, say so honestly and offer to help with what you do know
- Be empathetic, supportive, and clear
- Keep responses concise but informative (2-4 paragraphs for simple questions, more detail for complex ones)
- Format lists with bullet points for readability
- After answering, ask if they have any other questions about the trial or would like to proceed

Your response:"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def extract_key_patient_info(self, patient_profile: Dict) -> str:
        """
        Extract key patient information for confirmation.
        """
        patient_note = patient_profile.get('patient_note', {})
        note_text = patient_note.get('text', '')
        
        prompt = f"""Based on this patient information, extract and summarize the key medical details in a bulleted list format. 
Include: age, gender, main symptoms, medical history, and any relevant conditions.

Patient Information: {note_text}

Key Details (in bullet points):"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def format_criterion_naturally(self, readable_name: str, is_met: bool = True) -> str:
        """Format a criterion in natural language."""
        lower_name = readable_name.lower()
        
        # Handle age and gender specially
        if 'age' in lower_name and 'year' in lower_name:
            return "You meet the age requirement"
        elif lower_name in ['male gender', 'female gender', 'gender', 'sex']:
            return "You meet the gender requirement"
        elif 'male' in lower_name or 'female' in lower_name:
            return "You meet the gender requirement"
        
        # Handle temporal words (currently, in the past, etc.)
        if is_met:
            if 'currently' in lower_name:
                # "Acute infectious disease currently" -> "You currently have acute infectious disease"
                condition = lower_name.replace(' currently', '').strip()
                return f"You currently have {condition}"
            elif 'in the past' in lower_name or 'history of' in lower_name:
                condition = lower_name.replace(' in the past', '').replace('history of ', '').strip()
                return f"You've had {condition}"
            else:
                # Default case
                return f"You have {readable_name}"
        else:
            # For missing criteria
            article = "an" if readable_name[0].lower() in ['a', 'e', 'i', 'o', 'u'] else "a"
            if 'currently' in lower_name:
                condition = lower_name.replace(' currently', '').strip()
                return f"You don't currently have {article} {condition}"
            elif 'in the past' in lower_name:
                condition = lower_name.replace(' in the past', '').strip()
                return f"You haven't had {article} {condition}"
            else:
                return f"You don't have {article} {readable_name}"
    
    def generate_detailed_eligibility_explanation(self, reasoning: Dict) -> str:
        """
        Generate detailed and empathetic explanation of eligibility with specific reasons.
        """
        inclusion = reasoning.get('inclusion_criteria', {})
        exclusion = reasoning.get('exclusion_criteria', {})
        eligible = reasoning.get('eligible', False)
        
        if eligible:
            # Patient is eligible - explain why
            explanation_parts = []
            
            # Mention inclusion criteria met
            met_criteria = inclusion['details']['met']
            if met_criteria:
                # Separate age/gender from other criteria
                age_gender_criteria = []
                other_criteria = []
                
                for criterion in met_criteria:
                    readable_name = criterion.get('readable_name', criterion['criterion']).lower()
                    if any(word in readable_name for word in ['age', 'gender', 'male', 'female', 'sex']):
                        age_gender_criteria.append(criterion)
                    else:
                        other_criteria.append(criterion)
                
                # Format based on what criteria exist
                if age_gender_criteria and other_criteria:
                    explanation_parts.append(f"You meet all {len(met_criteria)} required inclusion criteria:")
                    explanation_parts.append(f"  • You meet the age and gender requirements")
                    for i, criterion in enumerate(other_criteria, 2):
                        readable_name = criterion.get('readable_name', criterion['criterion'])
                        formatted = self.format_criterion_naturally(readable_name, is_met=True)
                        explanation_parts.append(f"  • {formatted}")
                elif age_gender_criteria:
                    explanation_parts.append("You meet the age and gender requirements for this trial.")
                else:
                    explanation_parts.append(f"You meet all {len(met_criteria)} required inclusion criteria:")
                    for criterion in other_criteria:
                        readable_name = criterion.get('readable_name', criterion['criterion'])
                        formatted = self.format_criterion_naturally(readable_name, is_met=True)
                        explanation_parts.append(f"  • {formatted}")
            
            # Mention no exclusions violated
            if exclusion['total'] > 0:
                explanation_parts.append(f"\nAdditionally, you don't have any of the {exclusion['total']} exclusion conditions that would disqualify you from this trial.")
            
            return "\n".join(explanation_parts)
        
        else:
            # Patient is NOT eligible - provide detailed reasons
            explanation_parts = ["Unfortunately, you don't qualify for this trial. Here's why:\n"]
            
            reasons_listed = 0
            max_reasons = 5
            total_reasons = inclusion.get('missing', 0) + exclusion.get('violated', 0)
            
            # List missing inclusion criteria
            missing_criteria = inclusion['details']['missing']
            if missing_criteria:
                explanation_parts.append("**Missing Required Criteria:**")
                for i, criterion in enumerate(missing_criteria, 1):
                    if reasons_listed >= max_reasons:
                        break
                    readable_name = criterion.get('readable_name', criterion['criterion'])
                    formatted = self.format_criterion_naturally(readable_name, is_met=False)
                    explanation_parts.append(f"  {i}. {formatted}")
                    reasons_listed += 1
            
            # List violated exclusion criteria
            violated_criteria = exclusion['details']['violated']
            if violated_criteria and reasons_listed < max_reasons:
                explanation_parts.append("\n**Exclusion Criteria Violated:**")
                for i, criterion in enumerate(violated_criteria, 1):
                    if reasons_listed >= max_reasons:
                        break
                    readable_name = criterion.get('readable_name', criterion['criterion'])
                    
                    # Add article (a/an) based on first letter
                    article = "an" if readable_name[0].lower() in ['a', 'e', 'i', 'o', 'u'] else "a"
                    
                    # Format naturally
                    if readable_name.lower().startswith(('age', 'gender', 'sex', 'male', 'female')):
                        explanation_parts.append(f"  {i}. You have {readable_name} (which is an exclusion)")
                    else:
                        explanation_parts.append(f"  {i}. You have {article} {readable_name} (which is an exclusion)")
                    reasons_listed += 1
            
            # Add note if there are more reasons than shown
            if total_reasons > max_reasons:
                remaining = total_reasons - max_reasons
                explanation_parts.append(f"\n*Note: There are {remaining} additional reason(s) not listed here for brevity*")            
            return "\n".join(explanation_parts)
    
    def generate_trial_explanation(self, trial_data: Dict, reasoning: Dict) -> str:
        """
        Generate a natural language explanation of eligibility for a trial.
        """
        trial_info = trial_data['trial'].get('trial_info', {})
        title = trial_info.get('title', 'Unknown Title')
        brief_summary = trial_info.get('brief_summary', 'No summary available')[:150] + "..."
        
        eligible = reasoning['eligible']
        status = "✓ ELIGIBLE" if eligible else "✗ NOT ELIGIBLE"
        
        # Generate detailed explanation
        detailed_explanation = self.generate_detailed_eligibility_explanation(reasoning)
        
        return f"""**{title}**
{brief_summary}

**Status: {status}**

{detailed_explanation}"""
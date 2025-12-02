from flask import Flask, request, jsonify, send_file
import os
from agent import ClinicalTrialMatchingAgent

app = Flask(__name__)

# Initialize your existing agent
PATIENT_PROFILES_DIR = "patient_profiles"
TRIAL_PROFILES_DIR = "trial_profiles"
openai_api_key = os.getenv("OPENAI_API_KEY")

agent = ClinicalTrialMatchingAgent(
    patient_profiles_dir=PATIENT_PROFILES_DIR,
    trial_profiles_dir=TRIAL_PROFILES_DIR,
    openai_api_key=openai_api_key
)

# Patient names mapping
PATIENT_NAMES = {
    'sigir-20141': 'Alex Rivera',
    'sigir-20142': 'Jordan Chen',
    'sigir-20148': 'Taylor Patel',
    'sigir-201417': 'Casey Johnson',
    'sigir-201419': 'Morgan Kim',
    'sigir-201513': 'Riley Thompson',
    'sigir-201520': 'Avery Washington',
    'sigir-201524': 'Quinn Martinez'
}

@app.route('/')
def index():
    # Check if file exists and print debug info
    import os
    cwd = os.getcwd()
    print(f"Current directory: {cwd}")
    
    html_file = 'index.html'
    if os.path.exists(html_file):
        print(f"Found {html_file}")
        return send_file(html_file)
    else:
        print(f"ERROR: {html_file} not found in {cwd}")
        print(f"Files in directory: {os.listdir('.')}")
        return f"Error: {html_file} not found. Please make sure it's in the same directory as simple_server.py", 404

@app.route('/start', methods=['POST'])
def start_conversation():
    """Start conversation with selected patient"""
    data = request.json
    patient_id = data['patient_id']
    
    print(f"Loading patient: {patient_id}")
    
    # Load patient using your existing agent
    agent.current_patient_id = patient_id
    agent.current_patient_profile = agent.load_patient_profile(patient_id)
    
    if agent.current_patient_profile is None:
        print(f"ERROR: Could not load patient profile for {patient_id}")
        print(f"Looking in directory: {PATIENT_PROFILES_DIR}")
        import os
        if os.path.exists(PATIENT_PROFILES_DIR):
            files = os.listdir(PATIENT_PROFILES_DIR)
            print(f"Available files: {files}")
        else:
            print(f"ERROR: Directory {PATIENT_PROFILES_DIR} does not exist!")
        return jsonify({'error': f'Could not load patient profile for {patient_id}'}), 404
    
    agent.current_patient_name = PATIENT_NAMES.get(patient_id, 'Patient')
    
    # Return greeting immediately
    agent_greeting = "Hi! I'm here to help you explore clinical trial options that might be right for you. I know navigating clinical trials can feel overwhelming, but I'm here to make this process easier."
    
    return jsonify({
        'agent_greeting': agent_greeting
    })

@app.route('/generate-intro', methods=['POST'])
def generate_intro():
    """Generate patient intro and ask for additional info"""
    # Generate brief patient intro (symptoms only)
    patient_intro = agent.generate_patient_intro(agent.current_patient_profile)
    
    # Agent asks for additional info
    agent_ask = agent.ask_for_additional_info(agent.current_patient_profile, patient_intro)
    
    # Patient provides complete info
    patient_complete = agent.generate_complete_patient_response(agent.current_patient_profile)
    
    # Agent confirms understanding
    key_info = agent.extract_key_patient_info(agent.current_patient_profile)
    confirmation_prompt = f"Thank you for sharing that with me. Let me make sure I've understood your information correctly:\n\n{key_info}\n\nDoes this look accurate?"
    
    return jsonify({
        'patient_intro': patient_intro,
        'agent_ask': agent_ask,
        'patient_complete': patient_complete,
        'confirmation_prompt': confirmation_prompt
    })

@app.route('/analyze', methods=['POST'])
def analyze_trials():
    """Analyze all trials using your existing agent"""
    data = request.json
    patient_id = data['patient_id']
    
    # Use your existing analyze method
    all_trials = agent.analyze_all_trials(patient_id)
    
    # Format trials with explanations and additional details
    formatted_trials = []
    for trial_data in all_trials:
        trial = trial_data['trial']
        reasoning = trial_data['reasoning']
        explanation = agent.generate_detailed_eligibility_explanation(reasoning)
        
        # Extract additional trial information
        trial_info = trial.get('trial_info', {})
        
        formatted_trials.append({
            'trial': trial,
            'eligible': trial_data['eligible'],
            'explanation': explanation,
            'title': trial_info.get('title', 'N/A'),
            'brief_summary': trial_info.get('brief_summary', 'N/A'),
            'phase': trial_info.get('phase', 'N/A'),
            'diseases': trial_info.get('diseases', [])
        })
    
    return jsonify({'trials': formatted_trials})

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages with full conversation context"""
    data = request.json
    message = data.get('message', '')
    conversation_context = data.get('context', {})
    
    # Use context-aware chat method
    response = agent.context_aware_chat(message, conversation_context)
    
    return jsonify({'response': response})

@app.route('/get-preference-questions', methods=['POST'])
def get_preference_questions():
    """Generate one preference question at a time"""
    data = request.json
    eligible_trials = data.get('eligible_trials', [])
    question_number = data.get('question_number', 1)
    previous_qa = data.get('previous_qa', [])
    
    if len(eligible_trials) <= 1:
        return jsonify({'question': None, 'is_final': True})
    
    result = agent.generate_preference_questions(eligible_trials, question_number, previous_qa)
    
    return jsonify(result)

@app.route('/narrow-trials', methods=['POST'])
def narrow_trials():
    """Narrow down trials based on all collected preferences"""
    data = request.json
    eligible_trials = data.get('eligible_trials', [])
    preference_qa = data.get('preference_qa', [])
    
    recommendation = agent.narrow_trials_by_preferences(eligible_trials, preference_qa)
    
    # Generate the final recommendation message
    message = agent.generate_trial_recommendation_message(
        recommendation['trial'],
        recommendation['reasoning']
    )
    
    return jsonify({
        'recommended_trial': recommendation['trial'],
        'message': message
    })

@app.route('/check-conversation-end', methods=['POST'])
def check_conversation_end():
    """Check if user is satisfied and ready to end conversation"""
    data = request.json
    user_message = data.get('message', '')
    
    is_ending = agent.detect_conversation_end(user_message)
    
    if is_ending:
        outro = agent.generate_outro_message()
        return jsonify({'is_ending': True, 'outro': outro})
    else:
        return jsonify({'is_ending': False})

@app.route('/continue-narrowing', methods=['POST'])
def continue_narrowing():
    """Continue narrowing trials based on user feedback"""
    data = request.json
    eligible_trials = data.get('eligible_trials', [])
    user_feedback = data.get('feedback', '')
    
    follow_up = agent.continue_trial_narrowing(eligible_trials, user_feedback)
    
    return jsonify({'follow_up': follow_up})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
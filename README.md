# Trialogue - Clinical Trial Matching Agent

Trialogue is a conversational AI agent that helps patients find clinical trials they're eligible for through natural dialogue. It combines rule-based eligibility verification with SQL-powered preference matching and LLM-driven conversation.

## Features

- **Automated Eligibility Checking**: Rule-based Boolean logic matches patient profiles against trial inclusion/exclusion criteria
- **Personalized Recommendations**: SQL-based preference matching narrows eligible trials based on patient preferences
- **Conversational Interface**: Natural language interaction powered by GPT-4o
- **Patient-Centered Design**: Empathetic, respectful tone throughout the conversation

## Prerequisites

- **Python 3.8 or higher**
- **OpenAI API Key** ([platform.openai.com](https://platform.openai.com/api-keys))
- Patient profile data in `patient_profiles/` directory
- Trial profile data in `trial_profiles/` directory

## Installation

### 1. Clone or Download the Repository

```bash
git clone https://github.com/mjwalther/clinical-trial-agent
cd trial-matching-agent
```

### 2. Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install flask flask-cors langchain-openai langchain-core openai
```

### 3. Set Up OpenAI API Key

Create a `.env` file in the project root directory:

```bash
# Copy the example file
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=your-api-key-here
```

**Alternative:** Set the API key as an environment variable:

**macOS/Linux:**

```bash
export OPENAI_API_KEY='your-api-key-here'
```

**Windows (Command Prompt):**

```cmd
set OPENAI_API_KEY=your-api-key-here
```

**Windows (PowerShell):**

```powershell
$env:OPENAI_API_KEY='your-api-key-here'
```

### 4. Verify Directory Structure

Ensure your project has the following structure:

```
trial-matching-agent/
├── server.py
├── agent.py
├── index.html
├── requirements.txt
├── .env
├── patient_profiles/
│   ├── 20141.json
│   ├── 20142.json
│   └── ...
└── trial_profiles/
    ├── sigir-20141/
    │   ├── trial1.json
    │   └── trial2.json
    └── ...
```

## Running the Application

### Start the Server

```bash
python server.py
```

You should see output like:

```
======================================================================
Starting Trialogue Server
======================================================================
✓ Eligibility Matching: Rule-based Boolean logic (original)
✓ Preference Matching: SQL-powered queries (NEW!)
✓ Preference Database: trialogue_preferences.db
✓ Patient profiles: patient_profiles
✓ Trial profiles: trial_profiles
======================================================================
 * Running on http://127.0.0.1:5000
```

### Access the Application

Open your web browser and navigate to:

```
http://127.0.0.1:5000
```

```

## Usage

1. **Select a Patient**: Choose from the available sample patient profiles
2. **Patient Introduction**: The system generates a natural patient introduction
3. **Information Confirmation**: Verify the extracted patient information
4. **Trial Review**: Review all 10 trials with eligibility explanations
5. **Preference Questions**: Answer questions about your preferences (if multiple trials are eligible)
6. **Final Recommendation**: Receive a personalized trial recommendation with ClinicalTrials.gov link


```

## Technical Details

### Architecture

- **Backend**: Flask server (`server.py`) with agent logic (`agent.py`)
- **Frontend**: Single-page HTML/JavaScript application (`index.html`)
- **Database**: SQLite for preference matching (`trialogue_preferences.db`)
- **LLM**: OpenAI GPT-4o for conversation generation

### Key Components

1. **Eligibility Verification**: Rule-based matching of patient conditions against trial criteria
2. **SQL Preference Matching**: Scores trials based on user preferences (phase, invasiveness, priorities)
3. **Conversational Agent**: GPT-4o generates empathetic, context-aware responses

## Evaluation Results

- **Overall Correctness**: 81.25% across low and high-quality inputs
- **Conversation Quality**: 7.1/10 average rating
- **High-Quality Input Performance**: 100% correctness, 7.6/10 quality
- **Low-Quality Input Performance**: 62.5% correctness, 6.6/10 quality

## Dependencies

See `requirements.txt` for complete list:

- `flask==3.0.0` - Web framework
- `flask-cors==4.0.0` - Cross-origin resource sharing
- `langchain-openai==0.0.2` - OpenAI LLM integration
- `langchain-core==0.1.3` - LangChain core functionality
- `openai==1.6.1` - OpenAI API client

## Acknowledgments

Thanks to:

- Cyrus Zhou for foundational research and dataset provision
- Professor Monica Lam for CS 224V instruction
- Arjun Jain for project mentorship

## Contact

Matthias Jiro Walther (https://www.linkedin.com/in/jirowalther/)
George Song (https://www.linkedin.com/in/georgedsong/)

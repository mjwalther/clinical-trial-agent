# Trialogue Setup Guide

A step-by-step guide to getting Trialogue up and running on your machine.

## Quick Start (5 minutes)

### Step 1: Install Python

Check if you have Python 3.8 or higher:

```bash
python --version
```

If not installed, download from [python.org](https://www.python.org/downloads/)

### Step 2: Install Dependencies

Navigate to the project directory and run:

```bash
pip install -r requirements.txt
```

**If you encounter permission errors**, try:

```bash
pip install --user -r requirements.txt
```

### Step 3: Get OpenAI API Key

1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Sign up or log in
3. Click "Create new secret key"
4. Copy the key (it starts with `sk-`)

### Step 4: Configure Environment

**Option A: Using .env file (Recommended)**

1. Copy the example file:

   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and paste your API key:
   ```
   OPENAI_API_KEY=sk-your-actual-key-here
   ```

**Option B: Set environment variable directly**

For this terminal session only:

**Mac/Linux:**

```bash
export OPENAI_API_KEY='sk-your-actual-key-here'
```

**Windows Command Prompt:**

```cmd
set OPENAI_API_KEY=sk-your-actual-key-here
```

**Windows PowerShell:**

```powershell
$env:OPENAI_API_KEY='sk-your-actual-key-here'
```

### Step 5: Run the Server

```bash
python server.py
```

### Step 6: Open in Browser

Go to: [http://localhost:5000](http://localhost:5000)

---

## Detailed Setup Instructions

### Installing Python (if needed)

#### Windows

1. Download from [python.org](https://www.python.org/downloads/)
2. Run installer
3. ✅ Check "Add Python to PATH"
4. Verify: Open Command Prompt and type `python --version`

#### macOS

```bash
brew install python3
```

Or download from [python.org](https://www.python.org/downloads/)

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3 python3-pip
```

### Installing Dependencies Manually

If `requirements.txt` doesn't work, install packages one by one:

```bash
pip install flask
pip install flask-cors
pip install langchain-openai
pip install langchain-core
pip install openai
```

### Verifying Installation

Check that all packages installed correctly:

```bash
python -c "import flask; import openai; import langchain_openai; print('All packages installed successfully!')"
```

### Directory Structure Verification

Make sure you have these directories and files:

```
trialogue/
├── server.py              ← Server code
├── agent.py               ← Agent logic
├── index.html             ← Frontend interface
├── requirements.txt       ← Dependencies list
├── .env                   ← Your API key (create this)
├── patient_profiles/      ← Patient data (must exist)
│   ├── 20141.json
│   ├── 20142.json
│   └── ...
└── trial_profiles/        ← Trial data (must exist)
    ├── sigir-20141/
    └── ...
```

**Missing data directories?** Contact the project maintainer for the dataset.

---

## Common Issues & Solutions

### Issue: "No module named 'flask'"

**Solution:**

```bash
pip install flask
```

### Issue: "OPENAI_API_KEY not found"

**Solution:**

- Check your `.env` file exists and has the correct format
- Make sure there are no spaces around the `=` sign
- Try setting the environment variable directly (see Step 4, Option B)

### Issue: "Could not load patient profile"

**Solution:**

- Verify `patient_profiles/` directory exists
- Check that JSON files are named correctly (e.g., `20141.json`)
- Ensure JSON files are valid (not corrupted)

### Issue: "Port 5000 already in use"

**Solution:**

Either kill the process using port 5000:

**Mac/Linux:**

```bash
lsof -ti:5000 | xargs kill -9
```

**Windows:**

```cmd
netstat -ano | findstr :5000
taskkill /PID <PID_NUMBER> /F
```

Or change the port in `server.py`:

```python
app.run(debug=True, port=5001)  # Use port 5001 instead
```

### Issue: API Key Invalid

**Solution:**

- Double-check you copied the entire key (starts with `sk-`)
- Verify the key is active at [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- Create a new key if needed

### Issue: Slow Response Times

**Solution:**

- This is normal - GPT-4o takes time to generate responses
- Check your internet connection
- Verify your OpenAI API account has available credits

---

## Testing Your Setup

### Test 1: Server Starts

```bash
python server.py
```

Expected output:

```
======================================================================
Starting Trialogue Server
======================================================================
✓ Eligibility Matching: Rule-based Boolean logic (original)
✓ Preference Matching: SQL-powered queries (NEW!)
...
 * Running on http://127.0.0.1:5000
```

### Test 2: Homepage Loads

Open browser to [http://localhost:5000](http://localhost:5000)

You should see the Trialogue landing page.

### Test 3: Select Patient

Click "Get Started" and select a patient from the dropdown.

If you see the agent greeting, everything is working! 🎉

---

## Next Steps

Once setup is complete:

1. Try a conversation with each patient profile
2. Test different preference combinations
3. Review the eligibility explanations for accuracy
4. Check the final recommendations against ClinicalTrials.gov

---

## Getting Help

If you're still having issues:

1. Check the main README.md for more details
2. Review the error messages carefully
3. Search for the error on Stack Overflow
4. Contact the project maintainer

---

## Development Mode

For development with auto-reload:

```bash
export FLASK_ENV=development
python server.py
```

The server will automatically restart when you make code changes.

---

## Production Deployment

**⚠️ Warning:** This application is designed for research/demonstration purposes. For production deployment, you would need:

- Proper authentication/authorization
- Database migration from SQLite to PostgreSQL/MySQL
- Rate limiting for API calls
- HTTPS configuration
- Environment-specific configurations
- Error logging and monitoring
- Scalable hosting (e.g., AWS, Google Cloud, Azure)

---

## Updating Dependencies

To update all packages to their latest versions:

```bash
pip install --upgrade -r requirements.txt
```

Or update specific packages:

```bash
pip install --upgrade openai langchain-openai
```

---

## Uninstalling

To remove all installed packages:

```bash
pip uninstall -y -r requirements.txt
```

To remove the entire project:

```bash
cd ..
rm -rf trialogue/
```

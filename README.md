# Email Generation Assistant with Custom Evaluation Metrics

This project is a working prototype of an **Email Generation Assistant** built as part of an AI Engineer Candidate Assessment. It generates professional, well-written emails based on user-provided scenarios and evaluates model performance using custom metrics.

---

## 1. Project Overview & Features

The assistant takes three distinct parameters and outputs a structured email:
- **Intent:** The core purpose of the email.
- **Key Facts:** A list of critical bullet points that must be seamlessly included.
- **Tone:** The desired style (e.g., formal, casual, urgent, empathetic).

### Core Methodologies
- **Layered Prompting Strategy:** Combines **Role-Playing** (assigning a Professional Communications Specialist persona), **Few-Shot Examples** (demonstrating expected output schemas), and **Chain-of-Thought (CoT)** reasoning (requiring the model to outline its reasoning plan before drafting the email body).
- **Structured Generation:** Integrates `instructor` patched over an OpenAI-compatible client and validates output schemas using `pydantic` models.
- **Robust Parsing:** Employs Markdown JSON parsing (`Mode.MD_JSON`) enabling compatibility with open-source/smaller models that do not natively support Tool Calling (Function Calling).

---

## 2. Evaluation Strategy & Custom Metrics

The assistant's performance is measured using **three custom metrics** built with the `deepeval` framework:

1. **Fact Recall Metric:** Evaluates the completeness of key facts. Uses an LLM-as-a-Judge system to verify if each individual fact bullet point is semantically present in the generated email. Score = (facts recalled / total facts), ranging from `0.0` to `1.0`.
2. **Tone Adherence Metric:** Evaluates voice matching. Uses an LLM-as-a-Judge system to rate how closely the output aligns with the target tone using a 1–5 scoring rubric. Scores are normalized to a `0.0` to `1.0` scale.
3. **Structure & Professionalism Metric:** Measures email layout flow. Combines a programmatic regex check for standard greetings and sign-offs (40% weight) with an LLM-as-a-Judge check (60% weight) assessing paragraph structure and identifying conversational LLM fluff/meta-text (e.g., *"Here is the email you requested..."*).

---

## 3. System Architecture

The project is structured into modular Python files to enforce separation of concerns and testability:

```text
IMPORTANT/
│
├── .env                  # Local API keys (ignored by git)
├── .gitignore            # Git ignore definitions
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation
├── run_eval.py           # Evaluation Orchestrator (CLI entry point)
│
├── Input/
│   └── scenarios.json    # 10 pre-defined test scenarios & reference emails
│
├── src/
│   ├── __init__.py
│   ├── config.py         # Client initialization & API endpoints credentials management
│   ├── models.py         # Pydantic schemas (scenarios and output structures)
│   ├── generator.py      # Layered prompt implementation & generation wrapper
│   └── evaluator.py      # DeepEval custom metrics implementations
│
└── Output/
    ├── evaluation_report.json   # Detailed logs, metric definitions, and reasons
    └── evaluation_summary.csv   # Tabular comparison of Model A vs. Model B averages
```

---

## 4. Installation & Setup

### Prerequisites
- Python 3.10+
- A valid API Key (e.g., Groq API Key or Cerebras API Key)

### Step 1: Virtual Environment Setup
If you haven't activated your virtual environment:
```powershell
# Create venv if not present
python -m venv venv

# Activate on Windows PowerShell
.\venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
Create a `.env` file in the root of the project:
```env
GROQ_API_KEY=gsk_...
# OR
CEREBRAS_API_KEY=cbs_...
```

Configure your target models in `src/config.py` (e.g., `MODEL_A = "llama-3.1-8b-instant"` and `MODEL_B = "llama-3.3-70b-specdec"`).

---

## 5. Execution & Reporting

You can run the evaluation via the CLI runner:

### A. Run Dry Run (Single Scenario verification)
To test a single scenario first (to verify API key authentication and connection speeds):
```powershell
python run_eval.py --dry-run
```

### B. Run Full Evaluation
To evaluate all 10 scenarios across both configured models:
```powershell
python run_eval.py
```

### Output Files
After completion, two files are created under the `Output/` directory:
- **`Output/evaluation_report.json`:** Rich details containing metric definitions, scenario inputs, generated content, reasoning logs, and metrics scores.
- **`Output/evaluation_summary.csv`:** A clean tabular report detailing latency, scores, and status for each run, suitable for direct spreadsheet imports.

# Job Search Agent

An AI agent built with [smolagents](https://github.com/huggingface/smolagents) that autonomously searches, fetches, and ranks UK ML/AI Engineering job listings against a candidate skill profile.

## What it does

1. Calls the **Adzuna UK Jobs API** to pull real listings filtered to IT/tech roles
2. Fetches each job posting page and strips sidebar/navigation noise
3. Scores each role against a skill profile using a custom calculator tool
4. Filters out senior/lead/staff roles automatically
5. Returns a ranked list sorted by match score

## Architecture

`CodeAgent` orchestrating four tools:

| Tool | Purpose |
|---|---|
| `fetch_jobs_api` | UK job listings from Adzuna (Reed, CV-Library, Guardian Jobs) |
| `fetch_job_description` | Fetches and cleans job posting text |
| `score_match` | Scores description against a comma-separated skill list |
| `DuckDuckGoSearchTool` | Web search fallback |

## Setup

### 1. Get free API keys

- **Adzuna:** register at [developer.adzuna.com](https://developer.adzuna.com) → create app → copy App ID and App Key (free, 250 calls/month)
- **HuggingFace:** register at [huggingface.co](https://huggingface.co/join) → Settings → Tokens → New token (read access)

### 2. Install dependencies

```bash
git clone https://github.com/<your-username>/job-search-agent.git
cd job-search-agent
python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Set environment variables

**Windows:**
```
set ADZUNA_APP_ID=your_app_id
set ADZUNA_APP_KEY=your_app_key
set HF_TOKEN=your_hf_token
```

**Mac/Linux:**
```bash
export ADZUNA_APP_ID=your_app_id
export ADZUNA_APP_KEY=your_app_key
export HF_TOKEN=your_hf_token
```

### 4. Run

```bash
python job_agent.py
```

## Customize

Edit these two strings inside `main()` in `job_agent.py`:

```python
skill_profile = "Python, machine learning, PyTorch, FastAPI, AWS, data science, NLP, SQL"

# Change query, limit, or location inside fetch_jobs_api call in the task
task = f"""
Call fetch_jobs_api(query='machine learning engineer', limit=10).
...
"""
```

## Example output

```
#1
  Title    : Machine Learning Engineer
  Company  : Sanderson
  Location : London, UK
  Score    : Match score: 66.7%, matched (6/9): python, machine learning, pytorch, aws, data science, sql
  URL      : https://www.adzuna.co.uk/jobs/details/...

#2
  Title    : Machine Learning Engineer
  Company  : OpenSourced Ltd
  Location : Bristol, South West England
  Score    : Match score: 11.1%, matched (1/9): machine learning
  URL      : https://www.adzuna.co.uk/jobs/details/...
```

## Challenges faced

| Problem | Fix |
|---|---|
| Free HF model timed out on large pages | Capped fetched description to 3000 chars |
| Remotive API returned irrelevant roles (Sales, Copywriting) | Switched to Adzuna UK IT jobs API |
| Sidebar "Similar Jobs" sections inflated match scores | Strip HTML noise before scoring |
| Model fabricated descriptions when fetch failed | Explicit `FETCH_ERROR` / `SCORE_ERROR` signals force the agent to skip bad results |
| Free HF inference routed through restricted providers | Switched to `Llama-3.3-70B` hosted on HF's own servers |

## Stack

- [smolagents](https://github.com/huggingface/smolagents)
- [Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) via HF Inference API
- [Adzuna Jobs API](https://developer.adzuna.com)
- Python 3.11+

## License

MIT
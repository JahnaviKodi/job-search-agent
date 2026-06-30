# job-search-agent

A multi-tool AI agent built with [smolagents](https://github.com/huggingface/smolagents) that searches, aggregates, and scores UK Machine Learning / AI Engineering job listings against a candidate skill profile.

## Architecture

`CodeAgent` (writes and executes Python at each reasoning step) orchestrating four tools:

| Tool | Purpose |
|---|---|
| `fetch_jobs_api` | Structured job listings via the Remotive public API |
| `DuckDuckGoSearchTool` | Web search fallback when API coverage is thin |
| `VisitWebpageTool` | Fetches full posting text for scoring |
| `score_match` | Calculator tool — scores a posting against a comma-separated skill list |

The agent decides the order of tool calls itself: it tries the API first, falls back to web search if results are sparse, fetches full descriptions, scores each against the skill profile, then ranks and returns the result.

## Setup

```bash
git clone https://github.com/<your-username>/job-search-agent.git
cd job-search-agent
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
```

## Run

```bash
python job_agent.py
```

## Customize

Edit the `skill_profile` and `task` strings in `main()` inside `job_agent.py` to target different roles, locations, or skill sets.

## Notes

- The Remotive API is free and requires no authentication.
- `additional_authorized_imports` restricts the agent's Python execution to an explicit whitelist (`json`, `re`) — this is smolagents' default security boundary. For production use against untrusted models, run with `executor_type="e2b"` or `"docker"` instead of the default local interpreter.
- `max_steps=10` caps the reasoning loop to prevent runaway execution.

## Stack

`smolagents` · `litellm` (Claude Sonnet 4.6 backend) · `requests`

## License

MIT

"""
Job Search Aggregator Agent
Multi-tool CodeAgent built with smolagents.

Tools:
  - search_jobs_web   : web search for current open roles (DuckDuckGo)
  - fetch_jobs_api    : pulls structured listings from the Remotive job API
  - score_match       : calculator tool scoring a job description against a skill profile
  - visit_page        : fetches full text of a job posting page

Usage:
    export ANTHROPIC_API_KEY=your_key
    python job_agent.py
"""

import os
import re
import requests
from smolagents import CodeAgent, tool, LiteLLMModel, DuckDuckGoSearchTool, VisitWebpageTool


# ──────────────────────────────────────────────────────────────
# Tool 1 — Structured job API (Remotive: free, no auth required)
# ──────────────────────────────────────────────────────────────
@tool
def fetch_jobs_api(query: str, limit: int = 10) -> str:
    """Fetches structured job listings from the Remotive job API.

    Args:
        query: Search term for job title or keyword, e.g. 'machine learning'.
        limit: Maximum number of listings to return.
    """
    url = "https://remotive.com/api/remote-jobs"
    params = {"search": query}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return f"API error: {e}"

    jobs = data.get("jobs", [])[:limit]
    if not jobs:
        return "No listings found for this query."

    lines = []
    for j in jobs:
        lines.append(
            f"- {j.get('title')} | {j.get('company_name')} | "
            f"{j.get('candidate_required_location', 'N/A')} | {j.get('url')}"
        )
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Tool 2 — Calculator / scoring tool
# ──────────────────────────────────────────────────────────────
@tool
def score_match(job_description: str, skill_profile: str) -> str:
    """Scores how well a job description matches a candidate skill profile.

    Args:
        job_description: The full text of the job posting.
        skill_profile: Comma-separated list of candidate skills/keywords.
    """
    skills = [s.strip().lower() for s in skill_profile.split(",") if s.strip()]
    text = job_description.lower()

    matched = [s for s in skills if s in text]
    score = round(100 * len(matched) / max(len(skills), 1), 1)

    return (
        f"Match score: {score}%\n"
        f"Matched skills ({len(matched)}/{len(skills)}): {', '.join(matched) if matched else 'none'}"
    )


# ──────────────────────────────────────────────────────────────
# Agent setup
# ──────────────────────────────────────────────────────────────
def build_agent() -> CodeAgent:
    model = LiteLLMModel(model_id="anthropic/claude-sonnet-4-6")

    agent = CodeAgent(
        tools=[
            fetch_jobs_api,
            score_match,
            DuckDuckGoSearchTool(),
            VisitWebpageTool(),
        ],
        model=model,
        max_steps=10,
        additional_authorized_imports=["json", "re"],
    )
    return agent


def main():
    agent = build_agent()

    skill_profile = (
        "Python, machine learning, PyTorch, FastAPI, AWS, "
        "data science, computer vision, NLP, SQL"
    )

    task = f"""
    Find 5 current open graduate or entry-level Machine Learning / AI Engineering
    roles in the UK. Use fetch_jobs_api first for structured listings, then use
    web search to fill gaps if fewer than 5 relevant results come back.

    For each role found, score it against this skill profile using score_match:
    {skill_profile}

    Return a ranked list: job title, company, location, match score, and URL.
    Sort by match score descending.
    """

    result = agent.run(task)
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()

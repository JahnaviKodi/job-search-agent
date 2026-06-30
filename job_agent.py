"""
Job Search Aggregator Agent
Multi-tool CodeAgent built with smolagents.

Tools:
  - fetch_jobs_api        : UK ML/AI jobs from Adzuna API (real UK job boards)
  - fetch_job_description : fetches and cleans a job posting page
  - score_match           : scores a job description against a skill profile
  - DuckDuckGoSearchTool  : web search fallback

Setup:
    1. Register free at developer.adzuna.com → get App ID and App Key
    2. Register free at huggingface.co → get HF token (read access)

    Windows:
        set ADZUNA_APP_ID=your_app_id
        set ADZUNA_APP_KEY=your_app_key
        set HF_TOKEN=your_hf_token

    Mac/Linux:
        export ADZUNA_APP_ID=your_app_id
        export ADZUNA_APP_KEY=your_app_key
        export HF_TOKEN=your_hf_token

    python job_agent.py
"""

import os
import re
import requests
from smolagents import CodeAgent, tool, InferenceClientModel, DuckDuckGoSearchTool


# ─────────────────────────────────────────────────────────────────────
# Tool 1 — Adzuna UK Job API (free tier, 250 calls/month)
# ─────────────────────────────────────────────────────────────────────
@tool
def fetch_jobs_api(query: str, limit: int = 10) -> str:
    """Fetches UK job listings from the Adzuna API, filtered to the IT/tech
    category and excluding senior/lead/staff/principal/director titles.
    Returns one job per line in the exact format:
    title | company | location | url

    Args:
        query: Job search query, e.g. 'machine learning engineer'.
        limit: Maximum number of listings to return.
    """
    app_id  = os.getenv("ADZUNA_APP_ID")
    app_key = os.getenv("ADZUNA_APP_KEY")

    if not app_id or not app_key:
        return (
            "API error: ADZUNA_APP_ID and ADZUNA_APP_KEY not set. "
            "Register free at developer.adzuna.com"
        )

    url = "https://api.adzuna.com/v1/api/jobs/gb/search/1"
    params = {
        "app_id": app_id,
        "app_key": app_key,
        "results_per_page": limit * 2,   # fetch extra to allow for filtering
        "what": query,
        "where": "UK",
        "category": "it-jobs",
        "content-type": "application/json",
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return f"API error: {e}"

    jobs = data.get("results", [])
    if not jobs:
        return "No listings found for this query."

    # Exclude senior/lead/staff/principal roles — targeting graduate/entry-level
    senior_terms = [
        "senior", "staff", "lead", "principal", "architect",
        "head of", "director", "manager", "vp ", "chief",
    ]

    lines = []
    for j in jobs:
        title   = (j.get("title") or "").strip()
        company = (j.get("company", {}).get("display_name") or "Unknown").strip()
        loc     = (j.get("location", {}).get("display_name") or "UK").strip()
        link    = j.get("redirect_url") or ""

        if any(t in title.lower() for t in senior_terms):
            continue

        lines.append(f"{title} | {company} | {loc} | {link}")

        if len(lines) >= limit:
            break

    return "\n".join(lines) if lines else "No non-senior listings found."


# ─────────────────────────────────────────────────────────────────────
# Tool 2 — Job page fetcher with noise stripping
# ─────────────────────────────────────────────────────────────────────
@tool
def fetch_job_description(url: str) -> str:
    """Fetches a job posting URL and returns the cleaned description text,
    stripping navigation, footer, and sidebar noise (e.g. 'Similar Jobs'
    sections) that would pollute keyword scoring. Returns FETCH_ERROR:<reason>
    if the page cannot be fetched or is too short to be useful.

    Args:
        url: The full URL of the job posting page.
    """
    try:
        resp = requests.get(
            url, timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
            allow_redirects=True,
        )
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        return f"FETCH_ERROR: {e}"

    # Cut at known sidebar/noise markers before scoring
    cut_markers = [
        "Similar Remote Jobs", "similar remote jobs",
        "Similar Jobs", "similar jobs",
        "Related Jobs", "related jobs",
        "More jobs from", "more jobs from",
    ]
    for marker in cut_markers:
        idx = html.find(marker)
        if idx != -1:
            html = html[:idx]

    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) < 40:
        return "FETCH_ERROR: page too short or unreadable"

    # Return first 3000 chars — enough for skill scoring, avoids token bloat
    return text[:3000]


# ─────────────────────────────────────────────────────────────────────
# Tool 3 — Skill match scorer
# ─────────────────────────────────────────────────────────────────────
@tool
def score_match(job_description: str, skill_profile: str) -> str:
    """Scores how well a job description matches a candidate skill profile.
    Returns a sortable string: '<numeric_score>|<human_readable_result>'.
    Returns SCORE_ERROR:<reason> if the description is invalid.

    Args:
        job_description: The cleaned text of the job posting.
        skill_profile: Comma-separated list of candidate skills/keywords.
    """
    if job_description.startswith("FETCH_ERROR") or len(job_description.strip()) < 40:
        return "SCORE_ERROR: invalid or too-short description — skip this job"

    skills  = [s.strip().lower() for s in skill_profile.split(",") if s.strip()]
    text    = job_description.lower()
    matched = [s for s in skills if s in text]
    score   = round(100 * len(matched) / max(len(skills), 1), 1)

    readable = (
        f"Match score: {score}%, "
        f"matched ({len(matched)}/{len(skills)}): "
        f"{', '.join(matched) if matched else 'none'}"
    )
    # Prefix with numeric score so agent can sort numerically, not lexically
    return f"{score}|{readable}"


# ─────────────────────────────────────────────────────────────────────
# Agent setup
# ─────────────────────────────────────────────────────────────────────
def build_agent() -> CodeAgent:
    model = InferenceClientModel(
        model_id="meta-llama/Llama-3.3-70B-Instruct",
        timeout=180,
    )

    agent = CodeAgent(
        tools=[
            fetch_jobs_api,
            fetch_job_description,
            score_match,
            DuckDuckGoSearchTool(),
        ],
        model=model,
        max_steps=6,
        additional_authorized_imports=["json", "re"],
    )
    return agent


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    agent = build_agent()

    skill_profile = (
        "Python, machine learning, PyTorch, FastAPI, AWS, "
        "data science, computer vision, NLP, SQL"
    )

    task = f"""
    Step 1: Call fetch_jobs_api(query='machine learning engineer', limit=10).
    The result is plain text, one job per line, in this exact format:
        title | company | location | url
    Parse each line by splitting on ' | '.

    Step 2: For up to 5 of these jobs, call fetch_job_description(url).
    If the result starts with FETCH_ERROR, skip that job entirely.

    Step 3: For each job that has a valid description, call score_match
    with the description and this skill profile:
        {skill_profile}

    score_match returns a string like "44.4|Match score: 44.4%, matched...".
    If it starts with SCORE_ERROR, skip that job.
    Otherwise: split on the first '|' — left side is the numeric score,
    right side is the display text.

    Step 4: Sort all scored jobs by numeric score descending.
    Print and call final_answer with a list of tuples:
        (title, company, location, score_display_text, url)
    """

    result = agent.run(task)
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    # Pretty print if result is a list of tuples
    if isinstance(result, list):
        for i, job in enumerate(result, 1):
            if isinstance(job, (list, tuple)) and len(job) >= 5:
                print(f"\n#{i}")
                print(f"  Title    : {job[0]}")
                print(f"  Company  : {job[1]}")
                print(f"  Location : {job[2]}")
                print(f"  Score    : {job[3]}")
                print(f"  URL      : {job[4]}")
            else:
                print(job)
    else:
        print(result)


if __name__ == "__main__":
    main()
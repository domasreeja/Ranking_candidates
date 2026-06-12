# Redrob Candidate Ranker

An AI-assisted candidate ranking system built for the **Intelligent Candidate
Discovery & Ranking Challenge**. Ranks candidates against a job description
the way a thoughtful recruiter would - reading career history for *evidence*
of fit, not just matching keywords in a skills list - and produces a
top-100 shortlist with per-candidate reasoning.

## Why this architecture

The challenge's compute constraints rule out per-candidate LLM calls:

- No hosted LLM API calls during ranking
- No GPU
- Must run on the full candidate pool in **under 5 minutes on a 16GB CPU
  machine**

So instead of an LLM-in-the-loop ranker, this system encodes "what a great
recruiter actually checks" as an explicit, fast, inspectable feature
pipeline - TF-IDF semantic similarity + rule-based career-substance and
red-flag detection + a behavioral-signal multiplier. See `rank.py` for the
full design rationale (it's documented inline in detail).

## What it does

1. **Semantic fit** - TF-IDF cosine similarity between the JD's "what you'd
   actually be doing" text and each candidate's full career narrative
   (summary + every role's description) - *not* their skills tags. This is
   the key anti-keyword-stuffing move.

2. **Career substance** - regex-based detection of genuinely hands-on
   retrieval/ranking/embeddings/eval work *in role descriptions*, plus
   detection of JD-specific red flags: services-only background, title-chaser
   pattern, architecture/lead drift (no recent code), pure-research-no-
   production, recent-LLM-wrapper-only, vision/speech-only with no NLP/IR.

3. **Location & logistics fit** - Pune/Noida match, relocation willingness,
   notice period vs. the JD's <30-day preference.

4. **Behavioral multiplier** - a 0.55-1.0x multiplier built from
   `redrob_signals` (activity recency, recruiter response rate, interview
   completion, profile completeness, verification status). Multiplicative,
   not additive, per the JD's instruction to *down-weight* (not zero out)
   dormant-but-strong candidates.

5. **Honeypot detection ** - flags internally-impossible profiles (e.g.
   "expert" proficiency with ~0 months of use, broken date logic) and pushes
   them near the bottom of the ranking.

6. **Reasoning generation** - every row's `reasoning` text is built only from
   fields present in that candidate's own data (years of experience, current
   title, specific matched evidence phrases from career descriptions,
   location, notice period, response rate, any red flags). No name-templating,
   no claims about skills the candidate doesn't have.

## Usage

```bash
pip install -r requirements.txt
python3 rank.py data/candidates.json outputs/submission.csv
python3 validate_submission.py outputs/submission.csv
```

Runs in well under a second on the 50-candidate sample, and is designed to
scale linearly (TF-IDF fit + cosine similarity over a sparse matrix, plus
O(1) regex checks per candidate) to the full 100K-candidate pool within the
5-minute CPU budget.

## Repo layout

```
redrob-ranker/
├── rank.py                  # full ranking pipeline (heavily documented)
├── validate_submission.py   # provided validator
├── requirements.txt
├── data/
│   └── candidates.json      # input candidate pool
└── outputs/
    └── submission.csv       # ranked top-N output
```

## Note on the sample run

`outputs/submission.csv` in this repo was generated against the 50-candidate
sample dataset, so it contains 50 ranked rows rather than the 100 required
for a real submission. Running `rank.py` against the full
`candidates.jsonl` (100K candidates) produces the required top-100 rows with
no code changes.

## Possible extensions

- Swap the TF-IDF semantic layer for a small locally-hosted sentence
  embedding model (precomputed offline — still no GPU/API needed at ranking
  time) for richer semantic matching.
- Train an XGBoost learning-to-rank model on top of the current features if
  labeled relevance data becomes available — this was explicitly called out
  as a "nice to have" skill in the JD itself.

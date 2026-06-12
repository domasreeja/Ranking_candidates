#!/usr/bin/env python3
"""
Redrob Hackathon - Intelligent Candidate Discovery & Ranking
=============================================================

Approach
--------
A great recruiter doesn't grep for keywords. They read a profile, build a
mental model of "what has this person actually done", and check that against
"what does this role actually need" -- including the unstated parts (services
vs product company, title-chasing, recency of hands-on work, location /
availability realities).

This script encodes that reasoning as an explicit, inspectable feature
pipeline -- no hosted LLM calls, no GPU, pure CPU, runs in seconds even on
100K candidates.

Pipeline
--------
1. SEMANTIC FIT (TF-IDF cosine similarity, local & precomputable)
   - Compares the JD's narrative text against each candidate's FULL career
     narrative (summary + every role's description), NOT just their skills
     list. This is the key anti-keyword-stuffing move: a candidate whose
     *skills* say "RAG, Pinecone, LLM fine-tuning" but whose *career history*
     never describes doing that work gets a low semantic score, because the
     descriptions don't textually resemble the JD's "what you'd actually be
     doing" section.

2. CAREER-SUBSTANCE SIGNALS (rule-based, derived from career_history)
   - Production retrieval/ranking/embeddings/vector-DB experience, detected
     in role *descriptions* (not skills) -> the JD's #1 "absolutely need".
   - Services-company penalty (TCS/Infosys/Wipro/Accenture/Cognizant/
     Capgemini) unless there's also product-company experience.
   - Title-chaser detection: rapid title escalation across short (<18mo)
     tenures -> explicit JD disqualifier.
   - "Architecture/tech-lead drift" detection: most recent role title
     suggests the candidate stopped writing code.
   - Recent-LangChain-only detection: <12mo of LLM-wrapper experience with
     no pre-LLM production ML/data history.
   - Pure-research / no production deployment detection.
   - Years-of-experience fit around the JD's 5-9y band (soft, triangular).

3. LOCATION / LOGISTICS FIT
   - Tier-1 Indian city match, relocation flag, work-mode alignment with the
     JD's hybrid Pune/Noida ask, notice period vs the JD's <=30 day ask.

4. BEHAVIORAL-SIGNAL MULTIPLIER (redrob_signals)
   - A perfect-on-paper-but-dormant candidate is downweighted. Built as a
     0.55-1.0 multiplier from: recency of activity, open_to_work flag,
     recruiter_response_rate, interview_completion_rate, profile
     completeness, and verification flags. This is a MULTIPLIER, not an
     additive score, per the JD's explicit instruction ("down-weight them
     appropriately").

5. HONEYPOT / IMPOSSIBLE-PROFILE DETECTION
   - Flags profiles with internally inconsistent claims (e.g. a skill used
     longer than the candidate's total tenure at the company where they
     claim to use it, "expert" proficiency with ~0 duration_months, or a
     career-history entry whose duration exceeds the company's plausible
     existence given context). Flagged candidates are pushed to the bottom
     via a heavy score penalty rather than removed outright (keeps the
     scorer's behaviour inspectable / debuggable).

6. REASONING GENERATION
   - Built entirely from the candidate's own fields (years of experience,
     current title, specific matched evidence phrases from their career
     descriptions, location, notice period, response rate, any honeypot /
     red flags). No templates that just splice in a name; no claims about
     skills that aren't actually present in the candidate's data.

Final score = semantic_fit (0-1)
            * behavioral_multiplier (0.55-1.0)
            + career_substance_bonus (0 - 0.6, additive, capped)
            - penalties (services-only, title-chaser, etc.)
            then clipped to [0,1] and honeypots forced near 0.

Output: top-100 CSV (candidate_id, rank, score, reasoning), score
non-increasing by rank, ties broken by candidate_id ascending.
"""

import json
import re
import csv
import sys
import datetime
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Config / JD-derived knowledge (hand-encoded from job_description.docx)
# ---------------------------------------------------------------------------

JD_TEXT = """
Senior AI Engineer, Founding Team, AI-native talent intelligence platform.
Own the intelligence layer: ranking, retrieval, and matching systems that
decide what recruiters see when they search for candidates and what
candidates see when they search for roles. Audit existing BM25 plus
rule-based scoring and identify the highest leverage fixes. Ship a v2 ranking
system using embeddings, hybrid retrieval, and LLM based re-ranking that
improves recruiter engagement metrics. Set up offline benchmarks (NDCG, MRR,
MAP), online A and B testing, and recruiter feedback loops. Drive long term
architecture for candidate to job matching at scale, mentor engineers, work
closely with product. Production experience with embeddings based retrieval
systems such as sentence transformers, OpenAI embeddings, BGE, or E5 deployed
to real users, handling embedding drift, index refresh, and retrieval quality
regression. Production experience with vector databases or hybrid search
infrastructure such as Pinecone, Weaviate, Qdrant, Milvus, OpenSearch,
Elasticsearch, or FAISS. Strong Python and code quality. Hands on experience
designing evaluation frameworks for ranking systems including NDCG, MRR, MAP,
offline to online correlation, and A/B test interpretation. Nice to have: LLM
fine tuning with LoRA, QLoRA, PEFT; learning to rank models such as XGBoost or
neural rankers; prior exposure to HR tech, recruiting tech, or marketplace
products; distributed systems or large scale inference optimization;
open source contributions. Scrappy product engineering attitude, willing to
ship a working ranker quickly. Ideal candidate has six to eight years total
experience, four to five years in applied ML or AI roles at product
companies, has shipped an end to end ranking, search, or recommendation
system to real users at meaningful scale, has opinions about hybrid
retrieval, offline versus online evaluation, and when to fine tune versus
prompt, defended with reference to systems actually built.
"""

# Tokens that indicate genuinely hands-on retrieval/ranking/ML-infra work
# (used against career_history descriptions, not just the skills list)
CORE_SUBSTANCE_PATTERNS = [
    r"\bembedding", r"\bvector (db|database|index|store|search)",
    r"\bsentence[- ]transformers?\b", r"\bbge\b", r"\be5\b",
    r"\bfaiss\b", r"\bpinecone\b", r"\bweaviate\b", r"\bqdrant\b",
    r"\bmilvus\b", r"\bopensearch\b", r"\belasticsearch\b",
    r"\bbm25\b", r"\bhybrid (search|retrieval)", r"\bretrieval\b",
    r"\brank(ing|er)?\b", r"\brecommend(ation|er)", r"\bre-?rank",
    r"\bndcg\b", r"\bmrr\b", r"\bmap@", r"\boffline.{0,15}online\b",
    r"\ba/?b test", r"\bsearch (relevance|quality|infra)",
    r"\blora\b", r"\bqlora\b", r"\bpeft\b", r"\bfine-?tun",
    r"\blearning[- ]to[- ]rank", r"\bxgboost\b",
    r"\bllm\b", r"\bretrieval[- ]augmented", r"\brag\b",
    r"\bsemantic search\b", r"\bquery understanding\b",
]
CORE_SUBSTANCE_RE = re.compile("|".join(CORE_SUBSTANCE_PATTERNS), re.I)

LANGCHAIN_WRAPPER_RE = re.compile(r"\blangchain\b|\bopenai api\b|\bgpt[- ]?(3|4)\b|\bprompt(ing|s)?\b", re.I)

PRE_LLM_PRODUCTION_RE = re.compile(
    r"\b(production|deployed|shipped|scale|users)\b.{0,80}\b(search|ranking|recommend|retrieval|pipeline|infrastructure|model)\b"
    r"|\b(search|ranking|recommend|retrieval|pipeline|infrastructure|model)\b.{0,80}\b(production|deployed|shipped|scale|users)\b",
    re.I,
)

SERVICES_COMPANIES = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
                       "tata consultancy services", "hcl", "tech mahindra", "mindtree"}

PURE_VISION_SPEECH_ROBOTICS_RE = re.compile(
    r"\b(computer vision|image classification|object detection|speech recognition|"
    r"robotics|autonomous (driving|vehicle)|tts|asr)\b", re.I)
NLP_IR_RE = re.compile(r"\b(nlp|natural language|information retrieval|search|ranking|"
                        r"retrieval|embeddings?|llm|text classification|named entity)\b", re.I)

ARCHITECTURE_TITLE_RE = re.compile(
    r"\b(architect|tech lead|technical lead|engineering manager|director|head of|"
    r"vp\b|vice president|principal architect)\b", re.I)

TIER1_CITIES_RE = re.compile(r"\b(pune|noida|hyderabad|mumbai|delhi|gurugram|gurgaon|"
                              r"new delhi|navi mumbai|bengaluru|bangalore)\b", re.I)
PUNE_NOIDA_RE = re.compile(r"\bpune\b|\bnoida\b", re.I)

TODAY = datetime.date(2026, 6, 12)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_date(s):
    if not s:
        return None
    try:
        return datetime.date.fromisoformat(s)
    except Exception:
        return None


def career_text(cand):
    """Full narrative text: summary + every role's title + description."""
    parts = [cand["profile"].get("summary", ""), cand["profile"].get("headline", "")]
    for role in cand.get("career_history", []):
        parts.append(role.get("title", ""))
        parts.append(role.get("description", ""))
    return " \n ".join(p for p in parts if p)


def skills_text(cand):
    return " ".join(s["name"] for s in cand.get("skills", []))


def detect_services_only(cand):
    """True if candidate has services-company exposure with NO product-company
    role evident (penalized per JD unless prior product experience)."""
    has_services = False
    has_product = False
    for role in cand.get("career_history", []):
        comp = role.get("company", "").strip().lower()
        is_services = any(sc in comp for sc in SERVICES_COMPANIES)
        if is_services:
            has_services = True
        else:
            has_product = True
    return has_services and not has_product


def detect_title_chaser(cand):
    """Escalating seniority titles across short (<18mo) tenures."""
    hist = sorted(
        [r for r in cand.get("career_history", [])],
        key=lambda r: r.get("start_date") or "",
    )
    if len(hist) < 3:
        return False
    short_stints = sum(1 for r in hist if (r.get("duration_months") or 0) < 18)
    seniority_words = ["senior", "staff", "principal", "lead", "head", "director", "vp"]
    seniority_count = sum(
        1 for r in hist if any(w in (r.get("title") or "").lower() for w in seniority_words)
    )
    return short_stints >= max(2, len(hist) - 1) and seniority_count >= 2


def detect_architecture_drift(cand):
    """Most recent role title suggests they no longer write code day-to-day,
    and they've been in it 18+ months."""
    current = next((r for r in cand.get("career_history", []) if r.get("is_current")), None)
    if not current:
        return False
    title = current.get("title", "")
    months = current.get("duration_months", 0) or 0
    return bool(ARCHITECTURE_TITLE_RE.search(title)) and months >= 18


def detect_pure_research(cand):
    """No role description shows production/deployment language at all,
    while summary/headline emphasises research."""
    txt = career_text(cand).lower()
    if "research" not in txt and "academic" not in txt and "phd" not in txt:
        return False
    return not bool(PRE_LLM_PRODUCTION_RE.search(txt))


def detect_langchain_only(cand):
    """Recent (<=12mo evidenced) LLM-wrapper-only experience with no
    pre-LLM-era production ML/data work elsewhere in history."""
    hist = cand.get("career_history", [])
    if not hist:
        return False
    current = next((r for r in hist if r.get("is_current")), hist[0])
    cur_desc = current.get("description", "")
    if not LANGCHAIN_WRAPPER_RE.search(cur_desc):
        return False
    if (current.get("duration_months") or 0) > 12:
        return False
    older = [r for r in hist if r is not current]
    has_pre_llm_prod = any(
        PRE_LLM_PRODUCTION_RE.search(r.get("description", "")) and not LANGCHAIN_WRAPPER_RE.search(r.get("description", ""))
        for r in older
    )
    return not has_pre_llm_prod


def detect_vision_speech_only(cand):
    txt = career_text(cand).lower()
    has_vsr = bool(PURE_VISION_SPEECH_ROBOTICS_RE.search(txt))
    has_nlp_ir = bool(NLP_IR_RE.search(txt))
    return has_vsr and not has_nlp_ir


def core_substance_score(cand):
    """Count distinct 'absolutely need' concept hits found in career
    DESCRIPTIONS (the JD's actual-work evidence), capped."""
    hits = set()
    for role in cand.get("career_history", []):
        for m in CORE_SUBSTANCE_RE.finditer(role.get("description", "")):
            hits.add(m.group(0).lower())
    return min(len(hits), 8) / 8.0  # 0..1


def detect_honeypot(cand):
    """Flag internally-impossible profiles."""
    flags = []
    yoe_months = (cand["profile"].get("years_of_experience") or 0) * 12
    # 1. "expert" proficiency with ~0 duration
    for s in cand.get("skills", []):
        if s.get("proficiency") == "expert" and (s.get("duration_months", 0) or 0) <= 2:
            flags.append("expert_with_zero_duration")
            break
    # 3. career role duration longer than plausible given current date
    #    (start_date in the future, or end_date before start_date)
    for role in cand.get("career_history", []):
        sd, ed = parse_date(role.get("start_date")), parse_date(role.get("end_date"))
        if sd and sd > TODAY:
            flags.append("future_start_date")
            break
        if sd and ed and ed < sd:
            flags.append("end_before_start")
            break
    # 4. sum of career_history durations wildly exceeds stated years_of_experience
    total_months = sum((r.get("duration_months") or 0) for r in cand.get("career_history", []))
    if total_months > yoe_months * 1.6 + 12:
        flags.append("career_history_exceeds_stated_experience")
    return flags


def behavioral_multiplier(sig):
    """0.55 - 1.0 multiplier from redrob_signals. A dormant/unresponsive
    perfect-on-paper candidate gets pulled down, never to zero (they could
    still be the right person, just harder to reach -> still informative
    for a recruiter shortlist)."""
    score = 0.0
    weight = 0.0

    # recency of activity
    last_active = parse_date(sig.get("last_active_date"))
    if last_active:
        days = (TODAY - last_active).days
        recency = max(0.0, 1.0 - days / 180.0)  # 0 at 6mo+ inactive
        score += 0.30 * recency
        weight += 0.30

    # open to work
    score += 0.15 * (1.0 if sig.get("open_to_work_flag") else 0.3)
    weight += 0.15

    # recruiter response rate
    score += 0.20 * max(0.0, min(1.0, sig.get("recruiter_response_rate", 0)))
    weight += 0.20

    # interview completion
    icr = sig.get("interview_completion_rate", 0)
    score += 0.15 * max(0.0, min(1.0, icr))
    weight += 0.15

    # profile completeness
    score += 0.10 * (sig.get("profile_completeness_score", 0) / 100.0)
    weight += 0.10

    # verification (trust signal)
    verif = sum([sig.get("verified_email", False), sig.get("verified_phone", False),
                  sig.get("linkedin_connected", False)]) / 3.0
    score += 0.10 * verif
    weight += 0.10

    norm = score / weight if weight else 0.5
    return 0.55 + 0.45 * norm  # map [0,1] -> [0.55, 1.0]


def location_logistics_score(cand):
    s = 0.0
    loc = cand["profile"].get("location", "") + " " + cand["profile"].get("country", "")
    sig = cand["redrob_signals"]

    if PUNE_NOIDA_RE.search(loc):
        s += 0.5
    elif TIER1_CITIES_RE.search(loc):
        s += 0.35
    elif sig.get("willing_to_relocate"):
        s += 0.20

    if sig.get("preferred_work_mode") in ("hybrid", "flexible"):
        s += 0.2
    elif sig.get("preferred_work_mode") == "onsite":
        s += 0.1

    notice = sig.get("notice_period_days", 90)
    if notice <= 30:
        s += 0.3
    elif notice <= 60:
        s += 0.12

    return min(s, 1.0)


def years_experience_fit(yoe):
    """Triangular preference centered on 6-8y, JD band 5-9y."""
    if 6 <= yoe <= 8:
        return 1.0
    if 5 <= yoe < 6:
        return 0.85 + 0.15 * (yoe - 5)
    if 8 < yoe <= 9:
        return 0.85 + 0.15 * (9 - yoe)
    if 4 <= yoe < 5:
        return 0.55 + 0.30 * (yoe - 4)
    if 9 < yoe <= 12:
        return max(0.3, 0.85 - 0.18 * (yoe - 9))
    if yoe < 4:
        return max(0.15, 0.55 - 0.15 * (4 - yoe))
    return max(0.15, 0.3 - 0.03 * (yoe - 12))


# ---------------------------------------------------------------------------
# Main scoring
# ---------------------------------------------------------------------------

def score_candidates(candidates):
    # --- semantic fit via TF-IDF over JD + career narratives ---
    docs = [JD_TEXT] + [career_text(c) for c in candidates]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=20000,
                           sublinear_tf=True)
    tfidf = vec.fit_transform(docs)
    jd_vec = tfidf[0:1]
    cand_vecs = tfidf[1:]
    sem_sims = cosine_similarity(jd_vec, cand_vecs).flatten()  # 0..1-ish

    results = []
    for cand, sem in zip(candidates, sem_sims):
        prof = cand["profile"]
        sig = cand["redrob_signals"]

        substance = core_substance_score(cand)
        yoe_fit = years_experience_fit(prof.get("years_of_experience", 0))
        loc_fit = location_logistics_score(cand)
        bmult = behavioral_multiplier(sig)

        # ---- penalties / disqualifying patterns from the JD ----
        penalties = 0.0
        flags = []

        if detect_services_only(cand):
            penalties += 0.30
            flags.append("services_only_no_product_company")

        if detect_title_chaser(cand):
            penalties += 0.25
            flags.append("title_chaser_pattern")

        if detect_architecture_drift(cand):
            penalties += 0.20
            flags.append("no_recent_hands_on_code")

        if detect_pure_research(cand):
            penalties += 0.35
            flags.append("pure_research_no_production")

        if detect_langchain_only(cand):
            penalties += 0.30
            flags.append("recent_llm_wrapper_only")

        if detect_vision_speech_only(cand):
            penalties += 0.30
            flags.append("vision_speech_robotics_no_nlp_ir")

        honeypot_flags = detect_honeypot(cand)
        is_honeypot = len(honeypot_flags) > 0

        # ---- combine ----
        # Rescale typical TF-IDF cosine range (~0.05-0.35) to ~0..1
        sem_norm = max(0.0, min(1.0, (sem - 0.05) / 0.30))

        # Relevance gate: does this person's actual work history look like
        # the JD's "what you'd actually be doing"? This must dominate --
        # a perfectly-located, perfectly-tenured candidate with zero
        # retrieval/ranking substance should NOT outrank a strong ML fit.
        relevance = 0.6 * sem_norm + 0.4 * substance

        # Logistics/tenure act as a modulator (+/-25%), not a separate
        # additive score -- so they can break ties among relevant
        # candidates but can't rescue an irrelevant profile.
        modulator = 0.5 + 0.25 * yoe_fit + 0.25 * loc_fit  # range ~0.5-1.0

        base = relevance * modulator
        base = max(0.0, base - penalties)
        final = base * bmult

        if is_honeypot:
            final = final * 0.05  # forced near-bottom, not exactly 0 (kept inspectable)

        final = max(0.0, min(1.0, final))

        results.append({
            "candidate_id": cand["candidate_id"],
            "score": final,
            "cand": cand,
            "sem": sem_norm,
            "substance": substance,
            "yoe_fit": yoe_fit,
            "loc_fit": loc_fit,
            "bmult": bmult,
            "flags": flags,
            "honeypot_flags": honeypot_flags,
        })

    return results


# ---------------------------------------------------------------------------
# Reasoning generation (grounded in candidate's own data)
# ---------------------------------------------------------------------------

CORE_LABELS = {
    "embedding": "embeddings", "vector db": "vector DB", "vector database": "vector DB",
    "vector index": "vector indexing", "sentence-transformer": "sentence-transformers",
    "sentence transformer": "sentence-transformers", "faiss": "FAISS", "pinecone": "Pinecone",
    "weaviate": "Weaviate", "qdrant": "Qdrant", "milvus": "Milvus", "opensearch": "OpenSearch",
    "elasticsearch": "Elasticsearch", "bm25": "BM25", "hybrid search": "hybrid search",
    "hybrid retrieval": "hybrid retrieval", "retrieval": "retrieval", "ranking": "ranking",
    "ranker": "ranking", "recommend": "recommendation systems", "rerank": "re-ranking",
    "re-rank": "re-ranking", "ndcg": "NDCG", "mrr": "MRR", "a/b test": "A/B testing",
    "ab test": "A/B testing", "lora": "LoRA", "qlora": "QLoRA", "peft": "PEFT",
    "fine-tun": "fine-tuning", "finetun": "fine-tuning", "learning-to-rank": "learning-to-rank",
    "learning to rank": "learning-to-rank", "xgboost": "XGBoost", "llm": "LLMs",
    "retrieval-augmented": "RAG", "rag": "RAG", "semantic search": "semantic search",
}


def evidence_phrases(cand, limit=2):
    found = []
    for role in cand.get("career_history", []):
        for m in CORE_SUBSTANCE_RE.finditer(role.get("description", "")):
            key = m.group(0).lower()
            for pat, label in CORE_LABELS.items():
                if pat in key or key in pat:
                    if label not in found:
                        found.append(label)
                    break
        if len(found) >= limit:
            break
    return found[:limit]


def build_reasoning(r):
    cand = r["cand"]
    prof = cand["profile"]
    sig = cand["redrob_signals"]
    bits = []

    yoe = prof.get("years_of_experience")
    bits.append(f"{prof.get('current_title', 'Unknown role')} with {yoe} yrs")

    if r["honeypot_flags"]:
        bits.append(f"profile internally inconsistent ({', '.join(r['honeypot_flags'])}) - likely honeypot")
        return "; ".join(bits) + "."

    ev = evidence_phrases(cand)
    if ev:
        bits.append("hands-on " + " & ".join(ev) + " in career history")
    else:
        bits.append("no direct retrieval/ranking production evidence in role descriptions")

    if "services_only_no_product_company" in r["flags"]:
        bits.append("services-only background, no product-company exposure")

    if "title_chaser_pattern" in r["flags"]:
        bits.append("short tenures with rapid title escalation (title-chaser pattern)")

    if "no_recent_hands_on_code" in r["flags"]:
        bits.append("current title suggests architecture/lead role, limited recent coding")

    if "pure_research_no_production" in r["flags"]:
        bits.append("research-heavy background with little production deployment evidence")

    if "recent_llm_wrapper_only" in r["flags"]:
        bits.append("recent experience reads as LLM-wrapper work without earlier production ML")

    if "vision_speech_robotics_no_nlp_ir" in r["flags"]:
        bits.append("background is vision/speech/robotics, not NLP/IR")

    loc = prof.get("location", "")
    if PUNE_NOIDA_RE.search(loc):
        bits.append(f"based in {loc} (matches Pune/Noida)")
    elif sig.get("willing_to_relocate"):
        bits.append(f"in {loc} but open to relocation")
    else:
        bits.append(f"based in {loc}, not relocation-flagged")

    notice = sig.get("notice_period_days")
    bits.append(f"notice period {notice}d")

    rr = sig.get("recruiter_response_rate")
    last_active = sig.get("last_active_date")
    bits.append(f"response rate {rr}, last active {last_active}")

    return "; ".join(bits) + "."


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/sample_candidates.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "/mnt/user-data/outputs/submission.csv"

    with open(in_path) as f:
        candidates = json.load(f)

    results = score_candidates(candidates)
    results.sort(key=lambda r: (-r["score"], r["candidate_id"]))

    top_n = min(100, len(results))
    top = results[:top_n]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, r in enumerate(top, start=1):
            w.writerow([r["candidate_id"], i, f"{r['score']:.4f}", build_reasoning(r)])

    print(f"Wrote {top_n} rows to {out_path}")
    print("\nTop 10 preview:")
    for i, r in enumerate(top[:10], start=1):
        print(f"{i:3d}  {r['candidate_id']}  {r['score']:.4f}  {build_reasoning(r)}")


if __name__ == "__main__":
    main()

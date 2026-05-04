# question_generator.py
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd


# -----------------------------
# Config
# -----------------------------
CSV_PATH = "data/allmovieswithplot.csv"

N_VECTOR = 20
N_CYPHER = 20
N_GENERAL = 10
SEED = 42
OUT_PATH = "data/tool_routing_questions.json"

TOOL_GENERAL = "General Chat"
TOOL_VECTOR = "Movie Plot Search"
TOOL_CYPHER = "Movie information"

STOPWORDS = {
    "a","an","and","are","as","at","be","by","for","from","has","have","he","her","his",
    "i","in","into","is","it","its","of","on","or","she","that","the","their","them","they",
    "this","to","was","were","with","without","you","your","over","after","before","while",
    "when","who","whom","what","where","why","how","than","then","but"
}


# -----------------------------
# Data Model
# -----------------------------
@dataclass
class EvalQuestion:
    qid: str
    category: str                 # "vector" | "cypher" | "general"
    expected_tool: str            # tool label expected by evaluator
    question: str
    user_context: Dict[str, Any]
    ground_truth: Dict[str, Any]
    meta: Dict[str, Any]
    target_title: Optional[str] = None


# -----------------------------
# Helpers
# -----------------------------
def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def first_sentence(text: str) -> str:
    text = normalize_space(text)
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return parts[0] if parts else text


def extract_keywords(plot: str, k: int = 12) -> List[str]:
    plot = normalize_space(plot).lower()
    words = re.findall(r"[a-zA-Z']+", plot)
    words = [w for w in words if len(w) >= 4 and w not in STOPWORDS]

    freq: Dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    ranked = sorted(freq.items(), key=lambda x: (-x[1], -len(x[0]), x[0]))
    return [w for (w, _) in ranked[:k]]


def make_vector_question(plot: str, style: int) -> str:
    """
    Vector-search questions MUST be plot-based and MUST NOT include the movie title.
    """
    plot = normalize_space(plot)
    sent = first_sentence(plot)
    kws = extract_keywords(plot, k=12)
    pick = ", ".join(kws[:6]) if kws else sent[:120]

    styles = [
        f"I'm looking for a movie with a story like this: {sent}",
        f"Find a film based on this plot idea: {sent}",
        f"Which movie matches these plot elements: {pick}?",
        f"Help me find the movie based on this storyline: {sent}",
        f"Search a movie by plot: {sent}",
    ]
    return styles[style % len(styles)]


def make_cypher_question(title: str, style: int) -> str:
    """
    Cypher QA questions are factual DB questions and DO include the movie title.
    """
    title = normalize_space(title)
    templates = [
        f"Who directed {title}?",
        f"What year was {title} released?",
        f"What genre or genres does {title} belong to?",
        f"Who are the main actors in {title}?",
        f"Give me basic information about {title} (director, year, genres).",
        f"List the cast of {title}.",
        f"What is the release year of the movie {title}?",
        f"Show the genres for {title}.",
        f"Who is the director of the movie {title}?",
        f"Which actors starred in {title}?",
    ]
    return templates[style % len(templates)]


def make_general_question(style: int) -> str:
    """
    General chat questions should not require DB lookup or plot retrieval.
    """
    templates = [
        "What makes a mystery movie satisfying without too much violence?",
        "How can a film build suspense without gore?",
        "Explain the difference between noir and neo-noir.",
        "What are common storytelling techniques in crime thrillers?",
        "How should I choose a movie when I only know my mood?",
        "What makes a good plot twist in cinema?",
        "Why do some movies feel slow but still engaging?",
        "What are the criteria for judging whether a movie is well-directed?",
        "How can I describe my movie taste to get better recommendations?",
        "What makes dialogue in a film feel natural and believable?",
    ]
    return templates[style % len(templates)]


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "movie.title" not in df.columns or "movie.plot" not in df.columns:
        raise ValueError("CSV must contain columns: movie.title and movie.plot")

    # drop rows with missing plot/title
    df["movie.title"] = df["movie.title"].astype(str)
    df["movie.plot"] = df["movie.plot"].astype(str)
    df = df[(df["movie.title"].str.strip() != "") & (df["movie.plot"].str.strip() != "")]
    df = df[df["movie.plot"].str.len() >= 30]  # avoid ultra-short plots
    df = df.reset_index(drop=True)

    if len(df) < (N_VECTOR + N_CYPHER):
        raise ValueError(f"Not enough usable rows in CSV. Need at least {N_VECTOR + N_CYPHER}, found {len(df)}.")
    return df


def generate_questions(df: pd.DataFrame) -> List[EvalQuestion]:
    rng = random.Random(SEED)
    idxs = list(range(len(df)))
    rng.shuffle(idxs)

    vector_idxs = idxs[:N_VECTOR]
    cypher_idxs = idxs[N_VECTOR:N_VECTOR + N_CYPHER]

    questions: List[EvalQuestion] = []

    # 20 vector-search questions (plot-based, no title)
    for i, ix in enumerate(vector_idxs):
        title = df.loc[ix, "movie.title"]
        plot = df.loc[ix, "movie.plot"]
        q = make_vector_question(plot, style=i)
        questions.append(
            EvalQuestion(
                qid=f"V{i+1:02d}",
                category="vector",
                expected_tool=TOOL_VECTOR,
                question=q,
                user_context={},
                ground_truth={"target_title": title},  # optional
                meta={"seed": SEED, "source": "allmovieswithplot.csv"},
                target_title=title,
            )
        )

    # 20 cypher QA questions (factual, includes title)
    for i, ix in enumerate(cypher_idxs):
        title = df.loc[ix, "movie.title"]
        q = make_cypher_question(title, style=i)
        questions.append(
            EvalQuestion(
                qid=f"C{i+1:02d}",
                category="cypher",
                expected_tool=TOOL_CYPHER,
                question=q,
                user_context={},
                ground_truth={"target_title": title},  # optional
                meta={"seed": SEED, "source": "allmovieswithplot.csv"},
                target_title=title,
            )
        )

    # 10 general chat questions
    for i in range(N_GENERAL):
        q = make_general_question(style=i)
        questions.append(
            EvalQuestion(
                qid=f"G{i+1:02d}",
                category="general",
                expected_tool=TOOL_GENERAL,
                question=q,
                user_context={},
                ground_truth={},
                meta={"seed": SEED, "source": "templates"},
                target_title=None,
            )
        )

    rng.shuffle(questions)
    return questions


def save_questions(questions: List[EvalQuestion], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": "1.0",
        "seed": SEED,
        "counts": {
            "vector": sum(1 for q in questions if q.category == "vector"),
            "cypher": sum(1 for q in questions if q.category == "cypher"),
            "general": sum(1 for q in questions if q.category == "general"),
            "total": len(questions),
        },
        "questions": [asdict(q) for q in questions],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    df = load_dataset(CSV_PATH)
    questions = generate_questions(df)
    save_questions(questions, OUT_PATH)
    print(f"Saved {len(questions)} labeled questions to: {OUT_PATH}")


if __name__ == "__main__":
    main()

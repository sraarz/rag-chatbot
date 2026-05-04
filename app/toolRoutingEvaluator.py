# run_evaluation.py
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from langchain.callbacks.base import BaseCallbackHandler

from app.agent_strictprompt import chat_agent


QUESTIONS_PATH = "data/tool_routing_questions.json"
OUT_DIR = Path("outputs_strict")
OUT_CSV = OUT_DIR / "tool_routing_eval_results.csv"
OUT_SUMMARY_JSON = OUT_DIR / "tool_routing_eval_summary.json"
OUT_LOG_TXT = OUT_DIR / "tool_routing_eval_log.txt"

TOOL_GENERAL = "General Chat"
TOOL_VECTOR = "Movie Plot Search (Vector)"
TOOL_CYPHER = "Movie information (Graph)"


class TeeLogger:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.fp = None

    def __enter__(self):
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.fp = self.filepath.open("w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.fp:
            self.fp.close()

    def log(self, msg: str = ""):
        print(msg)
        if self.fp:
            self.fp.write(msg + "\n")
            self.fp.flush()


# -----------------------------
# Tool name normalization
# -----------------------------
def normalize_tool_name(name: str) -> str:
    if not name:
        return ""
    n = " ".join(str(name).strip().split())
    low = n.lower()
    mapping = {
        "general chat": TOOL_GENERAL,
        "movie plot search (vector)": TOOL_VECTOR,
        "movie plot search": TOOL_VECTOR,
        "movie information (graph)": TOOL_CYPHER,
        "movie information": TOOL_CYPHER,
    }
    return mapping.get(low, n)


# -----------------------------
# Callback to capture tool usage
# -----------------------------
class ToolCaptureCallback(BaseCallbackHandler):
    raise_error = False
    ignore_chain = False
    ignore_agent = False
    ignore_llm = True
    ignore_chat_model = True
    ignore_tool = False

    def __init__(self):
        super().__init__()
        self.first_tool: Optional[str] = None
        self.all_tools: List[str] = []
        self.tool_inputs: List[str] = []

    def on_tool_start(self, serialized, input_str=None, **kwargs):
        name = None
        if isinstance(serialized, dict):
            name = serialized.get("name") or serialized.get("id") or serialized.get("tool")
        tool_name = normalize_tool_name(name or "UNKNOWN_TOOL")
        self.all_tools.append(tool_name)
        self.tool_inputs.append("" if input_str is None else str(input_str))
        if self.first_tool is None:
            self.first_tool = tool_name


# -----------------------------
# IO helpers
# -----------------------------
def ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_questions(path: str | Path) -> List[Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "questions" not in payload:
        raise ValueError("Invalid questions JSON: missing 'questions'")
    return payload["questions"]


# -----------------------------
# Agent invocation (uses history)
# -----------------------------
def invoke_agent_with_history(question: str, session_id: str) -> Tuple[str, List[str], List[str], Optional[str], float]:
    cb = ToolCaptureCallback()
    t0 = time.time()

    try:
        result = chat_agent.invoke(
            {"input": question},
            {"configurable": {"session_id": session_id}, "callbacks": [cb]},
        )
        final_text = str(result.get("output", "")) if isinstance(result, dict) else str(result)
        dt = time.time() - t0
        return final_text, cb.all_tools, cb.tool_inputs, None, dt

    except Exception as e:
        dt = time.time() - t0
        return "", cb.all_tools, cb.tool_inputs, repr(e), dt


# -----------------------------
# Evaluation logic
# -----------------------------
def evaluate_tool_routing(questions: List[Dict[str, Any]], logger: TeeLogger) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    total = len(questions)

    for idx, q in enumerate(questions, start=1):
        qid = q.get("qid", "")
        category = q.get("category", "")
        expected_tool = normalize_tool_name(q.get("expected_tool", ""))
        question = q.get("question", "")
        target_title = q.get("target_title", None)

        logger.log("=" * 90)
        logger.log(f"Running query {idx}/{total} | qid={qid} | expected={expected_tool} | category={category}")
        logger.log(f"Question: {question}")

        session_id = f"tool-routing-eval-{qid}-{uuid.uuid4()}"
        final_text, tools_called, tool_inputs, error, runtime_sec = invoke_agent_with_history(question, session_id=session_id)

        # -----------------------------
        # CHANGE #1:
        # If NO tool was called, count it as General Chat
        # -----------------------------
        if error is None:
            if tools_called:
                predicted_tool = normalize_tool_name(tools_called[0])
            else:
                predicted_tool = TOOL_GENERAL  # treat as General Chat
        else:
            predicted_tool = "_Exception"

        correct = (predicted_tool == expected_tool) if error is None else False

        logger.log(f"Predicted tool: {predicted_tool}")
        logger.log(f"Tools called: {tools_called}")
        logger.log(f"Runtime: {runtime_sec:.4f} sec")
        if error:
            logger.log(f"ERROR: {error}")
        if final_text:
            logger.log("Final answer:")
            logger.log(final_text)

        rows.append(
            {
                "qid": qid,
                "run_index": idx,  # progress index 1..50
                "category": category,
                "expected_tool": expected_tool,
                "predicted_tool": predicted_tool,
                "correct": correct,
                "question": question,
                "target_title": target_title,
                "tools_called": json.dumps(tools_called, ensure_ascii=False),
                "tool_inputs": json.dumps(tool_inputs, ensure_ascii=False),
                "runtime_sec": round(runtime_sec, 4),
                "error": error or "",
                "final_text": final_text,
                "session_id": session_id,
            }
        )

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> Dict[str, Any]:
    total = len(df)
    overall_acc = float(df["correct"].mean()) if total else 0.0

    per_cat = {}
    for cat in sorted(df["category"].unique()):
        sub = df[df["category"] == cat]
        per_cat[cat] = {
            "n": int(len(sub)),
            "correct": int(sub["correct"].sum()),
            "accuracy": float(sub["correct"].mean()) if len(sub) else 0.0,
        }

    per_tool = {}
    for tool in sorted(df["expected_tool"].unique()):
        sub = df[df["expected_tool"] == tool]
        per_tool[tool] = {
            "n": int(len(sub)),
            "correct": int(sub["correct"].sum()),
            "accuracy": float(sub["correct"].mean()) if len(sub) else 0.0,
        }

    confusion = pd.crosstab(df["expected_tool"], df["predicted_tool"]).to_dict()

    return {
        "total": int(total),
        "overall_accuracy": overall_acc,
        "per_category": per_cat,
        "per_expected_tool": per_tool,
        "confusion": confusion,
        "exception_rate": float((df["predicted_tool"] == "_Exception").mean()) if total else 0.0,
        "avg_runtime_sec": float(df["runtime_sec"].mean()) if total else 0.0,
    }


def main():
    ensure_out_dir()

    questions = load_questions(QUESTIONS_PATH)

    with TeeLogger(OUT_LOG_TXT) as logger:
        logger.log("Starting Tool Routing Evaluation...")
        logger.log(f"Questions file: {QUESTIONS_PATH}")
        logger.log(f"Logging to: {OUT_LOG_TXT}")
        logger.log("")

        df = evaluate_tool_routing(questions, logger)

        df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        summary = summarize(df)
        OUT_SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        logger.log("\n==============================")
        logger.log("Tool Routing Evaluation Result")
        logger.log("==============================")
        logger.log(f"Results CSV:  {OUT_CSV}")
        logger.log(f"Summary JSON: {OUT_SUMMARY_JSON}\n")
        logger.log(json.dumps(summary, ensure_ascii=False, indent=2))

        logger.log("\nConfusion (counts):")
        logger.log(pd.crosstab(df["expected_tool"], df["predicted_tool"]).to_string())


if __name__ == "__main__":
    main()

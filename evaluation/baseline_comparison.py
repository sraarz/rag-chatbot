"""
This module defines a small framework for running baseline experiments on
the tool‐routing agent described in the parent project. A baseline is
defined by a configuration of the agent (e.g. whether it uses
conversation memory, vector search, graph search, and a strict prompt)
and the same set of evaluation questions is run against each
configuration. The results are collected into a pandas DataFrame and
summarised to compute overall accuracy, per‐category accuracy,
per‐expected‐tool accuracy, a confusion matrix and some timing
statistics. The summary is useful for comparing different baseline
strategies.

The key components of this module are:

* ``BaselineComparator`` – a class that loads a question set, builds
  agents according to requested configurations, evaluates them, and
  summarises the results.
* ``build_agent`` – a static method that returns a closure which
  invokes the underlying LangChain agent with or without memory and
  various tools enabled.
* ``evaluate_agent`` – run a list of questions through a supplied
  agent and record detailed results.
* ``summarise`` – compute aggregate statistics from an evaluation
  DataFrame.
* ``run_baselines`` – convenience method to evaluate a set of
  predefined baseline configurations.

The implementations here mirror the logic in the original
``analysis_extras.py`` file but are presented in English with clear
comments. They raise exceptions with helpful messages when required
dependencies are missing. To use this module, the parent project must
provide ``agent_strictprompt.py`` and ``toolRoutingEvaluator.py`` as
well as the necessary LangChain packages.
"""

from __future__ import annotations

import json
import time
import uuid
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from app.agent_strictprompt import (
    llm as _llm,
    graph as _graph,
    agent_prompt as _strict_agent_prompt,
    get_memory as _get_memory,
)

from app.tools.vector import get_movie_plot
from app.tools.cypher import cypher_qa

from app.toolRoutingEvaluator import (
    ToolCaptureCallback,
    normalize_tool_name,
    load_questions as _load_tool_questions,
)


try:
    # LangChain utilities
    from langchain.tools import Tool
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain_neo4j import Neo4jChatMessageHistory  # noqa: F401  # imported for side effects
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
    from langchain.schema import StrOutputParser
except Exception as e:
    raise ImportError(
        "Required modules for building the agent could not be imported. "
        "Ensure that LangChain and its Neo4j integration are installed. "
        f"Original import error: {e}"
    )


class BaselineComparator:
    """Run and summarise baseline experiments for the tool routing agent.

    Parameters
    ----------
    questions_path : str or pathlib.Path
        Path to the JSON file containing the evaluation questions. The
        file must be a JSON object with a top‑level ``"questions"`` key
        mapping to a list of question records.
    """

    def __init__(self, questions_path: str | Path) -> None:
        self.questions_path = Path(questions_path)
        # Load questions once during construction
        self.questions: List[Dict[str, Any]] = _load_tool_questions(self.questions_path)

    # ------------------------------------------------------------------
    # Agent construction
    # ------------------------------------------------------------------
    @staticmethod
    def build_agent(
        use_memory: bool = True,
        use_vector: bool = True,
        use_graph: bool = True,
        strict_prompt: bool = True,
        temperature: float = 0.0,
    ) -> Callable[[str, str], Tuple[str, List[str], List[str], Optional[str], float]]:
        """Construct a configurable agent for tool routing.

        This static method builds a LangChain ReAct agent with optional
        conversation memory and different tool configurations. The
        returned function accepts a user question and a session ID and
        executes the agent, returning the final answer, the sequence of
        tools called, their inputs, any error encountered, and the
        runtime in seconds.

        Parameters
        ----------
        use_memory : bool, optional
            If ``True``, a Neo4j chat message history is supplied so
            that the agent can retain context across turns. Defaults to
            ``True``.
        use_vector : bool, optional
            If ``True``, the vector search tool is included. Defaults
            to ``True``.
        use_graph : bool, optional
            If ``True``, the graph search (Cypher) tool is included.
            Defaults to ``True``.
        strict_prompt : bool, optional
            If ``True``, the strict agent prompt from
            ``agent_strictprompt`` is used. Otherwise a simple
            instruction is used. Defaults to ``True``.
        temperature : float, optional
            Temperature for the underlying language model. Zero makes
            the model deterministic. Defaults to ``0.0``.

        Returns
        -------
        Callable[[str, str], Tuple[str, List[str], List[str], Optional[str], float]]
            A closure that takes a question and session identifier and
            returns the result of invoking the agent.
        """
        # Prepare a deterministic binding of the language model if
        # possible. Not all LLM objects support ``bind``.
        try:
            llm_eval = _llm.bind(temperature=temperature)  # type: ignore[attr-defined]
        except Exception:
            llm_eval = _llm

        # Build the list of tools. Always include a general chat tool.
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a movie expert providing information about movies."),
                ("human", "{input}"),
            ]
        )
        movie_chat = chat_prompt | llm_eval | StrOutputParser()
        tools: List[Tool] = []
        tools.append(
            Tool.from_function(
                name="General Chat",
                description="For general movie chat not covered by other tools",
                func=movie_chat.invoke,
            )
        )
        # Optionally include vector search
        if use_vector:
            tools.append(
                Tool.from_function(
                    name="Movie Plot Search (Vector)",
                    description="For when you need to find information about movies based on a plot",
                    func=_get_movie_plot,
                )
            )
        # Optionally include graph search
        if use_graph:
            tools.append(
                Tool.from_function(
                    name="Movie information (Graph)",
                    description="Provide information about movies questions using Cypher",
                    func=_cypher_qa,
                )
            )

        # Choose the prompt template
        if strict_prompt:
            agent_prompt = _strict_agent_prompt
        else:
            # A simple system prompt for non‑strict experiments
            agent_prompt = PromptTemplate.from_template(
                (
                    "You are a helpful movie expert. Answer the user's question using the available tools if necessary.\n\n"
                    "Conversation history:\n{chat_history}\n\n"
                    "User query:\n{input}\n\n"
                    "{agent_scratchpad}"
                )
            )

        # Create the ReAct agent and its executor
        agent = create_react_agent(llm_eval, tools, agent_prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

        # Define the closure that will call the agent. If memory is
        # enabled, wrap the executor with a message history. Otherwise
        # call it directly. We capture tool usage via the
        # ``ToolCaptureCallback``.
        if use_memory:
            def _agent_call(question: str, session_id: str) -> Tuple[str, List[str], List[str], Optional[str], float]:
                cb = ToolCaptureCallback()
                start = time.time()
                try:
                    runnable = RunnableWithMessageHistory(
                        agent_executor,
                        _get_memory,
                        input_messages_key="input",
                        history_messages_key="chat_history",
                    )
                    result = runnable.invoke(
                        {"input": question},
                        {"configurable": {"session_id": session_id}, "callbacks": [cb]},
                    )
                    final_text = str(result.get("output", "")) if isinstance(result, dict) else str(result)
                    return final_text, cb.all_tools, cb.tool_inputs, None, time.time() - start
                except Exception as exc:
                    return "", cb.all_tools, cb.tool_inputs, repr(exc), time.time() - start
        else:
            def _agent_call(question: str, session_id: str) -> Tuple[str, List[str], List[str], Optional[str], float]:
                cb = ToolCaptureCallback()
                start = time.time()
                try:
                    result = agent_executor.invoke(
                        {"input": question},
                        {"callbacks": [cb]},
                    )
                    final_text = str(result.get("output", "")) if isinstance(result, dict) else str(result)
                    return final_text, cb.all_tools, cb.tool_inputs, None, time.time() - start
                except Exception as exc:
                    return "", cb.all_tools, cb.tool_inputs, repr(exc), time.time() - start

        return _agent_call

    # ------------------------------------------------------------------
    # Evaluation function
    # ------------------------------------------------------------------
    def evaluate_agent(
        self,
        agent_call: Callable[[str, str], Tuple[str, List[str], List[str], Optional[str], float]],
    ) -> pd.DataFrame:
        """Evaluate a given agent on all loaded questions.

        Each question record should contain at least the following keys:
        ``qid`` (a unique identifier), ``category`` (a short string),
        ``expected_tool`` (the canonical name of the correct tool),
        and ``question`` (the user question string). Optional keys like
        ``target_title`` are preserved in the output.

        For each question the agent is invoked with a fresh UUID-based
        session identifier. The predicted tool is derived from the
        first tool called in the intermediate steps. If no tool is
        called, the prediction defaults to "General Chat". If an
        exception occurs during invocation, the predicted tool is
        recorded as "_Exception" and the ``correct`` flag is set to
        ``False``.

        Parameters
        ----------
        agent_call : callable
            The function returned by :meth:`build_agent`. It takes a
            question string and a session ID and returns a tuple of
            ``(final_text, tools_called, tool_inputs, error, runtime_sec)``.

        Returns
        -------
        pandas.DataFrame
            A DataFrame where each row corresponds to a question and
            contains the captured evaluation data.
        """
        rows: List[Dict[str, Any]] = []
        total = len(self.questions)
        for idx, q in enumerate(self.questions, start=1):
            qid: str = q.get("qid", "")
            category: str = q.get("category", "")
            expected_tool: str = normalize_tool_name(q.get("expected_tool", ""))
            question: str = q.get("question", "")
            target_title: Optional[str] = q.get("target_title")

            # Generate a unique session ID for isolation
            session_id = f"baseline-eval-{qid}-{uuid.uuid4()}"
            final_text, tools_called, tool_inputs, error, runtime_sec = agent_call(
                question, session_id=session_id
            )

            # Determine the predicted tool name
            if error is None:
                if tools_called:
                    predicted_tool = normalize_tool_name(tools_called[0])
                else:
                    predicted_tool = "General Chat"
            else:
                predicted_tool = "_Exception"

            correct = (predicted_tool == expected_tool) if error is None else False

            rows.append(
                {
                    "qid": qid,
                    "run_index": idx,
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

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    @staticmethod
    def summarise(df: pd.DataFrame) -> Dict[str, Any]:
        """Compute aggregate statistics from an evaluation DataFrame.

        The returned dictionary contains:
        - ``total``: total number of evaluated questions.
        - ``overall_accuracy``: mean of the ``correct`` column.
        - ``per_category``: a nested dictionary keyed by category
          values with counts, number of correct predictions and
          accuracy per category.
        - ``per_expected_tool``: similar breakdown by expected tool.
        - ``confusion``: a nested dictionary representing the
          confusion matrix (counts of expected vs. predicted tool).
        - ``exception_rate``: fraction of rows where the predicted
          tool is ``_Exception``.
        - ``avg_runtime_sec``: mean runtime in seconds across all rows.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame produced by :meth:`evaluate_agent`.

        Returns
        -------
        dict
            A dictionary summarising the evaluation results.
        """
        total = len(df)
        overall_acc = float(df["correct"].mean()) if total else 0.0

        # Per category breakdown
        per_cat: Dict[str, Dict[str, Any]] = {}
        for cat in sorted(df["category"].unique()):
            sub = df[df["category"] == cat]
            per_cat[cat] = {
                "n": int(len(sub)),
                "correct": int(sub["correct"].sum()),
                "accuracy": float(sub["correct"].mean()) if len(sub) else 0.0,
            }

        # Per expected tool breakdown
        per_tool: Dict[str, Dict[str, Any]] = {}
        for tool in sorted(df["expected_tool"].unique()):
            sub = df[df["expected_tool"] == tool]
            per_tool[tool] = {
                "n": int(len(sub)),
                "correct": int(sub["correct"].sum()),
                "accuracy": float(sub["correct"].mean()) if len(sub) else 0.0,
            }

        # Confusion matrix as nested dict (Pandas uses predicted as columns)
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

    # ------------------------------------------------------------------
    # Run predefined baseline experiments
    # ------------------------------------------------------------------
    def run_baselines(
        self,
        k: int = 1,
        temperature: float = 0.0,
        save_details: bool = False,
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Evaluate a set of standard baseline configurations.

        Five baseline configurations are supported out of the box:

        - ``full``: conversation memory enabled, both vector and graph
          tools enabled, strict prompt.
        - ``no_rag``: memory enabled but both vector and graph tools
          disabled (the agent relies on general chat alone).
        - ``vector_only``: memory enabled, vector tool enabled, graph
          tool disabled.
        - ``graph_only``: memory enabled, graph tool enabled, vector
          tool disabled.
        - ``agentless``: memory disabled and both tools disabled; the
          agent uses only the simple prompt.

        The parameter ``k`` is provided for compatibility with
        recommendation metrics and is not used in this method. The
        ``temperature`` is passed to the underlying language model for
        all configurations. If ``save_details`` is ``True`` the detailed
        evaluation results for each baseline are written to a CSV file
        in the current working directory with a timestamped filename.

        Parameters
        ----------
        k : int, optional
            Currently unused but reserved for potential ranking metrics.
        temperature : float, optional
            Temperature for the underlying language model. Defaults to
            ``0.0``.
        save_details : bool, optional
            If ``True``, write detailed per‑row results to separate CSV
            files. Defaults to ``False``.

        Returns
        -------
        Tuple[pandas.DataFrame, Dict[str, pandas.DataFrame]]
            A summary DataFrame where each row corresponds to a baseline
            configuration, and a dictionary mapping the baseline name
            to its full results DataFrame.
        """
        baseline_configs: Dict[str, Dict[str, Any]] = {
            "full": {
                "use_memory": True,
                "use_vector": True,
                "use_graph": True,
                "strict_prompt": True,
            },
            "no_rag": {
                "use_memory": True,
                "use_vector": False,
                "use_graph": False,
                "strict_prompt": True,
            },
            "vector_only": {
                "use_memory": True,
                "use_vector": True,
                "use_graph": False,
                "strict_prompt": True,
            },
            "graph_only": {
                "use_memory": True,
                "use_vector": False,
                "use_graph": True,
                "strict_prompt": True,
            },
            "agentless": {
                "use_memory": False,
                "use_vector": False,
                "use_graph": False,
                "strict_prompt": False,
            },
        }

        summary_rows: List[Dict[str, Any]] = []
        detailed_dfs: Dict[str, pd.DataFrame] = {}

        for name, cfg in baseline_configs.items():
            agent_call = self.build_agent(
                use_memory=cfg["use_memory"],
                use_vector=cfg["use_vector"],
                use_graph=cfg["use_graph"],
                strict_prompt=cfg["strict_prompt"],
                temperature=temperature,
            )
            df = self.evaluate_agent(agent_call)
            detailed_dfs[name] = df
            summ = self.summarise(df)
            summ["scenario"] = name
            summary_rows.append(summ)
            # Optionally save detailed results for later inspection
            if save_details:
                timestamp = int(time.time())
                out_name = f"baseline_{name}_{timestamp}.csv"
                df.to_csv(out_name, index=False, encoding="utf-8-sig")

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv('baseline_comparison.csv')
        return summary_df, detailed_dfs

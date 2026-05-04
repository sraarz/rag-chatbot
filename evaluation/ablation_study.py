#!/usr/bin/env python3
"""
ablation_study.py
=================

This module exposes a convenience function and class for running
ablation studies on the tool routing agent. An ablation study
systematically removes or toggles components of a system to observe
their individual contributions to overall performance. For example,
one may disable the vector search tool, the graph search tool, or
conversation memory to measure how each affects accuracy and latency.

The core of this module is the :class:`AblationRunner` class. It is
constructed with the path to a questions file and provides a
``run_ablation`` method which accepts a dictionary of configuration
dicts. Each configuration describes whether memory, vector search,
graph search and strict prompts should be enabled. The runner builds
an agent for each configuration, evaluates it on all questions and
computes summary statistics. Optionally the full results for each
configuration can be persisted to CSV files.

Example usage::

    from ablation_study import AblationRunner
    configs = {
        "baseline": {"use_memory": True, "use_vector": True, "use_graph": True, "strict_prompt": True},
        "no_vector": {"use_memory": True, "use_vector": False, "use_graph": True, "strict_prompt": True},
        "no_graph": {"use_memory": True, "use_vector": True, "use_graph": False, "strict_prompt": True},
        "no_memory": {"use_memory": False, "use_vector": True, "use_graph": True, "strict_prompt": True},
    }
    runner = AblationRunner("data/tool_routing_questions.json")
    summary_df, details = runner.run_ablation(configs)

The ``summary_df`` contains one row per scenario with overall accuracy
and other metrics. The ``details`` dictionary maps each scenario name
to its full per‑question DataFrame.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from baseline_comparison import BaselineComparator


class AblationRunner:
    """Run ablation experiments on a set of tool routing configurations.

    Parameters
    ----------
    questions_path : str or pathlib.Path
        Path to the JSON file containing evaluation questions. The
        questions are loaded once during initialisation and reused for
        every ablation run.
    """

    def __init__(self, questions_path: str | Path) -> None:
        self.baseline = BaselineComparator(questions_path)

    def run_ablation(
        self,
        ablation_configs: Dict[str, Dict[str, Any]],
        temperature: float = 0.0,
        save_details: bool = False,
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Execute multiple agent configurations and collect summary statistics.

        Each entry in ``ablation_configs`` maps a scenario name to a
        dictionary specifying the four boolean flags recognised by
        :meth:`baseline_comparison.BaselineComparator.build_agent`:
        ``use_memory``, ``use_vector``, ``use_graph`` and
        ``strict_prompt``. Missing keys default to ``True`` for the
        first three and ``True`` for ``strict_prompt``; thus one can
        specify only the differences relative to the baseline.

        Parameters
        ----------
        ablation_configs : dict
            A mapping from scenario names to agent configuration
            dictionaries. Each dictionary may contain any subset of the
            keys ``use_memory``, ``use_vector``, ``use_graph`` and
            ``strict_prompt``.
        temperature : float, optional
            Temperature for the underlying language model. Defaults to
            ``0.0``.
        save_details : bool, optional
            If ``True``, write the detailed per‑row DataFrame for each
            scenario to a timestamped CSV file. Defaults to ``False``.

        Returns
        -------
        tuple
            A tuple ``(summary_df, detailed_dfs)`` where
            ``summary_df`` is a DataFrame with one row per scenario
            containing aggregated metrics and ``detailed_dfs`` is a
            dictionary mapping scenario names to their per‑question
            DataFrames.
        """
        summary_rows: List[Dict[str, Any]] = []
        detailed_dfs: Dict[str, pd.DataFrame] = {}
        for name, cfg in ablation_configs.items():
            agent_call = self.baseline.build_agent(
                use_memory=cfg.get("use_memory", True),
                use_vector=cfg.get("use_vector", True),
                use_graph=cfg.get("use_graph", True),
                strict_prompt=cfg.get("strict_prompt", True),
                temperature=temperature,
            )
            df = self.baseline.evaluate_agent(agent_call)
            detailed_dfs[name] = df
            summ = self.baseline.summarise(df)
            summ["scenario"] = name
            summary_rows.append(summ)
            if save_details:
                timestamp = int(time.time())
                out_name = f"ablation_{name}_{timestamp}.csv"
                df.to_csv(out_name, index=False, encoding="utf-8-sig")
        summary_df = pd.DataFrame(summary_rows)
        return summary_df, detailed_dfs


def run_ablation(
    questions_path: str | Path,
    ablation_configs: Dict[str, Dict[str, Any]],
    temperature: float = 0.0,
    save_details: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Convenience function to perform an ablation study without instantiating a class.

    This function constructs an :class:`AblationRunner` on the fly
    and delegates to :meth:`AblationRunner.run_ablation`. See the
    documentation of that method for parameter descriptions and return
    values.
    """
    runner = AblationRunner(questions_path)
    return runner.run_ablation(ablation_configs, temperature=temperature, save_details=save_details)
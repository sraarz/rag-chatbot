#!/usr/bin/env python3
"""
error_analysis.py
==================

This module provides helper functions for analysing the errors that
occur during a tool‐routing evaluation. When evaluating an agent that
selects tools to answer questions, it is useful to understand why
incorrect predictions happen. The functions defined here assign
high‐level categories to each row of an evaluation DataFrame based on
the presence or absence of errors and the correctness of the predicted
tool. The resulting categories can then be counted to provide a
summary of error types.

Functions
---------

categorise_errors(df)
    Examine each row of an evaluation DataFrame and add a new
    ``error_category`` column describing the type of failure.

error_summary(df)
    Count the number of occurrences of each error category.

Both functions expect a DataFrame with at least the following columns:
``error`` (a string or empty), ``correct`` (a boolean), and will
operate without modifying any of the other fields.
"""

from typing import Dict

import pandas as pd


def categorise_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Assign an error category to each row of an evaluation DataFrame.

    This function inspects the ``error`` and ``correct`` columns of the
    input DataFrame and produces a new column called ``error_category``.
    The categorisation logic is as follows:

    - If the ``error`` column is non‑empty and contains the substring
      ``"cypher"`` (case insensitive), the row is labelled as
      ``"cypher_generation_error"``. This typically corresponds to
      failures when generating Cypher queries for the graph tool.
    - If the ``error`` column is non‑empty and contains either
      ``"not found"`` or ``"failed"`` (case insensitive), the row is
      labelled as ``"retrieval_failure"``. These errors often arise
      when a tool fails to find the requested information.
    - If the ``error`` column is non‑empty but does not match the above
      substrings, the row is labelled as ``"other_exception"``. This
      catches miscellaneous exceptions thrown during execution.
    - If the ``error`` column is empty and the ``correct`` flag is
      ``False``, the row is labelled as ``"tool_misclassification"``.
      This indicates that the agent selected the wrong tool even though
      no runtime error occurred.
    - Otherwise, the row is labelled as ``"none"``, indicating either
      a correct prediction or that no error could be classified.

    Parameters
    ----------
    df : pandas.DataFrame
        The evaluation results. Must contain ``error`` (str) and
        ``correct`` (bool) columns.

    Returns
    -------
    pandas.DataFrame
        A copy of the input DataFrame with an additional
        ``error_category`` column.
    """
    categories = []
    for _, row in df.iterrows():
        err = row.get("error", "") or ""
        correct = bool(row.get("correct", False))
        if err:
            low = err.lower()
            if "cypher" in low:
                categories.append("cypher_generation_error")
            elif "not found" in low or "failed" in low:
                categories.append("retrieval_failure")
            else:
                categories.append("other_exception")
        else:
            if not correct:
                categories.append("tool_misclassification")
            else:
                categories.append("none")
    out_df = df.copy()
    out_df["error_category"] = categories
    return out_df


def error_summary(df: pd.DataFrame) -> Dict[str, int]:
    """Count the occurrences of each error category.

    After categorising the errors using :func:`categorise_errors`,
    this helper produces a simple dictionary mapping each category
    string to the number of rows belonging to that category. It is
    essentially a convenience wrapper around
    ``df['error_category'].value_counts()``.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame produced by :func:`categorise_errors` or any
        DataFrame that already contains an ``error_category`` column.

    Returns
    -------
    dict
        A mapping from category names to counts.
    """
    if "error_category" not in df.columns:
        raise ValueError(
            "The DataFrame does not contain an 'error_category' column. "
            "Call categorise_errors() first."
        )
    counts = df["error_category"].value_counts()
    return counts.to_dict()
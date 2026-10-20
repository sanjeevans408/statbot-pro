"""StatBot Pro - Smart Natural Language Analyst Engine.

Provides dual-tier data Q&A:
1. Smart Deterministic Engine: Handles aggregations, top/bottom rankings, correlations,
   distributions, counts, and group comparisons offline with zero API requirements.
2. LLM Autonomous Agent: Integrates with NVIDIA ChatNVIDIA or OpenAI for complex reasoning.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _find_best_column_match(query: str, columns: List[str]) -> Optional[str]:
    """Find the column name most likely referenced in the query."""
    q_norm = query.lower()
    
    # Exact substring match (prefer longest match to avoid partials)
    matches = [c for c in columns if c.lower() in q_norm]
    if matches:
        return max(matches, key=len)

    # Word-boundary / tokenized fuzzy match
    q_tokens = set(re.findall(r"\w+", q_norm))
    best_col: Optional[str] = None
    best_score = 0
    for col in columns:
        c_tokens = set(re.findall(r"\w+", col.lower()))
        intersection = q_tokens.intersection(c_tokens)
        if len(intersection) > best_score:
            best_score = len(intersection)
            best_col = col

    return best_col if best_score > 0 else None


def _find_two_column_matches(query: str, columns: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Identify two columns mentioned in the query (e.g. for correlation or groupby)."""
    q_norm = query.lower()
    found = [c for c in columns if c.lower() in q_norm]
    found = sorted(found, key=lambda c: q_norm.find(c.lower()))
    if len(found) >= 2:
        return found[0], found[1]
    return None, None


def _df_to_markdown_safe(df_subset: pd.DataFrame) -> str:
    """Convert dataframe to markdown table with safe fallback."""
    try:
        return df_subset.to_markdown(index=False)
    except Exception:
        headers = [str(c) for c in df_subset.columns]
        header_row = "| " + " | ".join(headers) + " |"
        sep_row = "| " + " | ".join(["---"] * len(headers)) + " |"
        data_rows = []
        for _, row in df_subset.iterrows():
            vals = [str(v) for v in row.values]
            data_rows.append("| " + " | ".join(vals) + " |")
        return "\n".join([header_row, sep_row] + data_rows)


def answer_query_offline(df: pd.DataFrame, query: str) -> Optional[str]:
    """Provide rich deterministic analytical answers without calling any LLM."""
    q = query.strip().lower()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    all_cols = df.columns.tolist()

    # 1. Row / Column / Shape queries
    if any(k in q for k in ["how many rows", "row count", "number of rows", "total records"]):
        return f"📊 **Total Rows:** The dataset contains **{len(df):,}** records."
    if any(k in q for k in ["how many columns", "column count", "number of columns", "features"]):
        return f"📋 **Total Columns ({len(df.columns)}):** {', '.join([f'`{c}`' for c in df.columns[:15]])}" + ("..." if len(df.columns) > 15 else "")
    if any(k in q for k in ["missing values", "null count", "missing count", "nulls"]):
        total_missing = int(df.isna().sum().sum())
        cols_with_missing = df.isna().sum()[lambda x: x > 0]
        if total_missing == 0:
            return "✅ **Data Completeness:** The dataset has **0 missing values** (100% complete)."
        missing_list = "\n".join([f"- **`{c}`**: {cnt:,} missing ({cnt/len(df)*100:.1f}%)" for c, cnt in cols_with_missing.items()])
        return f"⚠️ **Missing Values:** Found **{total_missing:,}** missing values across the dataset:\n\n{missing_list}"

    # 2. Correlation between two variables
    if "correlation" in q or "correlate" in q or "relationship" in q:
        col_a, col_b = _find_two_column_matches(query, numeric_cols)
        if col_a and col_b and col_a != col_b:
            corr_val = df[[col_a, col_b]].dropna().corr().iloc[0, 1]
            strength = "strong positive" if corr_val > 0.6 else "moderate positive" if corr_val > 0.3 else "strong negative" if corr_val < -0.6 else "moderate negative" if corr_val < -0.3 else "negligible"
            return (
                f"🔗 **Correlation between `{col_a}` and `{col_b}`:**\n\n"
                f"- **Pearson Correlation (r):** `{corr_val:+.4f}`\n"
                f"- **Interpretation:** There is a **{strength}** relationship between these two attributes."
            )

    # 3. Top / Bottom N Rankings by Category
    top_match = re.search(r"(top|highest|best|bottom|lowest|worst)\s*(\d+)?\s*(.+)", q)
    if top_match:
        direction = top_match.group(1)
        n_count = int(top_match.group(2)) if top_match.group(2) else 5
        n_count = min(max(1, n_count), 25)
        is_top = direction in ["top", "highest", "best"]

        # Look for category and metric
        matched_cat = _find_best_column_match(query, df.select_dtypes(include=["object", "string", "category"]).columns.tolist())
        matched_num = _find_best_column_match(query, numeric_cols)

        if matched_cat and matched_num:
            agg_data = (
                df.dropna(subset=[matched_cat, matched_num])
                .groupby(matched_cat, as_index=False)[matched_num]
                .mean()
                .sort_values(by=matched_num, ascending=not is_top)
                .head(n_count)
            )
            title = f"{'Top' if is_top else 'Bottom'} {n_count} `{matched_cat}` by Average `{matched_num}`"
            table_md = _df_to_markdown_safe(agg_data)
            return f"🏆 **{title}:**\n\n{table_md}"

    # 4. Grouped Comparison ("average [Metric] by [Category]")
    if " by " in q or " per " in q:
        matched_num = _find_best_column_match(query, numeric_cols)
        matched_cat = _find_best_column_match(query, df.select_dtypes(include=["object", "string", "category"]).columns.tolist())
        if matched_num and matched_cat:
            agg_func = "sum" if "total" in q or "sum" in q else "median" if "median" in q else "mean"
            grouped = (
                df.dropna(subset=[matched_cat, matched_num])
                .groupby(matched_cat, as_index=False)[matched_num]
                .agg(agg_func)
                .sort_values(by=matched_num, ascending=False)
                .head(10)
            )
            table_md = _df_to_markdown_safe(grouped)
            return f"📊 **{agg_func.title()} of `{matched_num}` by `{matched_cat}` (Top 10):**\n\n{table_md}"

    # 5. Direct Aggregations (mean, median, sum, min, max, std, variance)
    ops = {
        "mean": "mean",
        "average": "mean",
        "avg": "mean",
        "median": "median",
        "total": "sum",
        "sum": "sum",
        "minimum": "min",
        "min": "min",
        "maximum": "max",
        "max": "max",
        "highest": "max",
        "lowest": "min",
        "std": "std",
        "standard deviation": "std",
        "variance": "var",
    }
    selected_op = next((val for term, val in ops.items() if term in q), None)
    matched_col = _find_best_column_match(query, all_cols)

    if selected_op and matched_col:
        series = pd.to_numeric(df[matched_col], errors="coerce").dropna()
        if not series.empty:
            result = getattr(series, selected_op)()
            op_display = selected_op.title()
            formatted = f"{result:,.2f}" if isinstance(result, (int, float, np.floating, np.integer)) else str(result)
            return (
                f"📈 **Statistical Result:**\n\n"
                f"- **Operation:** `{op_display}`\n"
                f"- **Column:** `{matched_col}`\n"
                f"- **Computed Value:** **{formatted}**\n\n"
                f"*(Based on {len(series):,} valid numerical records)*"
            )

    # 6. Unique values or Category inspection
    if "unique" in q or "categories" in q or "distinct" in q:
        matched_col = _find_best_column_match(query, all_cols)
        if matched_col:
            uniques = df[matched_col].dropna().unique()
            count = len(uniques)
            sample_vals = [f"`{v}`" for v in uniques[:12]]
            return (
                f"🏷️ **Column `{matched_col}` Unique Values:**\n\n"
                f"- **Distinct Count:** **{count:,}**\n"
                f"- **Sample Values:** {', '.join(sample_vals)}" + ("..." if count > 12 else "")
            )

    return None


def run_analyst_query(df: pd.DataFrame, query: str, api_key: Optional[str] = None) -> str:
    """Execute query via NVIDIA ChatNVIDIA agent if API key is valid; fallback to deterministic engine."""
    effective_key = api_key or os.getenv("NVIDIA_API_KEY", "")

    # If valid NVIDIA API key is available, attempt LangChain agent
    if effective_key.startswith("nvapi-"):
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
            from langchain_experimental.agents import create_pandas_dataframe_agent

            model_name = os.getenv("NVIDIA_MODEL", "deepseek-ai/deepseek-v3.2")
            llm = ChatNVIDIA(model=model_name, temperature=0.1, max_completion_tokens=512)
            agent = create_pandas_dataframe_agent(
                llm,
                df,
                allow_dangerous_code=True,
                verbose=False,
            )
            response = agent.invoke(query)
            out = response.get("output", str(response)) if isinstance(response, dict) else str(response)
            return f"🤖 **AI Analyst Insight:**\n\n{out}"
        except Exception as err:
            # Fallback gracefully
            offline_ans = answer_query_offline(df, query)
            if offline_ans:
                return f"{offline_ans}\n\n*(Note: LLM agent encountered an error, so this answer was computed locally: {err})*"
            return f"⚠️ Could not complete query via LLM: {err}"

    # Offline / Local fallback
    offline_ans = answer_query_offline(df, query)
    if offline_ans:
        return offline_ans

    # Helpful guidance when query doesn't match predefined rules
    numeric_preview = ", ".join([f"`{c}`" for c in df.select_dtypes(include="number").columns[:5]])
    cat_preview = ", ".join([f"`{c}`" for c in df.select_dtypes(include=["object", "string", "category"]).columns[:5]])
    return (
        "💡 **StatBot Deterministic Assistant:**\n\n"
        "I can answer a wide range of questions locally without an API key! Try asking:\n\n"
        f"- **Averages & Sums:** *“What is the average {numeric_preview.split(',')[0]}?”*\n"
        f"- **Rankings:** *“Top 5 {cat_preview.split(',')[0] if cat_preview else 'items'} by {numeric_preview.split(',')[0]}”*\n"
        f"- **Correlations:** *“Correlation between two numeric columns”*\n"
        "- **Dataset Structure:** *“How many rows and columns?”* or *“Missing values”*\n\n"
        "*(To ask open-ended or reasoning questions, provide an NVIDIA API key in the sidebar).* "
    )

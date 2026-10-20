import os
import time
import re
import shlex
from typing import Any, Dict, Optional, Tuple, Literal

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

from requests.exceptions import ReadTimeout, ConnectionError as RequestsConnectionError
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_experimental.agents import create_pandas_dataframe_agent


# =========================
# Configuration
# =========================
MODEL_NAME = os.getenv("NVIDIA_MODEL", "deepseek-ai/deepseek-v3.2")
CSV_PATH = os.getenv("CSV_PATH", "sample.csv")
CHART_DIR = os.getenv("CHART_DIR", "charts")
REQUEST_TIMEOUT = int(os.getenv("NVIDIA_REQUEST_TIMEOUT", "90"))
MAX_COMPLETION_TOKENS = int(os.getenv("NVIDIA_MAX_COMPLETION_TOKENS", "512"))
ALLOW_DANGEROUS_CODE = os.getenv("ALLOW_DANGEROUS_CODE", "true").lower() in ("1", "true", "yes")
RETRIES = int(os.getenv("NVIDIA_RETRIES", "2"))
BACKOFF_BASE = float(os.getenv("NVIDIA_BACKOFF_BASE", "1.5"))

# =========================
# Helpers
# =========================
def _require_api_key() -> None:
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key or not api_key.startswith("nvapi-"):
        raise RuntimeError("NVIDIA_API_KEY is missing or invalid. Set it in your environment (nvapi-...)")


def _load_dataframe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    elif ext in [".parquet", ".pq"]:
        df = pd.read_parquet(path)
    elif ext == ".json":
        df = pd.read_json(path)
    else:
        df = pd.read_csv(path)

    # Normalize column names to avoid trailing/extra whitespace issues
    df.columns = [re.sub(r"\s+", " ", str(col).strip()) for col in df.columns]

    # Handle rows where the whole line is quoted and ends up in the first column
    if df.shape[1] >= 2:
        first_col = df.columns[0]
        other_all_nan = df.iloc[:, 1:].isna().all(axis=None)
        has_commas = df[first_col].astype(str).str.contains(",", regex=False).any()
        if other_all_nan and has_commas:
            split = df[first_col].astype(str).str.strip('"').str.split(",", expand=True)
            split = split.apply(lambda col: col.map(lambda v: v.strip() if isinstance(v, str) else v))
            if split.shape[1] == df.shape[1]:
                split.columns = df.columns
            else:
                split.columns = [f"col{i + 1}" for i in range(split.shape[1])]
            df = split

    # Trim whitespace in object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    # Coerce mostly-numeric columns to numbers
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() >= max(1, int(len(df) * 0.5)):
            df[col] = converted

    # Ensure an explicit index column is available for plotting and queries
    normalized = None
    for col in df.columns:
        if str(col).strip().lower() == "index":
            normalized = col
            break
    if normalized is not None and normalized != "index":
        df = df.rename(columns={normalized: "index"})
    elif "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "index"})
    elif "Unnamed:0" in df.columns:
        df = df.rename(columns={"Unnamed:0": "index"})
    elif "index" not in df.columns:
        df.insert(0, "index", range(len(df)))

    return df


def _normalize_col_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip()).lower()


def _split_two_columns(tokens: list[str], columns: list[str]) -> Tuple[Optional[str], Optional[str]]:
    col_map = {_normalize_col_name(c): c for c in columns}
    for i in range(len(tokens) - 1, 0, -1):
        left = _normalize_col_name(" ".join(tokens[:i]))
        right = _normalize_col_name(" ".join(tokens[i:]))
        if left in col_map and right in col_map:
            return col_map[left], col_map[right]
    return None, None


def _resolve_single_column(tokens: list[str], columns: list[str]) -> Optional[str]:
    col_map = {_normalize_col_name(c): c for c in columns}
    key = _normalize_col_name(" ".join(tokens))
    return col_map.get(key)


def _build_llm() -> ChatNVIDIA:
    return ChatNVIDIA(
        model=MODEL_NAME,
        temperature=0.1,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        model_kwargs={"request_timeout": REQUEST_TIMEOUT},
    )


def create_agent(df: pd.DataFrame):
    llm = _build_llm()
    return create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=ALLOW_DANGEROUS_CODE,
    )


def _invoke_with_retries(agent: Any, prompt: str, retries: int = RETRIES, backoff_base: float = BACKOFF_BASE):
    attempt = 0
    while True:
        try:
            return agent.invoke(prompt)
        except (ReadTimeout, RequestsConnectionError, Exception) as e:
            msg = str(e)
            is_transient = any(code in msg for code in ("504", "502", "503", "timeout", "Timeout"))
            attempt += 1
            if not is_transient or attempt > retries:
                raise
            sleep_s = backoff_base ** attempt
            print(f"Transient error ({e}). Retrying in {sleep_s:.1f}s...")
            time.sleep(sleep_s)


# =========================
# Local plotting (Safe & Extended)
# =========================
def _ensure_chart_path(chart_filename: str) -> str:
    os.makedirs(CHART_DIR, exist_ok=True)
    chart_path = os.path.join(CHART_DIR, chart_filename)
    if os.path.exists(chart_path):
        os.remove(chart_path)
    return chart_path


def plot_data(
    df: pd.DataFrame,
    kind: Literal["hist", "bar", "line", "scatter", "box", "pie", "area", "violin", "heatmap"],
    x: Optional[str] = None,
    y: Optional[str] = None,
    agg: Optional[Literal["mean", "sum", "count", "median", "min", "max"]] = None,
    bins: int = 30,
    title: Optional[str] = None,
    chart_filename: str = "output_chart.png",
) -> Dict[str, Optional[str]]:
    """Safe local plotting with support for multiple graph and chart types."""
    chart_path = _ensure_chart_path(chart_filename)
    plt.figure(figsize=(9, 5.5))

    if kind == "hist":
        if x is None:
            raise ValueError("For hist, provide x as the numeric column.")
        df[x].dropna().plot(kind="hist", bins=bins, color="#4f46e5", edgecolor="black", alpha=0.85)
        plt.xlabel(x)
        plt.ylabel("Frequency")

    elif kind == "bar":
        if x is None or y is None:
            raise ValueError("For bar, provide x (category) and y (value).")
        data = df[[x, y]].dropna()
        if agg:
            data = data.groupby(x, as_index=False)[y].agg(agg)
        data = data.sort_values(by=y, ascending=False).head(20)
        plt.bar(data[x].astype(str), data[y], color="#4f46e5", edgecolor="#3730a3")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(y)

    elif kind == "line":
        if x is None or y is None:
            raise ValueError("For line, provide x and y.")
        data = df[[x, y]].dropna()
        if pd.api.types.is_string_dtype(data[x]):
            with pd.option_context("mode.chained_assignment", None):
                try:
                    data[x] = pd.to_datetime(data[x], errors="coerce")
                except Exception:
                    pass
        data = data.dropna(subset=[x]).sort_values(by=x)
        plt.plot(data[x], data[y], color="#06b6d4", linewidth=2, marker="o", markersize=4)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(y)

    elif kind == "area":
        if x is None or y is None:
            raise ValueError("For area, provide x and y.")
        data = df[[x, y]].dropna().sort_values(by=x)
        plt.fill_between(range(len(data)), data[y], color="#4f46e5", alpha=0.4)
        plt.plot(range(len(data)), data[y], color="#4f46e5", linewidth=2)
        plt.xticks(range(0, len(data), max(1, len(data) // 10)), [str(data[x].iloc[i])[:10] for i in range(0, len(data), max(1, len(data) // 10))], rotation=45, ha="right")
        plt.ylabel(y)

    elif kind == "scatter":
        if x is None or y is None:
            raise ValueError("For scatter, provide x and y.")
        data = df[[x, y]].dropna()
        plt.scatter(data[x], data[y], color="#4f46e5", alpha=0.6, edgecolors="none")
        plt.xlabel(x)
        plt.ylabel(y)

    elif kind == "box":
        if x is None or y is None:
            raise ValueError("For box, provide x (category) and y (numeric).")
        data = df[[x, y]].dropna()
        y_numeric = pd.to_numeric(data[y], errors="coerce")
        if y_numeric.notna().sum() == 0:
            raise ValueError("Box plot requires numeric y values.")
        data = data.copy()
        data[y] = y_numeric
        
        # Limit to top 15 categories by count
        top_cats = data[x].value_counts().head(15).index
        data = data[data[x].isin(top_cats)]
        groups = [g[y].dropna().values for _, g in data.groupby(x)]
        labels = [str(name) for name, _ in data.groupby(x)]
        
        # Modern matplotlib tick_labels handling
        try:
            plt.boxplot(groups, tick_labels=labels, patch_artist=True)
        except TypeError:
            plt.boxplot(groups, labels=labels)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(y)

    elif kind == "pie":
        col = x or y
        if not col:
            raise ValueError("For pie, provide column name.")
        counts = df[col].value_counts().head(8)
        plt.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=140, colors=plt.cm.Paired.colors)

    elif kind == "heatmap":
        numeric_cols = df.select_dtypes(include="number").columns[:10]
        if len(numeric_cols) < 2:
            raise ValueError("Heatmap requires at least 2 numeric columns.")
        corr = df[numeric_cols].corr()
        cax = plt.matshow(corr, cmap="coolwarm", fignum=1)
        plt.colorbar(cax)
        plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
        plt.yticks(range(len(numeric_cols)), numeric_cols)

    else:
        raise ValueError(f"Unsupported kind: {kind}")

    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()

    chart_url = f"http://localhost:8000/{os.path.basename(chart_path)}"
    return {"answer": "Chart generated.", "chart_url": chart_url, "local_path": os.path.abspath(chart_path)}


# =========================
# Agent-driven Q&A (text only)
# =========================
def ask(agent: Any, question: str) -> str:
    resp = _invoke_with_retries(agent, question)
    return resp.get("output") if isinstance(resp, dict) else resp


# =========================
# CLI Interface
# =========================
def main():
    _require_api_key()
    df = _load_dataframe(CSV_PATH)
    print("Columns:", df.columns.tolist())
    print("Shape:", df.shape)

    agent = create_agent(df)

    print("\nStatBot Pro CLI - Available Commands:")
    print("  ask <your question>                                       -> LLM Q&A about dataframe")
    print("  plot bar <x> <y> [mean|sum|count|median] <filename.png>   -> bar chart")
    print("  plot hist <x> <bins> <filename.png>                       -> histogram")
    print("  plot line <x> <y> <filename.png>                          -> line chart")
    print("  plot area <x> <y> <filename.png>                          -> area chart")
    print("  plot scatter <x> <y> <filename.png>                       -> scatter plot")
    print("  plot box <x> <y> <filename.png>                           -> box plot")
    print("  plot pie <col> <filename.png>                             -> pie chart")
    print("  plot heatmap <filename.png>                               -> correlation heatmap")
    print("  quit\n")

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue
        if raw.lower() in {"quit", "exit", "q"}:
            break

        if raw.lower().startswith("ask "):
            question = raw[4:].strip()
            try:
                ans = ask(agent, question)
                print(ans)
            except Exception as e:
                print("Agent error:", e)
            continue

        if raw.lower().startswith("plot "):
            parts = shlex.split(raw)
            try:
                if len(parts) < 3:
                    print("Invalid plot command. See usage above.")
                    continue
                kind = parts[1].lower()
                args = parts[2:]
                filename = args[-1]
                args = args[:-1]

                if kind == "heatmap":
                    res = plot_data(df, kind="heatmap", chart_filename=filename, title="Correlation Matrix")
                    print(res)
                    continue

                if kind == "pie":
                    col = _resolve_single_column(args, df.columns.tolist())
                    if not col:
                        print(f"Column not found: {args}")
                        continue
                    res = plot_data(df, kind="pie", x=col, title=f"Proportions of {col}", chart_filename=filename)
                    print(res)
                    continue

                if kind == "bar":
                    agg = None
                    if args and args[-1].lower() in {"mean", "sum", "count", "median", "min", "max"}:
                        agg = args[-1].lower()
                        args = args[:-1]
                    x, y = _split_two_columns(args, df.columns.tolist())
                    if not x or not y:
                        print("Column not found. Wrap column names with spaces in quotes.")
                        continue
                    res = plot_data(df, kind="bar", x=x, y=y, agg=agg, title=f"{agg or ''} {y} by {x}".strip(), chart_filename=filename)
                    print(res)
                    continue

                if kind == "hist":
                    try:
                        bins = int(args[-1])
                        col_args = args[:-1]
                    except ValueError:
                        bins = 30
                        col_args = args
                    x = _resolve_single_column(col_args, df.columns.tolist())
                    if not x:
                        print("Column not found.")
                        continue
                    res = plot_data(df, kind="hist", x=x, bins=bins, title=f"Histogram of {x}", chart_filename=filename)
                    print(res)
                    continue

                if kind in ["line", "area", "scatter", "box"]:
                    x, y = _split_two_columns(args, df.columns.tolist())
                    if not x or not y:
                        print("Columns not found. Wrap names with spaces in quotes.")
                        continue
                    res = plot_data(df, kind=kind, x=x, y=y, title=f"{y} vs {x}", chart_filename=filename)
                    print(res)
                    continue

                print(f"Unsupported plot type: {kind}")
            except Exception as e:
                print("Plot error:", e)
            continue

        print("Unknown command. Start with 'ask ' or 'plot '.")

if __name__ == "__main__":
    main()

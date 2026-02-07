import os
import time
import re
import shlex
import shlex
from typing import Any, Dict, Optional, Tuple, Literal

import pandas as pd
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
CHART_DIR = os.getenv("CHART_DIR", ".")
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
        raise FileNotFoundError(f"CSV not found at: {path}")
    df = pd.read_csv(path)

    # Normalize column names to avoid trailing/extra whitespace issues
    df.columns = [re.sub(r"\s+", " ", str(col).strip()) for col in df.columns]

    # Handle rows where the whole line is quoted and ends up in the first column.
    if df.shape[1] >= 2:
        first_col = df.columns[0]
        other_all_nan = df.iloc[:, 1:].isna().all(axis=None)
        has_commas = df[first_col].astype(str).str.contains(",", regex=False).any()
        if other_all_nan and has_commas:
            split = df[first_col].astype(str).str.strip('"').str.split(",", expand=True)
            # pandas 3.x removed DataFrame.applymap; use column-wise map instead
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

    # Ensure an explicit index column is available for plotting and queries.
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
    # Keep it simple: no streaming; pass timeout in model_kwargs; use max_completion_tokens
    return ChatNVIDIA(
        model=MODEL_NAME,
        temperature=0.1,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        model_kwargs={
            # Avoid streaming to reduce fragility on Integrate endpoint
            "request_timeout": REQUEST_TIMEOUT
        },
    )


def create_agent(df: pd.DataFrame):
    llm = _build_llm()
    # Remove handle_parsing_errors (no longer supported by your installed version)
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
            # Treat 5xx/timeout as transient up to retries
            msg = str(e)
            is_transient = any(code in msg for code in ("504", "502", "503", "timeout", "Timeout"))
            attempt += 1
            if not is_transient or attempt > retries:
                raise
            sleep_s = backoff_base ** attempt
            print(f"Transient error ({e}). Retrying in {sleep_s:.1f}s...")
            time.sleep(sleep_s)


# =========================
# Local plotting (no LLM)
# =========================
def _ensure_chart_path(chart_filename: str) -> str:
    os.makedirs(CHART_DIR, exist_ok=True)
    chart_path = os.path.join(CHART_DIR, chart_filename)
    if os.path.exists(chart_path):
        os.remove(chart_path)
    return chart_path


def plot_data(
    df: pd.DataFrame,
    kind: Literal["hist", "bar", "line", "scatter", "box"],
    x: Optional[str] = None,
    y: Optional[str] = None,
    agg: Optional[Literal["mean", "sum", "count", "median"]] = None,
    bins: int = 30,
    title: Optional[str] = None,
    chart_filename: str = "output_chart.png",
) -> Dict[str, Optional[str]]:
    """
    Safe local plotting without LLM.

    Examples:
    - Histogram: kind="hist", x="Revenue", bins=30
    - Bar of avg by Region: kind="bar", x="Region", y="Revenue", agg="mean"
    - Line: kind="line", x="Date", y="Revenue"
    - Scatter: kind="scatter", x="Quantity", y="Revenue"
    - Box by category: kind="box", x="ProductCategory", y="Revenue"
    """
    chart_path = _ensure_chart_path(chart_filename)

    plt.figure(figsize=(8, 5))
    if kind == "hist":
        if x is None:
            raise ValueError("For hist, provide x as the numeric column.")
        df[x].dropna().plot(kind="hist", bins=bins, edgecolor="black")
        plt.xlabel(x)

    elif kind == "bar":
        if x is None or y is None:
            raise ValueError("For bar, provide x (category) and y (value).")
        data = df[[x, y]].dropna()
        if agg:
            data = data.groupby(x, as_index=False)[y].agg(agg)
        data = data.sort_values(by=y, ascending=False)
        plt.bar(data[x].astype(str), data[y])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(y)

    elif kind == "line":
        if x is None or y is None:
            raise ValueError("For line, provide x (typically time) and y.")
        data = df[[x, y]].dropna()
        # If x looks like dates, try parse
        if pd.api.types.is_string_dtype(data[x]):
            with pd.option_context("mode.chained_assignment", None):
                try:
                    data[x] = pd.to_datetime(data[x], errors="coerce")
                except Exception:
                    pass
        data = data.dropna(subset=[x])
        data = data.sort_values(by=x)
        plt.plot(data[x], data[y])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(y)

    elif kind == "scatter":
        if x is None or y is None:
            raise ValueError("For scatter, provide x and y.")
        data = df[[x, y]].dropna()
        plt.scatter(data[x], data[y], alpha=0.7)
        plt.xlabel(x)
        plt.ylabel(y)

    elif kind == "box":
        if x is None or y is None:
            raise ValueError("For box, provide x (category) and y.")
        # Boxplot by category
        data = df[[x, y]].dropna()
        y_numeric = pd.to_numeric(data[y], errors="coerce")
        if y_numeric.notna().sum() == 0:
            x_numeric = pd.to_numeric(data[x], errors="coerce")
            if x_numeric.notna().sum() > 0:
                raise ValueError(f"Box plot requires numeric y. Did you mean: plot box \"{y}\" \"{x}\" <filename.png>?")
            raise ValueError("Box plot requires numeric y values.")
        data = data.copy()
        data[y] = y_numeric
        groups = [g[y].dropna().values for _, g in data.groupby(x)]
        labels = [str(name) for name, _ in data.groupby(x)]
        plt.boxplot(groups, labels=labels)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(y)

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
# CLI
# =========================
def main():
    _require_api_key()
    df = _load_dataframe(CSV_PATH)
    print("Columns:", df.columns.tolist())
    print("Shape:", df.shape)

    # Build agent for Q&A
    agent = create_agent(df)

    print("Commands:")
    print("  ask <your question>            -> LLM Q&A about the dataframe")
    print("  plot bar <x> <y> [mean|sum|count|median] <filename.png>  -> bar chart (optional agg)")
    print("  plot hist <x> <bins> <filename.png>                       -> histogram")
    print("  plot line <x> <y> <filename.png>                          -> line chart")
    print("  plot scatter <x> <y> <filename.png>                       -> scatter")
    print("  plot box <x> <y> <filename.png>                           -> box plot")
    print("  Tip: wrap column names with spaces in quotes, e.g. plot box \"Avg Daily Distance (km)\" Brand out.png")
    print("  quit")

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
                if len(args) < 2:
                    print("Invalid plot command. See usage above.")
                    continue
                filename = args[-1]
                args = args[:-1]
                if kind == "bar":
                    # plot bar <x> <y> [agg] <filename>
                    if len(args) < 2:
                        print("Usage: plot bar <x> <y> [mean|sum|count|median] <filename.png>")
                        continue
                    agg = None
                    if args and args[-1].lower() in {"mean", "sum", "count", "median"}:
                        agg = args[-1].lower()
                        args = args[:-1]
                    x, y = _split_two_columns(args, df.columns.tolist())
                    if not x or not y:
                        print("Plot error: column name not found. Use quotes for columns with spaces.")
                        continue
                    result = plot_data(df, kind="bar", x=x, y=y, agg=agg, title=f"{agg or ''} {y} by {x}".strip(), chart_filename=filename)
                elif kind == "hist":
                    # plot hist <x> <bins> <filename>
                    if len(args) < 2:
                        print("Usage: plot hist <x> <bins> <filename.png>")
                        continue
                    try:
                        bins = int(args[-1])
                    except ValueError:
                        print("Usage: plot hist <x> <bins> <filename.png>")
                        continue
                    x = _resolve_single_column(args[:-1], df.columns.tolist())
                    if not x:
                        print("Plot error: column name not found. Use quotes for columns with spaces.")
                        continue
                    result = plot_data(df, kind="hist", x=x, bins=bins, title=f"Histogram of {x}", chart_filename=filename)
                elif kind == "line":
                    # plot line <x> <y> <filename>
                    if len(args) < 2:
                        print("Usage: plot line <x> <y> <filename.png>")
                        continue
                    x, y = _split_two_columns(args, df.columns.tolist())
                    if not x or not y:
                        print("Plot error: column name not found. Use quotes for columns with spaces.")
                        continue
                    result = plot_data(df, kind="line", x=x, y=y, title=f"{y} over {x}", chart_filename=filename)
                elif kind == "scatter":
                    # plot scatter <x> <y> <filename>
                    if len(args) < 2:
                        print("Usage: plot scatter <x> <y> <filename.png>")
                        continue
                    x, y = _split_two_columns(args, df.columns.tolist())
                    if not x or not y:
                        print("Plot error: column name not found. Use quotes for columns with spaces.")
                        continue
                    result = plot_data(df, kind="scatter", x=x, y=y, title=f"{y} vs {x}", chart_filename=filename)
                elif kind == "box":
                    # plot box <x> <y> <filename>
                    if len(args) < 2:
                        print("Usage: plot box <x> <y> <filename.png>")
                        continue
                    x, y = _split_two_columns(args, df.columns.tolist())
                    if not x or not y:
                        print("Plot error: column name not found. Use quotes for columns with spaces.")
                        continue
                    result = plot_data(df, kind="box", x=x, y=y, title=f"{y} by {x}", chart_filename=filename)
                else:
                    print("Unsupported plot kind. Use: bar, hist, line, scatter, box")
                    continue
                print(result)
            except Exception as e:
                print("Plot error:", e)
            continue

        print("Unknown command. Start with 'ask ' or 'plot ' as shown in usage.")

if __name__ == "__main__":
    main()

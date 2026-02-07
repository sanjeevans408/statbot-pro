import os
import time
import shlex
import builtins
import collections
import datetime
import decimal
import itertools
import json
import math
import random
import re
import statistics
from typing import Any, Dict, Optional, Tuple, Literal, List

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

from requests.exceptions import ReadTimeout, ConnectionError as RequestsConnectionError
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_experimental.agents import create_pandas_dataframe_agent
try:
    from langchain.callbacks.base import BaseCallbackHandler
except Exception:
    try:
        from langchain_core.callbacks.base import BaseCallbackHandler
    except Exception:
        BaseCallbackHandler = object  # type: ignore[assignment]


# =========================
# Configuration
# =========================
MODEL_NAME = os.getenv("NVIDIA_MODEL", "deepseek-ai/deepseek-v3.2")
CSV_PATH = os.getenv("CSV_PATH", "sample.csv")
CHART_DIR = os.getenv("CHART_DIR", "charts")
REQUEST_TIMEOUT = int(os.getenv("NVIDIA_REQUEST_TIMEOUT", "90"))
MAX_COMPLETION_TOKENS = int(os.getenv("NVIDIA_MAX_COMPLETION_TOKENS", "512"))
ALLOW_DANGEROUS_CODE = os.getenv("ALLOW_DANGEROUS_CODE", "true").lower() in ("1", "true", "yes")
ALLOW_LLM = os.getenv("ALLOW_LLM", "true").lower() in ("1", "true", "yes")
STREAM_THOUGHTS = os.getenv("STREAM_THOUGHTS", "true").lower() in ("1", "true", "yes")
THOUGHTS_MAX_CHARS = int(os.getenv("THOUGHTS_MAX_CHARS", "240"))
#ALLOW_DANGEROUS_CODE ="true"
#LLOW_LLM = "true"

RETRIES = int(os.getenv("NVIDIA_RETRIES", "2"))
BACKOFF_BASE = float(os.getenv("NVIDIA_BACKOFF_BASE", "1.5"))
SANDBOX_ENABLED = os.getenv("SANDBOX_ENABLED", "true").lower() in ("1", "true", "yes")
SANDBOX_DIR = os.getenv("SANDBOX_DIR", ".sandbox")
SANDBOX_ALLOWED_IMPORTS = {
    "pandas",
    "numpy",
    "math",
    "statistics",
    "datetime",
    "decimal",
    "re",
    "itertools",
    "collections",
    "json",
    "random",
}

# =========================
# Helpers
# =========================
def _require_api_key() -> None:
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key or not api_key.startswith("nvapi-"):
        raise RuntimeError("NVIDIA_API_KEY is missing or invalid. Set it in your environment (nvapi-...)")

def _safe_path(path: str, base_dir: str) -> str:
    base = os.path.normcase(os.path.realpath(os.fspath(base_dir)))
    full = os.path.normcase(os.path.realpath(os.fspath(path)))
    if full != base and not full.startswith(base + os.sep):
        raise ValueError(f"Blocked path outside allowed directory: {path}")
    return full


def _resolve_sandbox_dir(path: str) -> str:
    base_dir = os.path.abspath(path)
    cwd = os.path.abspath(os.getcwd())
    # Ensure sandbox is a subdirectory of the working dir to keep it isolated.
    if os.path.normcase(base_dir) == os.path.normcase(cwd):
        raise ValueError("SANDBOX_DIR must be a subdirectory of the working directory.")
    _safe_path(base_dir, cwd)
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def _squash_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _coerce_text(value: Any) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=True)
        except Exception:
            text = str(value)
    return _squash_text(text)


def _truncate_text(value: Any, limit: int) -> str:
    if value is None:
        return "<none>"
    text = _coerce_text(value)
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


class ThoughtStream:
    def __init__(self, enabled: bool = True, max_chars: int = THOUGHTS_MAX_CHARS):
        self.enabled = enabled
        self.max_chars = max_chars
        self.events: List[Dict[str, str]] = []

    def emit(self, kind: str, message: str) -> None:
        if not self.enabled:
            return
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {kind.upper():<7} {message}"
        print(line, flush=True)
        self.events.append({"ts": ts, "kind": kind, "message": message})


class ThoughtCallbackHandler(BaseCallbackHandler):
    def __init__(self, stream: ThoughtStream):
        self.stream = stream
        self.step = 0

    def on_agent_action(self, action, **kwargs):
        self.step += 1
        tool = getattr(action, "tool", "tool")
        tool_input = getattr(action, "tool_input", None)
        msg = f"Step {self.step}: tool {tool}"
        if tool_input is not None:
            msg += f" | input: {_truncate_text(tool_input, self.stream.max_chars)}"
        self.stream.emit("action", msg)

    def on_tool_end(self, output, **kwargs):
        obs = _truncate_text(output, self.stream.max_chars)
        self.stream.emit("observe", f"Step {self.step}: observation: {obs}")

    def on_chain_error(self, error, **kwargs):
        self.stream.emit("error", f"Agent error: {error}")

    def on_tool_error(self, error, **kwargs):
        self.stream.emit("error", f"Tool error: {error}")


class Sandbox:
    def __init__(self, base_dir: str, allowed_imports: set[str]):
        self.base_dir = _resolve_sandbox_dir(base_dir)
        self.allowed_imports = set(allowed_imports)
        self._orig_open = builtins.open
        self._orig_import = builtins.__import__
        self._orig_cwd: Optional[str] = None
        self._orig_os: Dict[str, Any] = {}

    def _coerce_path(self, path: Any) -> Any:
        if isinstance(path, (str, bytes, os.PathLike)):
            raw = os.fspath(path)
            if isinstance(raw, bytes):
                raw = os.fsdecode(raw)
            return _safe_path(raw, self.base_dir)
        return path

    def _safe_open(self, file: Any, *args: Any, **kwargs: Any):
        if isinstance(file, (str, bytes, os.PathLike)):
            file = self._coerce_path(file)
        return self._orig_open(file, *args, **kwargs)

    def _restricted_import(self, name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0):
        root = name.split(".", 1)[0]
        if root not in self.allowed_imports:
            raise ImportError(f"Import of '{root}' is blocked in sandbox.")
        return self._orig_import(name, globals, locals, fromlist, level)

    def restricted_builtins(self) -> Dict[str, Any]:
        safe = dict(vars(builtins))
        safe["open"] = self._safe_open
        safe["__import__"] = self._restricted_import
        return safe

    def restricted_globals(self) -> Dict[str, Any]:
        return {
            "__builtins__": self.restricted_builtins(),
            "pd": pd,
            "np": np,
            "math": math,
            "statistics": statistics,
            "datetime": datetime,
            "decimal": decimal,
            "re": re,
            "itertools": itertools,
            "collections": collections,
            "json": json,
            "random": random,
        }

    def _wrap_os_single(self, func):
        def wrapper(path: Any = ".", *args: Any, **kwargs: Any):
            return func(self._coerce_path(path), *args, **kwargs)
        return wrapper

    def _wrap_os_double(self, func):
        def wrapper(src: Any, dst: Any, *args: Any, **kwargs: Any):
            return func(self._coerce_path(src), self._coerce_path(dst), *args, **kwargs)
        return wrapper

    def _blocked_os(self, *_: Any, **__: Any):
        raise PermissionError("Operation blocked by sandbox.")

    def __enter__(self):
        self._orig_cwd = os.getcwd()
        os.chdir(self.base_dir)
        builtins.open = self._safe_open

        patch_single = (
            "open",
            "listdir",
            "scandir",
            "walk",
            "remove",
            "unlink",
            "rmdir",
            "mkdir",
            "makedirs",
            "stat",
            "lstat",
            "access",
            "chdir",
        )
        patch_double = ("rename", "replace")
        for name in patch_single:
            if hasattr(os, name):
                self._orig_os[name] = getattr(os, name)
                setattr(os, name, self._wrap_os_single(getattr(os, name)))
        for name in patch_double:
            if hasattr(os, name):
                self._orig_os[name] = getattr(os, name)
                setattr(os, name, self._wrap_os_double(getattr(os, name)))

        blocked = (
            "system",
            "popen",
            "startfile",
            "spawnl",
            "spawnle",
            "spawnlp",
            "spawnlpe",
            "spawnv",
            "spawnve",
            "spawnvp",
            "spawnvpe",
            "execl",
            "execle",
            "execlp",
            "execlpe",
            "execv",
            "execve",
            "execvp",
            "execvpe",
        )
        for name in blocked:
            if hasattr(os, name):
                self._orig_os[name] = getattr(os, name)
                setattr(os, name, self._blocked_os)
        return self

    def __exit__(self, exc_type, exc, tb):
        builtins.open = self._orig_open
        for name, func in self._orig_os.items():
            setattr(os, name, func)
        if self._orig_cwd is not None:
            os.chdir(self._orig_cwd)
        self._orig_os.clear()
        return False

    def run(self, func, *args: Any, **kwargs: Any):
        with self:
            return func(*args, **kwargs)


def _load_dataframe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at: {path}")
    # Restrict CSV access to current working directory tree
    _safe_path(path, os.getcwd())
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


def _init_sandbox() -> Optional[Sandbox]:
    if not SANDBOX_ENABLED:
        return None
    return Sandbox(SANDBOX_DIR, SANDBOX_ALLOWED_IMPORTS)


def _apply_sandbox_to_agent(agent: Any, sandbox: Sandbox) -> None:
    try:
        from langchain_experimental.tools.python.tool import PythonAstREPLTool
    except Exception:
        return
    for tool in getattr(agent, "tools", []):
        if isinstance(tool, PythonAstREPLTool):
            tool.globals = sandbox.restricted_globals()
            if tool.locals is None:
                tool.locals = {}
            orig_run = tool._run
            def _sandboxed_run(query: str, run_manager: Any = None, _orig: Any = orig_run):
                return sandbox.run(_orig, query, run_manager)
            tool._run = _sandboxed_run


def _build_sandboxed_repl(df: pd.DataFrame, sandbox: Sandbox):
    from langchain_experimental.tools.python.tool import PythonAstREPLTool
    return PythonAstREPLTool(globals=sandbox.restricted_globals(), locals={"df": df})


def create_agent(df: pd.DataFrame, sandbox: Optional[Sandbox]):
    if not ALLOW_LLM:
        raise RuntimeError("LLM features are disabled. Set ALLOW_LLM=true to enable.")
    if not ALLOW_DANGEROUS_CODE:
        raise RuntimeError("Dangerous code execution is disabled. Set ALLOW_DANGEROUS_CODE=true to enable.")
    if sandbox is None:
        raise RuntimeError("Sandboxing must be enabled to use the LLM agent. Set SANDBOX_ENABLED=true.")
    llm = _build_llm()
    # Remove handle_parsing_errors (no longer supported by your installed version)
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=False,
        allow_dangerous_code=ALLOW_DANGEROUS_CODE,
    )
    _apply_sandbox_to_agent(agent, sandbox)
    return agent


def _invoke_agent(agent: Any, prompt: str, callbacks: Optional[list] = None):
    if callbacks:
        try:
            return agent.invoke(prompt, config={"callbacks": callbacks})
        except TypeError:
            try:
                return agent.invoke(prompt, callbacks=callbacks)
            except TypeError:
                return agent.invoke(prompt)
    return agent.invoke(prompt)


def _invoke_with_retries(
    agent: Any,
    prompt: str,
    callbacks: Optional[list] = None,
    retries: int = RETRIES,
    backoff_base: float = BACKOFF_BASE,
):
    attempt = 0
    while True:
        try:
            return _invoke_agent(agent, prompt, callbacks)
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
    # prevent path traversal in filename
    safe_name = os.path.basename(chart_filename)
    chart_path = os.path.join(CHART_DIR, safe_name)
    _safe_path(chart_path, os.path.abspath(CHART_DIR))
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
def ask(
    agent: Any,
    question: str,
    stream_thoughts: bool = STREAM_THOUGHTS,
    return_thoughts: bool = False,
):
    callbacks = None
    thought_stream = None
    if stream_thoughts:
        thought_stream = ThoughtStream(enabled=True, max_chars=THOUGHTS_MAX_CHARS)
        thought_stream.emit("info", "Intermediate steps:")
        callbacks = [ThoughtCallbackHandler(thought_stream)]
    resp = _invoke_with_retries(agent, question, callbacks=callbacks)
    answer = resp.get("output") if isinstance(resp, dict) else resp
    if return_thoughts:
        return answer, (thought_stream.events if thought_stream else [])
    return answer


def _try_local_answer(df: pd.DataFrame, question: str) -> Optional[str]:
    q = question.strip().lower()
    stat_map = {
        "mean": "mean",
        "average": "mean",
        "avg": "mean",
        "median": "median",
        "sum": "sum",
        "total": "sum",
        "count": "count",
        "min": "min",
        "minimum": "min",
        "max": "max",
        "maximum": "max",
    }

    match = re.search(
        r"(mean|average|avg|median|sum|total|count|min|minimum|max|maximum)\s+value\s+of\s+the\s+['\"]?([\w\s\-]+)['\"]?\s+column",
        q,
    )
    if not match:
        match = re.search(
            r"(mean|average|avg|median|sum|total|count|min|minimum|max|maximum)\s+of\s+['\"]?([\w\s\-]+)['\"]?",
            q,
        )
    if not match:
        return None

    stat_key = match.group(1)
    col_raw = match.group(2).strip()
    stat = stat_map.get(stat_key)
    if not stat:
        return None

    # Resolve column name case-insensitively
    col_name = None
    for col in df.columns:
        if str(col).strip().lower() == col_raw:
            col_name = col
            break
    if col_name is None:
        return f"Column not found: {col_raw}"

    series = pd.to_numeric(df[col_name], errors="coerce")
    if series.notna().sum() == 0:
        return f"Column '{col_name}' has no numeric values to compute {stat}."

    value = getattr(series, stat)()
    return f"{stat}({col_name}) = {value}"


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


def run_security_audit(df: pd.DataFrame, sandbox: Optional[Sandbox]) -> Dict[str, Any]:
    if sandbox is None:
        return {"error": "Sandbox is disabled; enable SANDBOX_ENABLED=true to run the audit."}

    outside_target = os.path.abspath(CSV_PATH)
    inside_target = os.path.join(sandbox.base_dir, "audit_canary.txt")

    # Prepare a canary file inside the sandbox.
    os.makedirs(sandbox.base_dir, exist_ok=True)
    with open(inside_target, "w", encoding="utf-8") as handle:
        handle.write("audit")

    tool = _build_sandboxed_repl(df, sandbox)
    payloads = {
        "direct_os_remove": f"import os; os.remove(r'{outside_target}')",
        "pandas_os_remove": f"pd.io.common.os.remove(r'{outside_target}')",
        "open_outside": f"open(r'{outside_target}', 'rb').read(8)",
    }

    results: Dict[str, Any] = {}
    for name, code in payloads.items():
        try:
            results[name] = sandbox.run(tool.run, code)
        except Exception as exc:
            results[name] = f"{type(exc).__name__}: {exc}"

    results["outside_exists"] = os.path.exists(outside_target)
    results["inside_exists"] = os.path.exists(inside_target)
    return results


# =========================
# CLI
# =========================
def main():
    df = _load_dataframe(CSV_PATH)
    print("Columns:", df.columns.tolist())
    print("Shape:", df.shape)

    # Build agent for Q&A
    agent = None
    sandbox = _init_sandbox()
    if ALLOW_LLM:
        _require_api_key()
        agent = create_agent(df, sandbox)

    print("Commands:")
    print("  ask <your question>            -> LLM Q&A about the dataframe")
    print("  plot bar <x> <y> [mean|sum|count|median] <filename.png>  -> bar chart (optional agg)")
    print("  plot hist <x> <bins> <filename.png>                       -> histogram")
    print("  plot line <x> <y> <filename.png>                          -> line chart")
    print("  plot scatter <x> <y> <filename.png>                       -> scatter")
    print("  plot box <x> <y> <filename.png>                           -> box plot")
    print("  Tip: wrap column names with spaces in quotes, e.g. plot box \"Avg Daily Distance (km)\" Brand out.png")
    print("  audit                            -> run sandbox security audit")
    print(f"Thought streaming: {'on' if STREAM_THOUGHTS else 'off'} (set STREAM_THOUGHTS=false to disable)")
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
            if not agent:
                local = _try_local_answer(df, question)
                if local is not None:
                    print(local)
                else:
                    print("LLM Q&A is disabled. Set ALLOW_LLM=true and ALLOW_DANGEROUS_CODE=true to enable.")
                continue
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

        if raw.lower() == "audit":
            results = run_security_audit(df, sandbox)
            print("Security audit results:")
            for key, value in results.items():
                print(f"  {key}: {value}")
            continue

        print("Unknown command. Start with 'ask ' or 'plot ' as shown in usage.")

if __name__ == "__main__":
    main()

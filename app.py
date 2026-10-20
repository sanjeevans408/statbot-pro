"""StatBot Pro - Autonomous Multi-Format Data Workspace & Visualization Studio.

Features:
- Advanced multi-format dataset engine (CSV, TSV, Excel, JSON, Parquet, Arrow).
- Multi-chart interactive studio with 15+ chart types and Pinned Multi-Chart Canvas.
- Automated multi-chart executive insights gallery.
- Interactive data transformation and cleaning lab.
- Dual-mode conversational analyst (offline smart NLP + LangChain ChatNVIDIA agent).
- Polished, responsive executive UI/UX design.
"""
from __future__ import annotations

import datetime
import io
import os
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.io as pio
import streamlit as st

import analyst_engine
import chart_studio
import data_loader

# ==============================================================================
# Page Configuration & Global Theme
# ==============================================================================
st.set_page_config(
    page_title="StatBot Pro · Autonomous Data Workspace",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Styling
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
  }
  
  .stApp {
    background-color: #f8fafc;
    color: #0f172a;
  }

  .block-container {
    max-width: 1480px;
    padding-top: 1.6rem;
    padding-bottom: 3.5rem;
  }

  /* Sidebar Styling */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1120 0%, #0f172a 100%);
    border-right: 1px solid #1e293b;
  }
  [data-testid="stSidebar"] * {
    color: #f1f5f9;
  }
  [data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] .stCaption {
    color: #94a3b8;
  }

  /* Hero Header */
  .hero-card {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #1e293b 100%);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 20px;
    padding: 2rem 2.4rem;
    color: white;
    margin-bottom: 1.6rem;
    box-shadow: 0 16px 36px -10px rgba(15, 23, 42, 0.25);
    position: relative;
    overflow: hidden;
  }
  .hero-card::after {
    content: "";
    position: absolute;
    top: -50%;
    right: -10%;
    width: 380px;
    height: 380px;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.25) 0%, transparent 70%);
    pointer-events: none;
  }
  .hero-eyebrow {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #818cf8;
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .hero-title {
    font-size: 2.15rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    margin: 0 0 0.4rem 0;
    color: #ffffff;
  }
  .hero-subtitle {
    font-size: 0.98rem;
    color: #cbd5e1;
    margin: 0;
    max-width: 780px;
    line-height: 1.5;
  }

  /* Metric KPI Cards */
  div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
  }
  div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
  }
  div[data-testid="stMetric"] label {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #64748b !important;
  }
  div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-size: 1.65rem !important;
    font-weight: 750 !important;
    color: #0f172a !important;
  }

  /* Card Containers */
  .panel-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.3rem;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.03);
  }
  .panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.9rem;
    border-bottom: 1px solid #f1f5f9;
    padding-bottom: 0.6rem;
  }
  .panel-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #0f172a;
    margin: 0;
  }

  /* Badges & Tags */
  .status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.22rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 600;
  }
  .badge-success { background: #dcfce7; color: #15803d; }
  .badge-primary { background: #e0e7ff; color: #4338ca; }
  .badge-warning { background: #fef3c7; color: #b45309; }
  .badge-info { background: #e0f2fe; color: #0369a1; }

  /* Tab Nav Polish */
  .stTabs [data-baseweb="tab-list"] {
    gap: 0.6rem;
    border-bottom: 2px solid #e2e8f0;
    margin-bottom: 1.2rem;
  }
  .stTabs [data-baseweb="tab"] {
    font-size: 0.92rem;
    font-weight: 600;
    color: #64748b;
    padding: 0.65rem 1.1rem;
    border-radius: 8px 8px 0 0;
  }
  .stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #4338ca !important;
    border-bottom: 2.5px solid #4338ca;
  }

  /* Buttons */
  .stButton > button {
    border-radius: 10px;
    font-weight: 650;
    padding: 0.55rem 1.2rem;
    transition: all 0.15s ease;
  }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Session State Initialization
# ==============================================================================
if "pinned_charts" not in st.session_state:
    st.session_state["pinned_charts"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "cleaned_df" not in st.session_state:
    st.session_state["cleaned_df"] = None


# ==============================================================================
# Sidebar - Dataset Ingestion & Configuration
# ==============================================================================
with st.sidebar:
    st.markdown("<h2 style='margin:0; font-weight:800; letter-spacing:-0.03em;'>◈ StatBot Pro</h2>", unsafe_allow_html=True)
    st.caption("AUTONOMOUS MULTI-DATASET WORKSPACE")
    st.divider()

    st.markdown("#### 📁 Ingest Dataset")
    st.markdown(
        "<span style='font-size:0.75rem; color:#94a3b8;'>Supports <b>CSV, TSV, Excel (.xlsx/.xls), JSON, JSONL, Parquet, Arrow</b></span>",
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Upload dataset file",
        type=["csv", "tsv", "xlsx", "xls", "json", "jsonl", "parquet", "pq", "feather", "arrow"],
        help="Upload any structured data file. Encodings and delimiters are auto-detected.",
    )

    selected_sheet: Optional[str] = None
    if uploaded_file is not None and uploaded_file.name.endswith((".xlsx", ".xls", ".xlsm")):
        sheets = data_loader.get_excel_sheet_names(uploaded_file.getvalue())
        if len(sheets) > 1:
            selected_sheet = st.selectbox("📑 Select Excel Worksheet", sheets)
        else:
            selected_sheet = sheets[0] if sheets else None

    # Sample dataset selector
    sample_options = {
        "None (Use Uploaded Data)": None,
        "🏍️ Indian Used Two-Wheelers (10k rows - Vehicle Resale)": "sample.csv",
        "🛒 E-Commerce Retail Sales (1.5k rows - Orders & Profit)": "samples/ecommerce_sales.csv",
        "👥 HR & Employee Attrition (1k rows - Demographics & Salary)": "samples/hr_analytics.csv",
        "📈 Tech Stock Market Trends (750 rows - Daily Prices & Return)": "samples/stock_market_trends.csv",
    }
    
    use_sample_choice = st.selectbox(
        "Or explore a curated sample",
        options=list(sample_options.keys()),
        index=1 if uploaded_file is None else 0,
    )

    st.divider()
    with st.expander("⚙️ AI Analyst Settings (Optional)"):
        user_api_key = st.text_input(
            "NVIDIA API Key",
            type="password",
            value=os.getenv("NVIDIA_API_KEY", ""),
            help="Begins with nvapi-... Enables autonomous LLM dataframe reasoning. StatBot also has a built-in offline NLP engine.",
        )
        selected_model = st.selectbox(
            "Model Selection",
            ["deepseek-ai/deepseek-v3.2", "meta/llama-3.1-70b-instruct", "mistralai/mistral-large-2-instruct"],
            index=0,
        )
        if user_api_key:
            os.environ["NVIDIA_API_KEY"] = user_api_key
            os.environ["NVIDIA_MODEL"] = selected_model
            st.success("API Key Active")

    st.divider()
    st.markdown(
        "<div style='font-size:0.73rem; color:#64748b; line-height:1.4;'>"
        "🔒 <b>Local Isolation:</b> Data stays within your current session. "
        "Offline analysis runs entirely in-memory."
        "</div>",
        unsafe_allow_html=True,
    )


# ==============================================================================
# Data Loading & Verification Pipeline
# ==============================================================================
raw_df: Optional[pd.DataFrame] = None
active_source_name = "Untitled Dataset"
active_format = "CSV"

try:
    if uploaded_file is not None:
        active_source_name = uploaded_file.name
        active_format = os.path.splitext(uploaded_file.name)[1].upper().replace(".", "")
        raw_df = data_loader.load_dataset(
            uploaded_file.getvalue(),
            filename=uploaded_file.name,
            sheet_name=selected_sheet,
        )
    elif sample_options[use_sample_choice] is not None:
        sample_path = sample_options[use_sample_choice]
        active_source_name = os.path.basename(sample_path)
        active_format = os.path.splitext(sample_path)[1].upper().replace(".", "")
        raw_df = data_loader.load_dataset(sample_path)
    else:
        st.markdown(
            """
            <div class='hero-card'>
                <div class='hero-eyebrow'>◈ StatBot Pro Workspace</div>
                <h1 class='hero-title'>Welcome to StatBot Pro.</h1>
                <p class='hero-subtitle'>
                    Upload any CSV, Excel, Parquet, or JSON dataset from the left sidebar, 
                    or choose one of our diverse pre-loaded samples to start exploring multiple graphs, 
                    profiling data health, and asking questions.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()
except Exception as err:
    st.error(f"❌ Failed to load dataset: {err}")
    st.stop()

# Set current active dataframe (considering any transformations applied)
active_df = st.session_state["cleaned_df"] if st.session_state["cleaned_df"] is not None else raw_df
summary = data_loader.get_dataset_summary(active_df)


# ==============================================================================
# Hero Workspace Banner
# ==============================================================================
st.markdown(
    f"""
    <div class='hero-card'>
        <div class='hero-eyebrow'>
            <span>◈ ACTIVE WORKSPACE</span>
            <span>·</span>
            <span>FORMAT: {active_format}</span>
            <span>·</span>
            <span class='status-badge badge-success'>HEALTH {summary['quality_score']}%</span>
        </div>
        <h1 class='hero-title'>{active_source_name}</h1>
        <p class='hero-subtitle'>
            Ready for advanced multi-chart intelligence, side-by-side visualization canvases, 
            data cleaning, and autonomous statistical Q&A.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Metric KPI Header
kpi_cols = st.columns(5)
kpi_cols[0].metric("Total Rows", f"{summary['rows']:,}")
kpi_cols[1].metric("Features (Cols)", f"{summary['columns']:,}")
kpi_cols[2].metric("Missing Cells", f"{summary['missing_cells']:,}", f"{summary['missing_percentage']}%", delta_color="inverse")
kpi_cols[3].metric("Duplicate Rows", f"{summary['duplicate_rows']:,}", f"{summary['duplicate_percentage']}%", delta_color="inverse")
kpi_cols[4].metric("Memory Footprint", summary["memory_usage"])


# ==============================================================================
# Main Workspace Navigation Tabs
# ==============================================================================
tab_overview, tab_studio, tab_auto, tab_clean, tab_analyst = st.tabs([
    "📊 Executive Overview",
    "📈 Multi-Chart Studio",
    "⚡ Auto-Insights & Gallery",
    "🛠️ Data Cleaning Lab",
    "💬 StatBot AI Analyst",
])


# ==============================================================================
# TAB 1: Executive Overview & Column Diagnostics
# ==============================================================================
with tab_overview:
    col_prev, col_diag = st.columns([1.6, 1.0])
    with col_prev:
        st.markdown("##### 🔍 Interactive Data Preview")
        preview_cols = st.multiselect(
            "Visible columns",
            options=active_df.columns.tolist(),
            default=active_df.columns[: min(10, len(active_df.columns))].tolist(),
            label_visibility="collapsed",
        )
        display_df = active_df[preview_cols] if preview_cols else active_df
        st.dataframe(display_df.head(100), use_container_width=True, height=410)

    with col_diag:
        st.markdown("##### 🩺 Column Health & Types")
        profile_df = data_loader.get_column_profiles(active_df)
        st.dataframe(profile_df, use_container_width=True, height=410, hide_index=True)

    # Statistical Distribution Table for Numericals
    numeric_cols = summary["numeric_cols"]
    if numeric_cols:
        with st.expander("📐 Quantitative Descriptive Statistics", expanded=False):
            desc_df = active_df[numeric_cols].describe().round(2).T
            st.dataframe(desc_df, use_container_width=True)

    # Quick Export Button
    csv_bytes, fname, mime = data_loader.export_dataframe(active_df, "csv")
    st.download_button(
        label="📥 Download Active Dataset (CSV)",
        data=csv_bytes,
        file_name=f"statbot_{active_source_name}.csv",
        mime=mime,
    )


# ==============================================================================
# TAB 2: Multi-Chart Studio & Pinned Canvas
# ==============================================================================
with tab_studio:
    st.markdown("##### 🎨 Interactive Chart Builder")
    st.caption("Compose custom charts across 15+ visualization types, fine-tune dimensions, and pin multiple graphs to your dashboard.")

    studio_col1, studio_col2, studio_col3, studio_col4 = st.columns([1.1, 1, 1, 1])

    with studio_col1:
        chart_type = st.selectbox(
            "Chart Type",
            [
                "Bar Chart",
                "Line Chart",
                "Area Chart",
                "Scatter Plot",
                "3D Scatter Plot",
                "Bubble Chart",
                "Histogram & Density",
                "Box Plot",
                "Violin Plot",
                "Donut / Pie",
                "Treemap",
                "Correlation Heatmap",
                "Density Contour",
                "Radar / Spider",
                "Funnel Chart",
            ],
            index=0,
        )

    all_cols = active_df.columns.tolist()
    num_cols = summary["numeric_cols"]
    cat_cols = summary["categorical_cols"]

    with studio_col2:
        x_default = cat_cols[0] if cat_cols else (all_cols[0] if all_cols else None)
        x_col = st.selectbox("Primary Axis (X)", options=all_cols, index=all_cols.index(x_default) if x_default in all_cols else 0)

    with studio_col3:
        y_default = num_cols[0] if num_cols else (all_cols[min(1, len(all_cols)-1)] if all_cols else None)
        y_col = st.selectbox("Metric Axis (Y)", options=[None] + all_cols, index=([None] + all_cols).index(y_default) if y_default in all_cols else 0)

    with studio_col4:
        agg_choice = st.selectbox("Aggregation", ["Mean", "Sum", "Count", "Median", "Min", "Max"], index=0)

    # Secondary Parameters Accordion
    with st.expander("🛠️ Advanced Visual Dimensions (Color, Size, 3D, Themes)", expanded=False):
        adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)
        with adv_col1:
            color_dim = st.selectbox("Color / Legend Group", options=[None] + all_cols, index=0)
        with adv_col2:
            size_dim = st.selectbox("Size Dimension", options=[None] + num_cols, index=0)
        with adv_col3:
            z_dim = st.selectbox("Z Axis (for 3D Scatter)", options=[None] + num_cols, index=0)
        with adv_col4:
            palette = st.selectbox("Color Palette", list(chart_studio.PALETTES.keys()), index=0)

        adv_sub1, adv_sub2, adv_sub3, adv_sub4 = st.columns(4)
        with adv_sub1:
            theme = st.selectbox("Plotly Theme", list(chart_studio.THEMES.keys()), index=0)
        with adv_sub2:
            top_n = st.slider("Top N Categories", min_value=5, max_value=50, value=15, step=5)
        with adv_sub3:
            show_trend = st.checkbox("OLS Trendline (Scatter)", value=False)
            log_y = st.checkbox("Logarithmic Y Axis", value=False)
        with adv_sub4:
            custom_title = st.text_input("Custom Title", placeholder="Auto-generated if empty")

    # Render Active Visual
    try:
        active_fig = chart_studio.build_figure(
            df=active_df,
            chart_type=chart_type,
            x_col=x_col,
            y_col=y_col,
            z_col=z_dim,
            color_col=color_dim,
            size_col=size_dim,
            agg_func=agg_choice,
            top_n=top_n,
            title=custom_title if custom_title.strip() else None,
            palette_name=palette,
            theme_name=theme,
            show_trendline=show_trend,
            log_y=log_y,
        )

        st.plotly_chart(active_fig, use_container_width=True)

        # Pin to Canvas & Export controls
        btn_c1, btn_c2, btn_c3 = st.columns([1.4, 1.4, 4.0])
        with btn_c1:
            if st.button("📌 Pin to Multi-Chart Canvas", type="primary"):
                chart_record = {
                    "id": str(uuid.uuid4())[:8],
                    "title": custom_title.strip() or f"{chart_type}: {y_col or ''} by {x_col or ''}",
                    "figure": active_fig,
                    "created_at": datetime.datetime.now().strftime("%H:%M:%S"),
                }
                st.session_state["pinned_charts"].append(chart_record)
                st.success("Chart pinned to canvas below!")
                st.rerun()

        with btn_c2:
            html_bytes = pio.to_html(active_fig, include_plotlyjs="cdn").encode("utf-8")
            st.download_button(
                "📥 Export Interactive HTML",
                data=html_bytes,
                file_name=f"{chart_type.lower().replace(' ', '_')}.html",
                mime="text/html",
            )
    except Exception as exc:
        st.warning(f"⚠️ Could not render visual with current settings: {exc}")

    # ==========================================================================
    # Pinned Multi-Chart Canvas (Multiple Graphs & Charts Grid)
    # ==========================================================================
    st.divider()
    canvas_header, canvas_ctrl = st.columns([3, 1])
    with canvas_header:
        st.markdown(f"#### 📌 Pinned Multi-Chart Canvas ({len(st.session_state['pinned_charts'])})")
        st.caption("View and compare multiple charts simultaneously side-by-side in a responsive dashboard canvas.")
    with canvas_ctrl:
        if st.session_state["pinned_charts"]:
            if st.button("🗑️ Clear Canvas"):
                st.session_state["pinned_charts"].clear()
                st.rerun()

    pinned = st.session_state["pinned_charts"]
    if not pinned:
        st.info("💡 No charts pinned yet. Customize a chart above and click **'📌 Pin to Multi-Chart Canvas'** to build your custom multi-graph dashboard.")
    else:
        # Layout toggle: 2-column or 1-column
        view_cols_setting = st.radio("Canvas Layout", ["2 Columns (Side-by-side)", "1 Column (Full Width)"], horizontal=True)
        num_grid_cols = 2 if "2 Columns" in view_cols_setting else 1

        cols = st.columns(num_grid_cols)
        for idx, chart_item in enumerate(pinned):
            col_target = cols[idx % num_grid_cols]
            with col_target:
                with st.container(border=True):
                    card_head1, card_head2 = st.columns([3, 1])
                    with card_head1:
                        st.markdown(f"**{chart_item['title']}**")
                        st.caption(f"Created at {chart_item['created_at']}")
                    with card_head2:
                        if st.button("✕ Remove", key=f"del_{chart_item['id']}"):
                            st.session_state["pinned_charts"] = [c for c in pinned if c["id"] != chart_item["id"]]
                            st.rerun()
                    st.plotly_chart(chart_item["figure"], use_container_width=True)


# ==============================================================================
# TAB 3: Auto-Insights & Multi-Chart Gallery
# ==============================================================================
with tab_auto:
    st.markdown("##### ⚡ Autonomous Multi-Chart Intelligence Gallery")
    st.caption("StatBot automatically analyzes dataset distributions, correlations, outliers, and categorical hierarchies to generate a comprehensive 6-chart executive suite.")

    with st.spinner("Generating automated analytical gallery…"):
        auto_insights = chart_studio.generate_auto_insights(active_df, max_charts=6)

    if not auto_insights:
        st.warning("Not enough numeric or categorical columns to generate automated insights.")
    else:
        # Render 2-column executive dashboard grid
        gallery_cols = st.columns(2)
        for idx, item in enumerate(auto_insights):
            target_col = gallery_cols[idx % 2]
            with target_col:
                with st.container(border=True):
                    st.markdown(f"**{item['title']}**")
                    st.caption(item["desc"])
                    st.plotly_chart(item["figure"], use_container_width=True)


# ==============================================================================
# TAB 4: Data Cleaning & Transform Lab
# ==============================================================================
with tab_clean:
    st.markdown("##### 🛠️ Data Cleaning & Preprocessing Workbench")
    st.caption("Apply transformations non-destructively: drop nulls, impute with statistical estimators, remove duplicate records, or filter cohorts.")

    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        st.markdown("**1. Missing Value Strategy**")
        impute_opt = st.selectbox(
            "Imputation Method",
            ["Keep As-Is", "Drop Rows with Any Missing Values", "Impute Numerics (Median) & Categoricals (Mode)", "Impute with Column Mode"],
            index=0,
        )
        impute_map = {
            "Keep As-Is": "none",
            "Drop Rows with Any Missing Values": "drop",
            "Impute Numerics (Median) & Categoricals (Mode)": "mean_median",
            "Impute with Column Mode": "mode",
        }

    with tc2:
        st.markdown("**2. Duplicate Deduplication**")
        drop_dups = st.checkbox("Remove Duplicate Rows", value=False)

    with tc3:
        st.markdown("**3. Column Pruning**")
        drop_cols = st.multiselect("Columns to remove", options=active_df.columns.tolist())

    # Value-based filter
    with st.expander("🔎 Row Filtering Options", expanded=False):
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            filter_target_col = st.selectbox("Filter by column", [None] + active_df.columns.tolist(), index=0)
        with f_col2:
            filter_val = None
            if filter_target_col:
                if pd.api.types.is_numeric_dtype(active_df[filter_target_col]):
                    min_val = float(active_df[filter_target_col].min())
                    max_val = float(active_df[filter_target_col].max())
                    filter_val = st.slider("Select numeric range", min_value=min_val, max_value=max_val, value=(min_val, max_val))
                else:
                    sample_vals = active_df[filter_target_col].dropna().unique().tolist()[:40]
                    filter_val = st.multiselect("Select values to keep", options=sample_vals)

    # Action buttons
    act_col1, act_col2 = st.columns([1.2, 5])
    with act_col1:
        if st.button("Apply Transformations", type="primary"):
            new_df = data_loader.apply_transformations(
                raw_df,
                drop_columns=drop_cols,
                drop_duplicates=drop_dups,
                impute_strategy=impute_map[impute_opt],
                custom_filter_col=filter_target_col,
                custom_filter_val=filter_val,
            )
            st.session_state["cleaned_df"] = new_df
            st.success(f"Transformed: {len(new_df):,} rows remaining ({len(raw_df) - len(new_df)} removed).")
            st.rerun()

    with act_col2:
        if st.session_state["cleaned_df"] is not None:
            if st.button("↺ Reset to Original Raw Data"):
                st.session_state["cleaned_df"] = None
                st.rerun()

    st.divider()
    st.markdown("##### 📥 Export Cleaned Dataset")
    exp_c1, exp_c2, exp_c3, exp_c4 = st.columns(4)
    with exp_c1:
        b_csv, f_csv, m_csv = data_loader.export_dataframe(active_df, "csv")
        st.download_button("Download CSV", data=b_csv, file_name=f_csv, mime=m_csv)
    with exp_c2:
        b_xlsx, f_xlsx, m_xlsx = data_loader.export_dataframe(active_df, "excel")
        st.download_button("Download Excel (.xlsx)", data=b_xlsx, file_name=f_xlsx, mime=m_xlsx)
    with exp_c3:
        b_json, f_json, m_json = data_loader.export_dataframe(active_df, "json")
        st.download_button("Download JSON", data=b_json, file_name=f_json, mime=m_json)
    with exp_c4:
        b_pq, f_pq, m_pq = data_loader.export_dataframe(active_df, "parquet")
        st.download_button("Download Parquet", data=b_pq, file_name=f_pq, mime=m_pq)


# ==============================================================================
# TAB 5: StatBot AI Analyst (Natural Language Q&A)
# ==============================================================================
with tab_analyst:
    st.markdown("##### 💬 Autonomous Data Analyst")
    st.caption("Ask questions in plain English. StatBot computes exact mathematical results locally or utilizes NVIDIA LLM agents when an API key is provided.")

    # Suggestion Chips
    st.markdown("<span style='font-size:0.8rem; font-weight:600; color:#64748b;'>SUGGESTED QUESTIONS:</span>", unsafe_allow_html=True)
    chip_cols = st.columns(4)
    suggested_q = None
    if chip_cols[0].button("📊 Average of primary metric"):
        if summary["numeric_cols"]:
            suggested_q = f"What is the average {summary['numeric_cols'][0]}?"
    if chip_cols[1].button("🏆 Top 5 categories"):
        if summary["categorical_cols"] and summary["numeric_cols"]:
            suggested_q = f"Top 5 {summary['categorical_cols'][0]} by {summary['numeric_cols'][0]}"
    if chip_cols[2].button("🔗 Correlation between variables"):
        if len(summary["numeric_cols"]) >= 2:
            suggested_q = f"Correlation between {summary['numeric_cols'][0]} and {summary['numeric_cols'][1]}"
    if chip_cols[3].button("📋 Total row & column count"):
        suggested_q = "How many rows and columns are in this dataset?"

    query_input = st.text_input(
        "Ask a question about the dataset",
        value=suggested_q or "",
        placeholder="e.g., What is the average price? Or Top 5 brands by resale value...",
        key="analyst_query_input",
    )

    if st.button("Submit Question", type="primary", disabled=not query_input.strip()):
        with st.spinner("StatBot is analyzing the dataset…"):
            answer = analyst_engine.run_analyst_query(
                active_df,
                query_input,
                api_key=os.getenv("NVIDIA_API_KEY"),
            )
            st.session_state["chat_history"].append({"q": query_input, "a": answer})

    # Render History
    if st.session_state["chat_history"]:
        st.divider()
        st.markdown("##### 📜 Recent Analyses")
        for chat in reversed(st.session_state["chat_history"]):
            with st.chat_message("user"):
                st.markdown(f"**{chat['q']}**")
            with st.chat_message("assistant"):
                st.markdown(chat["a"])

st.divider()
st.caption("StatBot Pro Enterprise Edition · Autonomous Data Workspace & Multi-Chart Intelligence")

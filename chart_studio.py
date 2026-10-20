"""StatBot Pro - Advanced Visualization Studio & Multi-Chart Engine.

Generates 16+ interactive Plotly charts, automated multi-chart analytics galleries,
multi-chart canvas management, and customizable theme palettes.
"""
from __future__ import annotations

import io
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PALETTES = {
    "StatBot Indigo": ["#4f46e5", "#06b6d4", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6", "#3b82f6", "#14b8a6"],
    "Cyberpunk Neon": ["#00f0ff", "#ff007f", "#ffe600", "#7928ca", "#00dfa2", "#ff5555", "#bd93f9", "#50fa7b"],
    "Deep Ocean": ["#1e3a8a", "#2563eb", "#0284c7", "#06b6d4", "#0d9488", "#10b981", "#3b82f6", "#60a5fa"],
    "Sunset Warmth": ["#e11d48", "#f43f5e", "#fb923c", "#f59e0b", "#eab308", "#84cc16", "#10b981", "#06b6d4"],
    "Emerald Luxe": ["#064e3b", "#047857", "#059669", "#10b981", "#34d399", "#6ee7b7", "#a7f3d0", "#2dd4bf"],
    "Monochrome Slate": ["#0f172a", "#334155", "#475569", "#64748b", "#94a3b8", "#cbd5e1", "#e2e8f0", "#1e293b"],
}

THEMES = {
    "Clean White": "plotly_white",
    "Modern Dark": "plotly_dark",
    "Presentation": "presentation",
    "Soft Minimal": "seaborn",
    "Classic": "ggplot2",
}


def build_figure(
    df: pd.DataFrame,
    chart_type: str,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    z_col: Optional[str] = None,
    color_col: Optional[str] = None,
    size_col: Optional[str] = None,
    agg_func: str = "mean",
    bins: int = 30,
    top_n: int = 15,
    title: Optional[str] = None,
    palette_name: str = "StatBot Indigo",
    theme_name: str = "Clean White",
    show_trendline: bool = False,
    donut_hole: float = 0.45,
    log_x: bool = False,
    log_y: bool = False,
) -> go.Figure:
    """Build a rich interactive Plotly figure based on user specifications."""
    colors = PALETTES.get(palette_name, PALETTES["StatBot Indigo"])
    template = THEMES.get(theme_name, "plotly_white")

    # Clean working copy
    plot_df = df.copy()

    # Pre-filter high-cardinality x_col for categorical aggregations
    if chart_type in ["Bar Chart", "Donut / Pie", "Treemap", "Funnel Chart"] and x_col:
        val_counts = plot_df[x_col].value_counts()
        if len(val_counts) > top_n and top_n > 0:
            top_keys = val_counts.nlargest(top_n).index
            plot_df = plot_df[plot_df[x_col].isin(top_keys)]

    fig: Optional[go.Figure] = None

    # 1. Bar Chart
    if chart_type == "Bar Chart":
        if not x_col or not y_col:
            raise ValueError("Bar Chart requires both an X column and a Y column.")
        
        # Check aggregation
        agg_map = {"mean": "mean", "sum": "sum", "count": "count", "median": "median", "min": "min", "max": "max"}
        op = agg_map.get(agg_func.lower(), "mean")

        group_cols = [x_col]
        if color_col and color_col != x_col and color_col in plot_df.columns:
            group_cols.append(color_col)

        if op == "count":
            grouped = plot_df.groupby(group_cols, as_index=False).size()
            grouped.rename(columns={"size": "Count"}, inplace=True)
            val_col = "Count"
        else:
            plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
            grouped = plot_df.dropna(subset=[y_col]).groupby(group_cols, as_index=False)[y_col].agg(op)
            grouped[y_col] = grouped[y_col].round(2)
            val_col = y_col

        grouped = grouped.sort_values(by=val_col, ascending=False)
        fig = px.bar(
            grouped,
            x=x_col,
            y=val_col,
            color=color_col if color_col in grouped.columns else None,
            barmode="group",
            color_discrete_sequence=colors,
            title=title or f"{agg_func.title()} {y_col} by {x_col}",
            template=template,
        )

    # 2. Line Chart
    elif chart_type == "Line Chart":
        if not x_col or not y_col:
            raise ValueError("Line Chart requires an X column (timeline or category) and Y column.")
        
        # Sort by x if possible
        clean_line_df = plot_df.dropna(subset=[x_col, y_col]).sort_values(by=x_col)
        fig = px.line(
            clean_line_df,
            x=x_col,
            y=y_col,
            color=color_col if color_col in clean_line_df.columns else None,
            markers=True,
            color_discrete_sequence=colors,
            title=title or f"{y_col} trend over {x_col}",
            template=template,
        )

    # 3. Area Chart
    elif chart_type == "Area Chart":
        if not x_col or not y_col:
            raise ValueError("Area Chart requires an X column and a Y column.")
        clean_area_df = plot_df.dropna(subset=[x_col, y_col]).sort_values(by=x_col)
        fig = px.area(
            clean_area_df,
            x=x_col,
            y=y_col,
            color=color_col if color_col in clean_area_df.columns else None,
            color_discrete_sequence=colors,
            title=title or f"Cumulative {y_col} across {x_col}",
            template=template,
        )

    # 4. Scatter Plot
    elif chart_type == "Scatter Plot":
        if not x_col or not y_col:
            raise ValueError("Scatter Plot requires both X and Y columns.")
        clean_scatter_df = plot_df.dropna(subset=[x_col, y_col])
        trend_mode = "ols" if show_trendline and pd.api.types.is_numeric_dtype(clean_scatter_df[x_col]) else None
        
        fig = px.scatter(
            clean_scatter_df,
            x=x_col,
            y=y_col,
            color=color_col if color_col in clean_scatter_df.columns else None,
            size=size_col if size_col in clean_scatter_df.columns and pd.api.types.is_numeric_dtype(clean_scatter_df[size_col]) else None,
            trendline=trend_mode,
            color_discrete_sequence=colors,
            title=title or f"Correlation: {x_col} vs {y_col}",
            template=template,
            opacity=0.75,
        )

    # 5. 3D Scatter Plot
    elif chart_type == "3D Scatter Plot":
        if not x_col or not y_col or not z_col:
            raise ValueError("3D Scatter Plot requires X, Y, and Z numeric columns.")
        clean_3d_df = plot_df.dropna(subset=[x_col, y_col, z_col])
        fig = px.scatter_3d(
            clean_3d_df,
            x=x_col,
            y=y_col,
            z=z_col,
            color=color_col if color_col in clean_3d_df.columns else None,
            size=size_col if size_col in clean_3d_df.columns else None,
            color_discrete_sequence=colors,
            title=title or f"3D Space: {x_col} × {y_col} × {z_col}",
            template=template,
            opacity=0.8,
        )

    # 6. Bubble Chart
    elif chart_type == "Bubble Chart":
        if not x_col or not y_col:
            raise ValueError("Bubble Chart requires X, Y, and a Size column.")
        clean_bubble = plot_df.dropna(subset=[x_col, y_col])
        sz = size_col if size_col and size_col in clean_bubble.columns else y_col
        fig = px.scatter(
            clean_bubble,
            x=x_col,
            y=y_col,
            size=sz,
            color=color_col if color_col in clean_bubble.columns else None,
            color_discrete_sequence=colors,
            title=title or f"Bubble Map: {x_col} vs {y_col} (Size: {sz})",
            template=template,
        )

    # 7. Histogram & Density
    elif chart_type == "Histogram & Density":
        target = x_col or y_col
        if not target:
            raise ValueError("Histogram requires at least one column.")
        clean_hist = plot_df.dropna(subset=[target])
        fig = px.histogram(
            clean_hist,
            x=target,
            nbins=bins,
            marginal="box",
            color=color_col if color_col in clean_hist.columns else None,
            color_discrete_sequence=colors,
            title=title or f"Distribution Profile: {target}",
            template=template,
            opacity=0.85,
        )

    # 8. Box Plot
    elif chart_type == "Box Plot":
        target_y = y_col or x_col
        if not target_y:
            raise ValueError("Box Plot requires a numeric value column.")
        clean_box = plot_df.dropna(subset=[target_y])
        fig = px.box(
            clean_box,
            x=x_col if x_col != target_y else None,
            y=target_y,
            color=color_col if color_col in clean_box.columns else (x_col if x_col != target_y else None),
            points="outliers",
            color_discrete_sequence=colors,
            title=title or f"Spread & Outlier Detection: {target_y}",
            template=template,
        )

    # 9. Violin Plot
    elif chart_type == "Violin Plot":
        target_y = y_col or x_col
        if not target_y:
            raise ValueError("Violin Plot requires a numeric value column.")
        clean_violin = plot_df.dropna(subset=[target_y])
        fig = px.violin(
            clean_violin,
            x=x_col if x_col != target_y else None,
            y=target_y,
            box=True,
            points="all",
            color=color_col if color_col in clean_violin.columns else (x_col if x_col != target_y else None),
            color_discrete_sequence=colors,
            title=title or f"Density Violin: {target_y}",
            template=template,
        )

    # 10. Donut / Pie Chart
    elif chart_type == "Donut / Pie":
        cat_col = x_col or color_col
        if not cat_col:
            raise ValueError("Donut / Pie chart requires a categorical column.")
        
        counts = plot_df[cat_col].value_counts().nlargest(top_n).reset_index()
        counts.columns = [cat_col, "Count"]
        fig = px.pie(
            counts,
            names=cat_col,
            values="Count",
            hole=donut_hole,
            color_discrete_sequence=colors,
            title=title or f"Proportions of {cat_col}",
            template=template,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")

    # 11. Treemap
    elif chart_type == "Treemap":
        cat_cols = [c for c in [x_col, color_col] if c and c in plot_df.columns]
        if not cat_cols:
            raise ValueError("Treemap requires at least one categorical column.")
        fig = px.treemap(
            plot_df,
            path=cat_cols,
            values=y_col if y_col and pd.api.types.is_numeric_dtype(plot_df[y_col]) else None,
            color_discrete_sequence=colors,
            title=title or f"Hierarchical Share: {' > '.join(cat_cols)}",
            template=template,
        )

    # 12. Correlation Heatmap
    elif chart_type == "Correlation Heatmap":
        numeric_cols = plot_df.select_dtypes(include="number").columns.tolist()
        if len(numeric_cols) < 2:
            raise ValueError("Correlation Heatmap requires at least two numeric columns.")
        corr_matrix = plot_df[numeric_cols].corr(numeric_only=True).round(2)
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r" if palette_name != "Cyberpunk Neon" else "Viridis",
            zmin=-1,
            zmax=1,
            title=title or "Pearson Correlation Matrix",
            template=template,
        )

    # 13. Density Contour (2D Heatmap)
    elif chart_type == "Density Contour":
        if not x_col or not y_col:
            raise ValueError("Density Contour requires both X and Y columns.")
        clean_density = plot_df.dropna(subset=[x_col, y_col])
        fig = px.density_heatmap(
            clean_density,
            x=x_col,
            y=y_col,
            marginal_x="histogram",
            marginal_y="histogram",
            title=title or f"2D Bivariate Density: {x_col} & {y_col}",
            template=template,
        )

    # 14. Radar / Spider Chart
    elif chart_type == "Radar / Spider":
        cat_col = x_col
        numeric_cols = plot_df.select_dtypes(include="number").columns.tolist()[:6]
        if not cat_col or not numeric_cols:
            raise ValueError("Radar chart requires a category column and numeric dimensions.")
        
        # Aggregate mean per category for top 5 categories
        top_cats = plot_df[cat_col].value_counts().nlargest(5).index
        radar_df = plot_df[plot_df[cat_col].isin(top_cats)].groupby(cat_col)[numeric_cols].mean()
        
        # Normalize 0-1 for radar display
        normalized = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min() + 1e-9)
        
        fig = go.Figure(template=template)
        for idx, (cat_name, row) in enumerate(normalized.iterrows()):
            color = colors[idx % len(colors)]
            fig.add_trace(go.Scatterpolar(
                r=row.values.tolist() + [row.values[0]],
                theta=numeric_cols + [numeric_cols[0]],
                fill="toself",
                name=str(cat_name),
                line=dict(color=color),
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title=title or f"Multi-Variable Radar Comparison across {cat_col}",
            showlegend=True,
        )

    # 15. Funnel Chart
    elif chart_type == "Funnel Chart":
        cat_col = x_col or color_col
        if not cat_col:
            raise ValueError("Funnel Chart requires a category column.")
        counts = plot_df[cat_col].value_counts().nlargest(top_n).reset_index()
        counts.columns = [cat_col, "Count"]
        fig = px.funnel(
            counts,
            y=cat_col,
            x="Count",
            color_discrete_sequence=colors,
            title=title or f"Funnel Progression: {cat_col}",
            template=template,
        )

    if fig is None:
        raise ValueError(f"Unsupported chart type: {chart_type}")

    # Layout enhancements
    fig.update_layout(
        margin=dict(l=40, r=40, t=55, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="rgba(255, 255, 255, 0.95)", font_size=12),
    )

    if log_x:
        fig.update_xaxes(type="log")
    if log_y:
        fig.update_yaxes(type="log")

    return fig


def generate_auto_insights(df: pd.DataFrame, max_charts: int = 6) -> List[Dict[str, Any]]:
    """Automatically discover key patterns and compile an executive multi-chart dashboard."""
    insights: List[Dict[str, Any]] = []
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    # 1. Primary Metric Distribution (Histogram + Box)
    if numeric_cols:
        primary_num = numeric_cols[0]
        # Prefer numeric column with standard deviations
        best_num = max(numeric_cols, key=lambda c: df[c].std() if df[c].std() == df[c].std() else 0)
        fig_dist = px.histogram(
            df,
            x=best_num,
            marginal="box",
            nbins=32,
            title=f"Core Distribution: {best_num}",
            color_discrete_sequence=["#4f46e5"],
            template="plotly_white",
        )
        fig_dist.update_layout(margin=dict(l=30, r=30, t=50, b=30))
        insights.append({
            "title": f"Distribution Profile: {best_num}",
            "desc": f"Examines frequency density, central tendency (mean: {df[best_num].mean():.2f}), and outliers.",
            "figure": fig_dist,
            "type": "distribution",
        })

    # 2. Key Category Performance (Top Bar Chart)
    if categorical_cols and numeric_cols:
        cat = categorical_cols[0]
        num = numeric_cols[0]
        grouped = df.groupby(cat, as_index=False)[num].mean().sort_values(by=num, ascending=False).head(12)
        fig_bar = px.bar(
            grouped,
            x=cat,
            y=num,
            title=f"Average {num} by {cat}",
            color=num,
            color_continuous_scale="Blues",
            template="plotly_white",
        )
        fig_bar.update_layout(margin=dict(l=30, r=30, t=50, b=30))
        insights.append({
            "title": f"Category Benchmark: {cat} vs {num}",
            "desc": f"Ranks leading categories by average {num}.",
            "figure": fig_bar,
            "type": "ranking",
        })

    # 3. Correlation Matrix Heatmap
    if len(numeric_cols) >= 2:
        top_num = numeric_cols[:8]
        corr = df[top_num].corr(numeric_only=True).round(2)
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Cross-Metric Correlation Matrix",
            template="plotly_white",
        )
        fig_corr.update_layout(margin=dict(l=30, r=30, t=50, b=30))
        insights.append({
            "title": "Correlation Analysis",
            "desc": "Identifies strong positive or negative linear associations across quantitative attributes.",
            "figure": fig_corr,
            "type": "correlation",
        })

    # 4. Proportions & Share (Donut Chart)
    if categorical_cols:
        cat_pie = categorical_cols[min(1, len(categorical_cols) - 1)]
        val_counts = df[cat_pie].value_counts().head(8).reset_index()
        val_counts.columns = [cat_pie, "Records"]
        fig_pie = px.pie(
            val_counts,
            names=cat_pie,
            values="Records",
            hole=0.45,
            title=f"Proportional Share: {cat_pie}",
            color_discrete_sequence=["#4f46e5", "#06b6d4", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6"],
            template="plotly_white",
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(margin=dict(l=30, r=30, t=50, b=30))
        insights.append({
            "title": f"Composition Breakdown: {cat_pie}",
            "desc": f"Visualizes the relative proportion of records across {cat_pie}.",
            "figure": fig_pie,
            "type": "composition",
        })

    # 5. Outlier & Dispersion Comparison (Box Plot)
    if categorical_cols and numeric_cols:
        cat = categorical_cols[0]
        num = numeric_cols[min(1, len(numeric_cols) - 1)]
        top_cats = df[cat].value_counts().head(6).index
        filtered_df = df[df[cat].isin(top_cats)]
        fig_box = px.box(
            filtered_df,
            x=cat,
            y=num,
            color=cat,
            title=f"Dispersion & Outliers: {num} by {cat}",
            color_discrete_sequence=["#4f46e5", "#06b6d4", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6"],
            template="plotly_white",
        )
        fig_box.update_layout(margin=dict(l=30, r=30, t=50, b=30), showlegend=False)
        insights.append({
            "title": f"Spread & Outlier Inspection: {num}",
            "desc": "Compares interquartile ranges, medians, and statistical outliers across cohorts.",
            "figure": fig_box,
            "type": "dispersion",
        })

    # 6. Relationship & Trend / Scatter
    if len(numeric_cols) >= 2:
        x_n = numeric_cols[0]
        y_n = numeric_cols[1]
        sample_size = min(1500, len(df))
        sampled = df.sample(sample_size, random_state=42) if len(df) > sample_size else df
        fig_scat = px.scatter(
            sampled,
            x=x_n,
            y=y_n,
            trendline="ols",
            title=f"Bivariate Trendline: {x_n} vs {y_n}",
            color_discrete_sequence=["#06b6d4"],
            template="plotly_white",
            opacity=0.7,
        )
        fig_scat.update_layout(margin=dict(l=30, r=30, t=50, b=30))
        insights.append({
            "title": f"Bivariate Interaction: {x_n} × {y_n}",
            "desc": f"Plots regression slope and scatter spread across {sample_size} records.",
            "figure": fig_scat,
            "type": "relationship",
        })

    # 7. Time-Series Timeline (if date column exists)
    if datetime_cols and numeric_cols:
        dt_col = datetime_cols[0]
        val_col = numeric_cols[0]
        df_sorted = df.dropna(subset=[dt_col, val_col]).sort_values(by=dt_col)
        fig_line = px.line(
            df_sorted,
            x=dt_col,
            y=val_col,
            title=f"Temporal Trajectory: {val_col} over {dt_col}",
            color_discrete_sequence=["#10b981"],
            template="plotly_white",
        )
        fig_line.update_layout(margin=dict(l=30, r=30, t=50, b=30))
        insights.append({
            "title": f"Time-Series Trend: {val_col}",
            "desc": f"Tracks sequential evolution along {dt_col}.",
            "figure": fig_line,
            "type": "trend",
        })

    return insights[:max_charts]

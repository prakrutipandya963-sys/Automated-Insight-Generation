import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .config import label, unit


COLORS = {"High": "#BD4A3C", "Medium": "#C58A28", "Low": "#258F85"}


def severity_chart(insights: pd.DataFrame):
    counts = insights["severity"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
    fig = px.bar(x=counts.index, y=counts.values, color=counts.index,
                 color_discrete_map=COLORS, text=counts.values,
                 labels={"x": "Review priority", "y": "Insights"})
    fig.update_layout(showlegend=False, height=340, margin=dict(l=10, r=10, t=20, b=20))
    fig.update_yaxes(dtick=1)
    return fig


def correlation_chart(matrix: pd.DataFrame, counts: pd.DataFrame):
    fig = go.Figure(go.Heatmap(
        z=matrix.to_numpy(), x=[label(c) for c in matrix.columns],
        y=[label(c) for c in matrix.index], customdata=counts.to_numpy(),
        zmin=-1, zmax=1, zmid=0, colorscale="RdBu", texttemplate="%{z:.2f}",
        hovertemplate="%{y} / %{x}<br>Pearson r: %{z:.3f}<br>Paired rows: %{customdata}<extra></extra>",
        hoverongaps=False, colorbar=dict(title="r"),
    ))
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=20, b=20))
    fig.update_yaxes(autorange="reversed")
    return fig


def district_chart(data: pd.DataFrame, indicator: str):
    # Reindex each series to calendar months so lines never bridge missing months.
    months = pd.date_range(data["month"].min(), data["month"].max(), freq="MS")
    parts = []
    for district, group in data.groupby("district"):
        series = group.set_index("month")[[indicator]].reindex(months)
        series.index.name = "month"
        series["district"] = district
        parts.append(series.reset_index())
    chart_data = pd.concat(parts, ignore_index=True)
    fig = px.line(chart_data, x="month", y=indicator, color="district", markers=True,
                  labels={"month": "Month", indicator: f"{label(indicator)} {unit(indicator)}".strip(),
                          "district": "District"},
                  color_discrete_sequence=px.colors.qualitative.Safe)
    fig.update_traces(connectgaps=False)
    fig.update_xaxes(dtick="M1", tickformat="%b %Y")
    fig.update_layout(height=390, margin=dict(l=10, r=10, t=20, b=20), legend=dict(orientation="h", y=-0.25))
    return fig

import numpy as np
import pandas as pd
from plotly import express as px
from plotly import graph_objects as go
from plotly import subplots as sp


def create_multiplot_heatmap(df_input: pd.DataFrame, columns: list[str]):
    df = df_input.copy()
    num_plots = len(columns)

    df.sort_values(by="order", inplace=True)
    fig = sp.make_subplots(
        rows=num_plots,
        cols=1,
        subplot_titles=[str(col) for col in columns],
        shared_xaxes=True,
        vertical_spacing=0.15 / num_plots,
    )

    # Dynamically compute colorbar lengths and positions
    colorbar_len = min(0.3, 0.6 / num_plots)
    colorbar_positions = np.linspace(
        1 - (1 / (2 * num_plots)), (1 / (2 * num_plots)), num=num_plots
    )

    for i, col in enumerate(columns):
        if df[col].dtype == "object" or df[col].dtype.name == "category":
            df[col] = df[col].fillna("NOT_AVAILABLE")
            categories = pd.Categorical(df[col])
            df[col] = categories.codes
            tickvals = list(range(len(categories.categories)))
            ticktext = categories.categories.tolist()
            if len(categories.categories) == 1:
                colorscale_val = [[0, "#636efa"], [1, "#636efa"]]
            else:
                colorscale_val = px.colors.qualitative.Set3[: len(categories.categories)]
        else:
            tickvals = None
            ticktext = None
            colorscale_val = "Viridis"

        heatmap = go.Heatmap(
            x=df["time"],
            y=df["order"],
            z=df[col],
            colorscale=colorscale_val,
            colorbar=dict(
                title=None,
                x=1.05,
                y=colorbar_positions[i],
                len=colorbar_len,
                yanchor="middle",
                tickvals=tickvals,
                ticktext=ticktext,
            ),
        )
        fig.add_trace(heatmap, row=i + 1, col=1)

        fig.update_yaxes(title_text="order", row=i + 1, col=1)

    fig.update_layout(
        title_text="Validation values", height=max(400, 300 * num_plots), showlegend=False
    )
    return fig


def create_plot_by_segment(df):
    # Ensure the data is sorted by order and time
    df = df.sort_values(["order", "time"])

    # Mapping for segment_closure_status colors (used for the markers)
    unique_statuses = df["segment_closure_status"].unique()
    color_palette = px.colors.qualitative.Plotly
    status_color_map = {
        status: color_palette[i % len(color_palette)] for i, status in enumerate(unique_statuses)
    }

    # Mapping for order colors (used for the lines)
    unique_orders = df["order"].unique()
    order_color_palette = px.colors.qualitative.Dark24
    order_color_map = {
        order: order_color_palette[i % len(order_color_palette)]
        for i, order in enumerate(unique_orders)
    }

    # Create the figure
    fig = go.Figure()

    # Loop over the unique orders
    for order in unique_orders:
        # Filter the data by order and sort by time
        df_order = df[df["order"] == order].sort_values("time")

        # Add a trace with mode 'lines+markers'
        fig.add_trace(
            go.Scatter(
                x=df_order["time"],
                y=df_order["running_mean"],
                mode="lines+markers",
                name=f"Order {order}",
                line=dict(color=order_color_map[order]),
                marker=dict(
                    color=[
                        status_color_map[status] for status in df_order["segment_closure_status"]
                    ],
                    size=8,
                ),
                text=df_order["segment_closure_status"],
                hovertemplate="Time: %{x}<br>Value: %{y}<br>Status: %{text}<extra></extra>",
            )
        )

    # Add horizontal reference lines at y=0.15 and y=0.75
    fig.add_hline(
        y=0.15,
        line_dash="dash",
        line_color="red",
        annotation_text="0.15",
        annotation_position="bottom left",
    )
    fig.add_hline(
        y=0.75,
        line_dash="dash",
        line_color="green",
        annotation_text="0.75",
        annotation_position="top left",
    )

    # Update layout
    fig.update_layout(
        title="Time vs Current Running Mean", xaxis_title="Time", yaxis_title="Current Running Mean"
    )

    return fig

import plotly.graph_objs as go
import pandas as pd

# Example data
df = pd.DataFrame({
    "trial": range(1, 11),
    "logical_error_rate": [0.1, 0.08, 0.13, 0.09, 0.11, 0.07, 0.14, 0.10, 0.12, 0.09]
})

# Create interactive chart
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["trial"],
    y=df["logical_error_rate"],
    mode='lines+markers',
    name='Logical Error Rate'
))

fig.update_layout(
    title="Fusion QEC Logical Error Rate by Trial",
    xaxis_title="Trial",
    yaxis_title="Logical Error Rate"
)

fig.write_html("data/Fusion_Chart.html")
fig.show()
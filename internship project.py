import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import plotly.graph_objects as go

# Load data
@st.cache
def load_data():
    data = pd.read_csv("Spotify-2000.csv")
    return data

data = load_data()

# Preprocessing
data = data.drop("Index", axis=1)
data2 = data[["Beats Per Minute (BPM)", "Loudness (dB)", "Liveness", "Valence", "Acousticness", "Speechiness"]]

# Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data2)
data_scaled = pd.DataFrame(data_scaled, columns=data2.columns)

# Clustering
kmeans = KMeans(n_clusters=10)
clusters = kmeans.fit_predict(data_scaled)
data["Music Segments"] = clusters + 1

# Streamlit UI
st.title('Music Segments Analysis')
st.write(data.head())

# Plotting
PLOT = go.Figure()
for i in range(1, 11):
    cluster_data = data[data["Music Segments"] == i]
    PLOT.add_trace(go.Scatter3d(x=cluster_data['Beats Per Minute (BPM)'],
                                y=cluster_data['Energy'],
                                z=cluster_data['Danceability'],
                                mode='markers', marker_size=6, marker_line_width=1,
                                name=f'Cluster {i}'))

PLOT.update_traces(hovertemplate='Beats Per Minute (BPM): %{x} <br>Energy: %{y} <br>Danceability: %{z}')

PLOT.update_layout(width=800, height=800, autosize=True, showlegend=True,
                   scene=dict(xaxis=dict(title='Beats Per Minute (BPM)', titlefont_color='black'),
                              yaxis=dict(title='Energy', titlefont_color='black'),
                              zaxis=dict(title='Danceability', titlefont_color='black')),
                   font=dict(family="Gilroy", color='black', size=12))

st.plotly_chart(PLOT)

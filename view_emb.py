import numpy as np
import streamlit as st
from umap import UMAP
import plotly.express as px
import pandas as pd
from pathlib import Path


DATA_DIR = Path('/hdd/bdhammel/photo_dataset/')
db_path = DATA_DIR/'db.pkl'

df = pd.read_pickle(db_path)
umap = UMAP(n_components=3, init='random', random_state=0)

proj = umap.fit_transform(np.asarray(df.vector.to_list()))

fig = px.scatter_3d(proj, x=0, y=1, z=2)
fig.update_traces(marker_size=5)

st.plotly_chart(fig, use_container_width=True)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="🏠 AI House Price",
    page_icon="🏠",
    layout="wide"
)

# ---------------------------
# STYLE (small visual upgrade)
# ---------------------------
st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    h1, h2, h3 {color: #ffffff;}
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# TITLE
# ---------------------------
st.title("🏠 AI House Price Predictor")
st.markdown("### Modern ML-powered real estate analysis dashboard")

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def ld():
    d = fetch_california_housing()
    x = pd.DataFrame(d.data, columns=d.feature_names)
    y = pd.Series(d.target, name="price")
    return x, y

x, y = ld()

# ---------------------------
# MODEL
# ---------------------------
@st.cache_resource
def mdl():
    m = RandomForestRegressor(n_estimators=120, random_state=42)
    m.fit(x, y)
    return m

m = mdl()

# ---------------------------
# SIDEBAR INPUT
# ---------------------------
st.sidebar.header("🏡 Customize your house")

def inp():
    d = {}
    for c in x.columns:
        d[c] = st.sidebar.slider(
            c,
            float(x[c].min()),
            float(x[c].max()),
            float(x[c].mean())
        )
    return pd.DataFrame(d, index=[0])

u = inp()

# ---------------------------
# PREDICTION
# ---------------------------
p = m.predict(u)[0]
price = p * 100000
avg = y.mean() * 100000

# loading effect
with st.spinner("Calculating AI prediction..."):
    prog = st.progress(100)

# ---------------------------
# METRICS
# ---------------------------
c1, c2, c3 = st.columns(3)

c1.metric("💰 Your House Price", f"${price:,.0f}")
c2.metric("📊 Dataset Average", f"${avg:,.0f}")
c3.metric("📉 Difference", f"${price-avg:,.0f}")

st.divider()

# ---------------------------
# TABS
# ---------------------------
t1, t2, t3, t4 = st.tabs(["📥 Input", "📊 Analytics", "🗺️ Map", "🧠 Model"])

# ---------------------------
# TAB 1
# ---------------------------
with t1:
    st.subheader("Your Input Data")

    st.dataframe(u, use_container_width=True)

    fig = px.bar(
        x=u.columns,
        y=u.iloc[0],
        title="House Feature Overview"
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# TAB 2
# ---------------------------
with t2:
    st.subheader("Analytics Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            y,
            nbins=40,
            title="Price Distribution"
        )
        fig.add_vline(x=p, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        imp = pd.DataFrame({
            "Feature": x.columns,
            "Importance": m.feature_importances_
        }).sort_values("Importance")

        fig = px.bar(
            imp,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Feature Importance"
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# TAB 3
# ---------------------------
with t3:
    st.subheader("Geographic View")

    mp = x.copy()
    mp["price"] = y

    st.map(mp.rename(columns={
        "Latitude": "lat",
        "Longitude": "lon"
    }))

# ---------------------------
# TAB 4
# ---------------------------
with t4:
    st.subheader("Model Insight")

    st.info("Random Forest Regressor is used for prediction")

    st.write("""
    ✔ Multiple decision trees  
    ✔ Handles complex relationships  
    ✔ Strong for real estate data  
    """)

    corr = x.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns
    ))

    fig.update_layout(title="Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# COMPARISON
# ---------------------------
st.divider()

st.subheader("📊 Compare with Dataset")

cmp = pd.DataFrame({
    "Your House": u.iloc[0],
    "Average": x.mean()
})

st.bar_chart(cmp)

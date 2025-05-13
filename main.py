import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="HIV Spread Simulation", layout="wide")

# This is the sidebar section
with st.sidebar:
    st.title("🧬 HIV Simulation Settings")
    population_size = st.slider("Population Size", 100, 10000, 1000, step=100)
    condom_use = st.slider("Condom Usage Rate (%)", 0, 100, 60)
    partner_change_rate = st.slider("Partner Change Rate", 0.1, 5.0, 1.0, step=0.1)
    treatment_rate = st.slider("Treatment Coverage (%)", 0, 100, 50)
    testing_frequency = st.slider("Testing Frequency (days)", 7, 365, 90)
    simulation_days = st.slider("Days to Simulate", 30, 1000, 365, step=30)

    st.markdown("---")
    st.info("Adjust these sliders to simulate behavior-based HIV spread dynamics.")

# This is the Main Title section
st.title("📊 HIV Spread and Intervention Simulator")
st.markdown("Simulates how HIV spreads in a population and how interventions affect outcomes.")

# This section has some dummy metrics (this is where you replace with real Mesa model output during integration)
np.random.seed(42)
days = np.arange(simulation_days)
prevalence = np.clip(np.cumsum(np.random.randn(simulation_days) * 0.05 + 0.1), 0, population_size)
incidence = np.clip(np.random.randn(simulation_days) * 2 + 10, 0, None)
deaths = np.cumsum(np.random.rand(simulation_days) * 0.2)

# This is the section for top Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Infected", f"{int(prevalence[-1])} people")
col2.metric("New Infections (Day 1)", f"{int(incidence[0])}")
col3.metric("Total Deaths", f"{int(deaths[-1])}")

st.markdown("---")

# this is the section for Charts
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("📈 HIV Prevalence Over Time")
    fig1, ax1 = plt.subplots()
    ax1.plot(days, prevalence, color="crimson", linewidth=2)
    ax1.set_xlabel("Days")
    ax1.set_ylabel("People Infected")
    ax1.grid(True)
    st.pyplot(fig1)

with chart_col2:
    st.subheader("🧩 HIV Incidence Over Time")
    fig2, ax2 = plt.subplots()
    ax2.bar(days, incidence, color="darkgreen")
    ax2.set_xlabel("Days")
    ax2.set_ylabel("New Infections")
    st.pyplot(fig2)

with st.expander("📊 Advanced Metrics"):
    st.line_chart(deaths, use_container_width=True)
    st.caption("Cumulative HIV-related deaths over the simulation period.")

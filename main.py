import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --- SIR Model Classes and Functions ---
class SIRModel:
    def __init__(self, N, beta, gamma, initial_infected):
        self.N = N
        self.beta = beta
        self.gamma = gamma
        self.S = N - initial_infected
        self.I = initial_infected
        self.R = 0
        self.data = []
        self._record_step(0)

    def _record_step(self, step):
        self.data.append({
            "Step": step,
            "Susceptible": self.S,
            "Infected": self.I,
            "Recovered": self.R
        })

    def step(self):
        new_infected = self.beta * self.S * self.I / self.N
        new_recovered = self.gamma * self.I

        self.S = max(self.S - new_infected, 0)
        self.I = max(self.I + new_infected - new_recovered, 0)
        self.R = self.N - self.S - self.I
        self._record_step(len(self.data))

    def get_data(self):
        return pd.DataFrame(self.data)

def run_sir_model(S0, I0, R0, beta, gamma, days):
    model = SIRModel(S0 + I0 + R0, beta, gamma, I0)
    for _ in range(days):
        model.step()
    df = model.get_data()
    df.rename(columns={"Step": "Day"}, inplace=True)
    return df

def plot_sir_log_chart(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['Day'], df['Susceptible'], label='Susceptible', color='blue')
    ax.plot(df['Day'], df['Infected'], label='Infected', color='red')
    ax.plot(df['Day'], df['Recovered'], label='Recovered', color='green')

    ax.set_yscale('log')
    ax.set_xlabel('Day')
    ax.set_ylabel('Population Count (log scale)')
    ax.set_title('SIR Model Dynamics (Log Scale)')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("🧪 HIV Transmission Simulation Platform")

# Model selection
model_choice = st.radio("Select Model Type", ["Agent-Based (ABM)", "SIR (Compartmental)"])

# Sidebar parameters
st.sidebar.header("📊 Simulation Parameters")
population_size = st.sidebar.slider("Population Size", 100, 2000, 500, 100)
initial_infected = st.sidebar.slider("Initially Infected", 1, 100, 10)
simulation_days = st.sidebar.slider("Simulation Days", 100, 1000, 365, 50)
partner_change_rate = st.sidebar.slider("Partner Change Rate", 0.01, 1.0, 0.05)
condom_use_prob = st.sidebar.slider("Condom Use Probability", 0.0, 1.0, 0.5)
treatment_rate = st.sidebar.slider("Treatment Rate (ART)", 0.0, 1.0, 0.3)
intervention_start = st.sidebar.slider("Intervention Start Day", 0, simulation_days, 100)
intervention_effectiveness = st.sidebar.slider("Intervention Effectiveness", 0.0, 1.0, 0.5)

# Run simulation button
run_button = st.button("🚀 Run Simulation")

if run_button:
    if model_choice == "Agent-Based (ABM)":
        from hiv_model import HIVModel, run_model, plot_hiv_results

        with st.spinner("Running Agent-Based Model..."):
            results, model = run_model(
                population_size=population_size,
                initial_infected=initial_infected,
                days=simulation_days,
                partner_change_rate=partner_change_rate,
                condom_use_prob=condom_use_prob,
                treatment_rate=treatment_rate,
                intervention_day=intervention_start,
                intervention_effectiveness=intervention_effectiveness
            )

        st.subheader("📈 ABM Simulation Results")
        fig1 = plot_hiv_results(results)
        st.pyplot(fig1)

        st.write("📌 Final Statistics:")
        st.write(results.tail(1).T)

    else:
        with st.spinner("Running SIR model..."):
            S0 = population_size - initial_infected
            I0 = initial_infected
            R0 = 0
            beta = partner_change_rate * 0.05
            gamma = treatment_rate

            results = run_sir_model(S0, I0, R0, beta, gamma, simulation_days)

        st.subheader("📈 SIR Model Dynamics")
        st.line_chart(results.set_index("Day"))

        st.subheader("📉 SIR Dynamics (Log Scale)")
        fig2 = plot_sir_log_chart(results)
        st.pyplot(fig2)

        st.write("📌 Final Statistics:")
        st.write(results.tail(1).T)

# Footer
st.markdown("---")
st.markdown("📘 Developed for HIV Intervention Modeling using ABM and SIR | Powered by Streamlit")

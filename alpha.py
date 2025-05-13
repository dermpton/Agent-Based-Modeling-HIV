import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import random
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

st.set_page_config(page_title="HIV Spread Simulation", layout="wide")

# Define the states of the agents
class State:
    SUSCEPTIBLE = 0
    INFECTED = 1
    RECOVERED = 2  # This represents treated/managed cases

# Define the agent class
class HIVAgent(Agent):
    """An agent in the HIV infection model."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.state = State.SUSCEPTIBLE
        self.infection_time = 0
        self.days_since_last_test = 0
        self.knows_status = False
        
    def step(self):
        """Agent's step in the model."""
        # Increment days since last test
        self.days_since_last_test += 1
        
        # Check if it's time for testing
        if self.days_since_last_test >= self.model.testing_frequency:
            self.days_since_last_test = 0
            if self.state == State.INFECTED:
                # Probability to discover status based on testing
                if random.random() < 0.95:  # 95% test accuracy
                    self.knows_status = True
        
        # Treatment opportunity
        if self.state == State.INFECTED and self.knows_status:
            # Check if agent gets treatment based on treatment coverage
            if random.random() < self.model.treatment_rate:
                self.state = State.RECOVERED
        
        # Interaction logic for infected individuals
        if self.state == State.INFECTED:
            # Partner change rate determines how many interactions per step
            interactions = max(1, int(round(self.model.partner_change_rate)))
            for _ in range(interactions):
                self.interact_randomly()

    def interact_randomly(self):
        """Interact with a random agent and potentially infect them."""
        # Get a list of all susceptible agents excluding self
        susceptible_agents = [agent for agent in self.model.schedule.agents 
                             if agent != self and agent.state == State.SUSCEPTIBLE]

        # If there are susceptible agents, select one randomly to interact with
        if susceptible_agents:
            chosen_agent = self.random.choice(susceptible_agents)
            
            # Apply condom usage logic
            if random.random() < self.model.condom_use:
                # Condom used - reduced transmission risk (95% effective)
                effective_ptrans = self.model.ptrans * 0.05
            else:
                # No condom used - full transmission risk
                effective_ptrans = self.model.ptrans
                
            # Attempt infection with the effective transmission probability
            if self.random.random() < effective_ptrans:
                chosen_agent.state = State.INFECTED
                chosen_agent.infection_time = self.model.schedule.time
                chosen_agent.knows_status = False

# Define the model class
class HIVModel(Model):
    """A model for HIV infection spread with interventions."""

    def __init__(self, N, ptrans, condom_use, partner_change_rate, treatment_rate, 
                 testing_frequency, initial_infected_count=5):
        self.num_agents = N
        self.ptrans = ptrans  # Base transmission probability
        self.condom_use = condom_use / 100.0  # Convert from percentage to probability
        self.partner_change_rate = partner_change_rate
        self.treatment_rate = treatment_rate / 100.0  # Convert from percentage to probability
        self.testing_frequency = testing_frequency
        self.schedule = RandomActivation(self)
        
        # Data collection for analysis
        self.datacollector = DataCollector(
            model_reporters={
                "Susceptible": lambda m: self.count_state(m, State.SUSCEPTIBLE),
                "Infected": lambda m: self.count_state(m, State.INFECTED),
                "Treated": lambda m: self.count_state(m, State.RECOVERED),
                "New_Infections": lambda m: self.count_new_infections(m)
            }
        )
        
        # Track new infections per step
        self.new_infections_current_step = 0
        self.previous_infected_count = 0
        
        # Create agents
        for i in range(self.num_agents):
            agent = HIVAgent(i, self)
            self.schedule.add(agent)

        # Infect a specified number of random agents at the start
        num_to_infect = min(initial_infected_count, N)
        initial_infected_agents = random.sample(list(self.schedule.agents), num_to_infect)
        for agent in initial_infected_agents:
             agent.state = State.INFECTED
             agent.infection_time = 0  # Infection starts at time 0
        
        self.previous_infected_count = num_to_infect

    def count_state(self, model, state):
        """Helper method to count agents in a given state."""
        return sum(1 for agent in model.schedule.agents if agent.state == state)
        
    def count_new_infections(self, model):
        """Count new infections in the current step."""
        current_infected = self.count_state(model, State.INFECTED) + self.count_state(model, State.RECOVERED)
        new_infections = max(0, current_infected - self.previous_infected_count)
        self.previous_infected_count = current_infected
        return new_infections

    def step(self):
        """Advance the model by one step."""
        self.datacollector.collect(self)
        self.schedule.step()

# Sidebar settings
with st.sidebar:
    st.title("🧬 HIV Simulation Settings")
    population_size = st.slider("Population Size", 100, 10000, 1000, step=100)
    condom_use = st.slider("Condom Usage Rate (%)", 0, 100, 60)
    partner_change_rate = st.slider("Partner Change Rate", 0.1, 5.0, 1.0, step=0.1)
    treatment_rate = st.slider("Treatment Coverage (%)", 0, 100, 50)
    testing_frequency = st.slider("Testing Frequency (days)", 7, 365, 90)
    simulation_days = st.slider("Days to Simulate", 30, 1000, 365, step=30)
    
    # Base transmission probability (hidden from UI for simplicity but can be exposed)
    ptrans = 0.2  
    
    st.markdown("---")
    st.info("Adjust these sliders to simulate behavior-based HIV spread dynamics.")
    
    if st.button("Run Simulation", type="primary"):
        with st.spinner("Running simulation..."):
            # Create and run the model with the selected parameters
            model = HIVModel(
                N=population_size,
                ptrans=ptrans,
                condom_use=condom_use,
                partner_change_rate=partner_change_rate,
                treatment_rate=treatment_rate,
                testing_frequency=testing_frequency,
                initial_infected_count=5
            )
            
            # Run the model for the specified number of days
            for _ in range(simulation_days):
                model.step()
            
            # Get the model data
            model_data = model.datacollector.get_model_vars_dataframe()
            
            # Store results in session state to access after the button is clicked
            st.session_state.model_data = model_data
            st.session_state.simulation_run = True
    
# Main page content
st.title("📊 HIV Spread and Intervention Simulator")
st.markdown("Simulates how HIV spreads in a population and how interventions affect outcomes.")

# Check if simulation has been run
if 'simulation_run' not in st.session_state:
    st.session_state.simulation_run = False
    st.info("👈 Adjust the parameters in the sidebar and click 'Run Simulation' to start")

if st.session_state.simulation_run:
    model_data = st.session_state.model_data

    # Top metrics
    col1, col2, col3 = st.columns(3)
    
    # Calculate metrics
    final_infected = model_data['Infected'].iloc[-1]
    peak_new_infections = model_data['New_Infections'].max()
    total_treated = model_data['Treated'].iloc[-1]
    
    col1.metric("Total Infected", f"{int(final_infected)} people")
    col2.metric("Peak New Infections", f"{int(peak_new_infections)}")
    col3.metric("Total Treated", f"{int(total_treated)}")

    st.markdown("---")

    # Prevalence and Incidence Charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("📈 HIV Prevalence Over Time")
        fig1, ax1 = plt.subplots()
        ax1.plot(model_data.index, model_data['Infected'], color="crimson", linewidth=2, label="Active Cases")
        ax1.plot(model_data.index, model_data['Treated'], color="orange", linewidth=2, label="Treated Cases")
        ax1.set_xlabel("Days")
        ax1.set_ylabel("People")
        ax1.legend()
        ax1.grid(True)
        st.pyplot(fig1)

    with chart_col2:
        st.subheader("🧩 HIV Incidence Over Time")
        fig2, ax2 = plt.subplots()
        ax2.bar(model_data.index, model_data['New_Infections'], color="darkgreen")
        ax2.set_xlabel("Days")
        ax2.set_ylabel("New Infections")
        st.pyplot(fig2)

    with st.expander("📊 Advanced Metrics"):
        st.subheader("Population Composition")
        fig3, ax3 = plt.subplots()
        ax3.stackplot(model_data.index, 
                      model_data['Susceptible'], 
                      model_data['Infected'],
                      model_data['Treated'],
                      labels=['Susceptible', 'Active Infection', 'Treated'],
                      colors=['green', 'red', 'blue'])
        ax3.set_xlabel("Days")
        ax3.set_ylabel("Number of People")
        ax3.legend(loc='upper right')
        st.pyplot(fig3)
        
        # Calculate and display more statistics
        infection_rate = (final_infected + total_treated) / population_size * 100
        treatment_success = (total_treated / (final_infected + total_treated) * 100) if (final_infected + total_treated) > 0 else 0
        
        st.markdown(f"""
        ### Summary Statistics
        - **Infection Rate**: {infection_rate:.2f}% of population
        - **Treatment Success Rate**: {treatment_success:.2f}%
        - **Simulation Duration**: {simulation_days} days
        """)
else:
    # Placeholder for when simulation hasn't been run yet
    st.info("Run the simulation using the sidebar controls to see results here.")
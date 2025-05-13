import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


# Define the states of the agents
class State:
    SUSCEPTIBLE = 0
    INFECTED = 1
    RECOVERED = 2

# Define the agent class
class SIRAgent:
    """An agent in the SIR infection model."""
    
    def __init__(self, unique_id):
        self.unique_id = unique_id
        self.state = State.SUSCEPTIBLE
        self.infection_time = 0

    def interact(self, other_agents, ptrans, current_time):
        """Interact with a random agent and potentially infect them."""
        # If infected, interact with susceptible agents
        if self.state == State.INFECTED and other_agents:
            for agent in other_agents:
                if agent.state == State.SUSCEPTIBLE:
                    if random.random() < ptrans:
                        agent.state = State.INFECTED
                        agent.infection_time = current_time

    def check_recovery(self, current_time, recovery_time):
        """Check if the agent recovers."""
        if self.state == State.INFECTED:
            if current_time - self.infection_time >= recovery_time:
                self.state = State.RECOVERED

# Define the model class
class SIRModel:
    """A simplified SIR model with random interactions."""

    def __init__(self, N, ptrans, recovery_time, initial_infected_count=1):
        self.num_agents = N
        self.ptrans = ptrans
        self.recovery_time = recovery_time
        self.current_time = 0
        
        # Create agents
        self.agents = [SIRAgent(i) for i in range(self.num_agents)]

        # Infect a specified number of random agents at the start
        num_to_infect = min(initial_infected_count, N)
        initial_infected_agents = random.sample(self.agents, num_to_infect)
        for agent in initial_infected_agents:
            agent.state = State.INFECTED
            agent.infection_time = 0  # Infection starts at time 0
        
        # Initialize data collection
        self.data = []
        self._collect_data()

    def step(self):
        """Advance the model by one step."""
        self.current_time += 1
        
        # Randomly order agents for fairness
        random.shuffle(self.agents)
        
        # Each infected agent interacts with susceptible agents
        for agent in self.agents:
            if agent.state == State.INFECTED:
                other_agents = [a for a in self.agents if a != agent]
                agent.interact(other_agents, self.ptrans, self.current_time)
        
        # Check for recoveries
        for agent in self.agents:
            agent.check_recovery(self.current_time, self.recovery_time)
        
        # Collect data for this step
        self._collect_data()
    
    def _collect_data(self):
        """Collect data on the current state of the model."""
        susceptible = sum(1 for agent in self.agents if agent.state == State.SUSCEPTIBLE)
        infected = sum(1 for agent in self.agents if agent.state == State.INFECTED)
        recovered = sum(1 for agent in self.agents if agent.state == State.RECOVERED)
        
        self.data.append({
            'Step': self.current_time,
            'Susceptible': susceptible,
            'Infected': infected,
            'Recovered': recovered
        })
    
    def get_data(self):
        """Return the collected data as a pandas DataFrame."""
        return pd.DataFrame(self.data)

# Function to run the SIR model
def run_sir_model(S0, I0, R0, beta, gamma, days):
    """Run the SIR model simulation."""
    model = SIRModel(S0 + I0 + R0, beta, gamma, I0)
    for _ in range(days):
        model.step()
    data = model.get_data()

    # Rename 'Step' to 'Day' for compatibility with your usage in Streamlit
    data.rename(columns={'Step': 'Day'}, inplace=True)
    return data

# Function to plot SIR results (adjusted for log scale)
def plot_sir_log_chart(df):
    """
    Plot the SIR dynamics on a log scale.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df['Day'], df['Susceptible'], label='Susceptible', color='blue')
    ax.plot(df['Day'], df['Infected'], label='Infected', color='red')
    ax.plot(df['Day'], df['Recovered'], label='Recovered', color='green')

    ax.set_yscale('log')  # Log scale for y-axis
    ax.set_xlabel('Day')
    ax.set_ylabel('Population Count (log scale)')
    ax.set_title('SIR Model Dynamics (Log Scale)')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    plt.tight_layout()
    return fig


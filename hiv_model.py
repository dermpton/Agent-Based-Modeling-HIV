import enum
import numpy as np
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import matplotlib.pyplot as plt
import pandas as pd

# Enum for HIV status
class HIVStatus(enum.IntEnum):
    SUSCEPTIBLE = 0
    INFECTED_UNTREATED = 1
    INFECTED_TREATED = 2
    AIDS_UNTREATED = 3
    AIDS_TREATED = 4
    DEAD = 5

# Person agent definition
class PersonAgent(Agent):
    def __init__(self, unique_id, model):
        # Explicitly initializing the Agent base class
        super().__init__(unique_id, model)  
        self.hiv_status = HIVStatus.SUSCEPTIBLE
        self.infection_time = None
        self.progression_time = None
        self.knows_status = False
        self.on_treatment = False


    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def update_disease_status(self):
        if self.hiv_status in [HIVStatus.SUSCEPTIBLE, HIVStatus.DEAD]:
            return
        
        if not self.knows_status and self.random.random() < self.model.testing_rate:
            self.knows_status = True
            if self.random.random() < self.model.treatment_coverage:
                self.on_treatment = True
                if self.hiv_status == HIVStatus.INFECTED_UNTREATED:
                    self.hiv_status = HIVStatus.INFECTED_TREATED
                elif self.hiv_status == HIVStatus.AIDS_UNTREATED:
                    self.hiv_status = HIVStatus.AIDS_TREATED
        
        if self.hiv_status in [HIVStatus.INFECTED_UNTREATED, HIVStatus.INFECTED_TREATED]:
            time_since_infection = self.model.schedule.time - self.infection_time
            effective_time = time_since_infection / 2 if self.on_treatment else time_since_infection
            if effective_time >= self.progression_time:
                self.hiv_status = HIVStatus.AIDS_TREATED if self.on_treatment else HIVStatus.AIDS_UNTREATED
        
        death_prob = {
            HIVStatus.AIDS_UNTREATED: self.model.death_rate_untreated,
            HIVStatus.AIDS_TREATED: self.model.death_rate_treated,
            HIVStatus.INFECTED_UNTREATED: self.model.death_rate_untreated / 10
        }.get(self.hiv_status, 0)
        
        if self.random.random() < death_prob:
            self.hiv_status = HIVStatus.DEAD
            self.model.deaths_this_step += 1
            self.model.schedule.remove(self)

    def interact(self):
        if self.hiv_status not in [HIVStatus.INFECTED_UNTREATED, HIVStatus.INFECTED_TREATED,
                                 HIVStatus.AIDS_UNTREATED, HIVStatus.AIDS_TREATED]:
            return
        
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        if len(cellmates) <= 1:
            return
        
        base_prob = self.model.base_transmission_rate
        if self.hiv_status == HIVStatus.AIDS_UNTREATED:
            base_prob *= 2.5
        elif self.hiv_status == HIVStatus.AIDS_TREATED:
            base_prob *= 0.5
        
        if self.on_treatment:
            base_prob *= (1 - self.model.treatment_effectiveness)
        
        for other in cellmates:
            if other.hiv_status != HIVStatus.SUSCEPTIBLE or self.random.random() > 0.05:
                continue
            effective_prob = base_prob * (0.05 if self.random.random() < self.model.condom_use_rate else 1)
            if self.random.random() < effective_prob:
                other.hiv_status = HIVStatus.INFECTED_UNTREATED
                other.infection_time = self.model.schedule.time
                other.progression_time = self.model.get_progression_time()
                other.knows_status = False
                self.model.new_infections += 1

    def step(self):
        if self.hiv_status != HIVStatus.DEAD:
            self.update_disease_status()
            self.move()
            self.interact()

# Model class
class HIVModel(Model):
    def __init__(self, N=1000, width=20, height=20,
                 initial_infected=10,
                 base_transmission_rate=0.04,
                 condom_use_rate=0.5,
                 treatment_coverage=0.2,
                 treatment_effectiveness=0.96,
                 disease_progression_time=3650,
                 disease_progression_sd=1095,
                 death_rate_untreated=0.1,
                 death_rate_treated=0.02,
                 testing_rate=0.3):
        
        super().__init__()
        self.num_agents = N
        self.initial_infected = initial_infected
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        
        self.base_transmission_rate = base_transmission_rate
        self.condom_use_rate = condom_use_rate
        self.treatment_coverage = treatment_coverage
        self.treatment_effectiveness = treatment_effectiveness
        self.disease_progression_time = disease_progression_time
        self.disease_progression_sd = disease_progression_sd
        self.death_rate_untreated = death_rate_untreated / 365
        self.death_rate_treated = death_rate_treated / 365
        self.testing_rate = testing_rate / 365

        # Create agents
        for i in range(self.num_agents):
            a = PersonAgent(i, self)
            self.schedule.add(a)
            self.grid.place_agent(a, (self.random.randrange(width), self.random.randrange(height)))
        
        # Infect selected number of initial individuals
        infected_agents = self.random.sample(self.schedule.agents, self.initial_infected)
        for agent in infected_agents:
            agent.hiv_status = HIVStatus.INFECTED_UNTREATED
            agent.infection_time = 0
            agent.progression_time = self.get_progression_time()

        self.datacollector = DataCollector(
            model_reporters={
                "Step": lambda m: m.schedule.time,
                "Susceptible": lambda m: self.count_status(HIVStatus.SUSCEPTIBLE),
                "HIV+ Untreated": lambda m: self.count_status(HIVStatus.INFECTED_UNTREATED),
                "HIV+ Treated": lambda m: self.count_status(HIVStatus.INFECTED_TREATED),
                "AIDS Untreated": lambda m: self.count_status(HIVStatus.AIDS_UNTREATED),
                "AIDS Treated": lambda m: self.count_status(HIVStatus.AIDS_TREATED),
                "Dead": lambda m: self.count_status(HIVStatus.DEAD),
                "New Infections": lambda m: m.new_infections,
                "Deaths": lambda m: m.deaths_this_step
            })
        
        self.new_infections = 0
        self.deaths_this_step = 0

    def count_status(self, status):
        return sum(1 for a in self.schedule.agents if a.hiv_status == status)

    def get_progression_time(self):
        return int(np.random.normal(self.disease_progression_time, self.disease_progression_sd))

    def step(self):
        self.new_infections = 0
        self.deaths_this_step = 0
        self.schedule.step()
        self.datacollector.collect(self)


# Function to run the Agent-Based Model with parameters
def run_model_with_params(years=10, population=1000, treatment_coverage=0.2, condom_use=0.5):
    model = HIVModel(
        N=population,
        treatment_coverage=treatment_coverage,
        condom_use_rate=condom_use
    )
    for _ in range(int(years * 365)):
        model.step()
    return model.datacollector.get_model_vars_dataframe()

# Compare intervention strategies
def compare_interventions():
    scenarios = {
        "Baseline": {"treatment_coverage": 0.2, "condom_use": 0.5},
        "High Treatment": {"treatment_coverage": 0.8, "condom_use": 0.5},
        "High Prevention": {"treatment_coverage": 0.2, "condom_use": 0.8},
        "Combined": {"treatment_coverage": 0.8, "condom_use": 0.8}
    }

    results = {}
    for name, params in scenarios.items():
        df = run_model_with_params(
            years=2,
            population=1000,
            treatment_coverage=params["treatment_coverage"],
            condom_use=params["condom_use"]
        )
        df["Scenario"] = name
        results[name] = df

    return results

# Function to plot the comparison for a metric
def plot_comparison(results_dict, metric="New Infections"):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, df in results_dict.items():
        if "Step" not in df.columns:
            continue
        df["Year"] = df["Step"] / 365
        ax.plot(df["Year"], df[metric], label=name)
    
    ax.set_xlabel("Year")
    ax.set_ylabel(metric)
    ax.set_title(f"Comparison of {metric} Across Scenarios")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig  # ✅ return the figure instead of plt.show()

def run_model(population_size, initial_infected, days,
              partner_change_rate, condom_use_prob,
              treatment_rate, intervention_day, intervention_effectiveness):

    model = HIVModel(
        N=population_size,
        initial_infected=initial_infected,
        base_transmission_rate=partner_change_rate,
        condom_use_rate=condom_use_prob,
        treatment_coverage=treatment_rate,
        treatment_effectiveness=intervention_effectiveness
    )

    for day in range(days):
        if day == intervention_day:
            model.treatment_coverage = min(1.0, model.treatment_coverage + 0.3)
            model.condom_use_rate = min(1.0, model.condom_use_rate + 0.2)
        model.step()

    results_df = model.datacollector.get_model_vars_dataframe()
    return results_df, model

def plot_hiv_results(results_df):
    fig, ax = plt.subplots(figsize=(12, 6))
    results_df.set_index("Step")[[
        "Susceptible", "HIV+ Untreated", "HIV+ Treated",
        "AIDS Untreated", "AIDS Treated", "Dead"
    ]].plot(ax=ax)
    ax.set_title("HIV Model Dynamics Over Time")
    ax.set_ylabel("Number of Individuals")
    ax.set_xlabel("Simulation Step (Days)")
    ax.grid(True)
    plt.tight_layout()
    return fig

# Example of running the model with intervention comparison and plotting results
results_dict = compare_interventions()
plot_comparison(results_dict)

# app.py
import time
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import IntEnum
from mesa.agent import Agent
from mesa.model import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

# PARAMETER DEFINITIONS:
# N: Total number of agents (people) in the simulation
# init_inf: Initial number of infected agents at the start
# base_trans: Base probability of HIV transmission per contact
# condom_use: Reduces base transmission rate (as a proportion)
# treat_cov: Proportion of tested individuals who receive treatment
# treat_eff: Effectiveness of treatment in reducing HIV progression
# prog_time: Average time (in days) from infection to AIDS without treatment
# prog_sd: Standard deviation in progression time across agents
# dr_unc: Annual death rate for untreated AIDS cases
# dr_tr: Annual death rate for treated AIDS cases
# test_rate: Probability per year that an individual gets tested
# days: Number of steps/days the simulation runs

# --- PAGE CONFIG & STYLES -------------------------------------------------
st.set_page_config(page_title="HIV MODEL - AGENT BASED SIMULATION", layout="wide")
st.markdown("""
<style>
.main .block-container { padding: 1rem 2rem; }
h1 { text-align: center; }
</style>
""", unsafe_allow_html=True)

# --- HEADER & DESCRIPTION --------------------------------------------------
st.title("HIV MODEL - AGENT BASED SIMULATION")
st.markdown("""
**Objective:** Simulate how HIV spreads and evolves in a population over time, and evaluate the impact of interventions (e.g., increased treatment coverage, condom use) on HIV prevalence, incidence, and mortality.

**Aims:**
1. Model individual-level interactions and disease progression.  
2. Compare epidemic trajectories under different prevention and treatment scenarios.  
3. Provide an interactive dashboard to explore parameter influence on outcomes.
""")

# Explanatory Modeling
with st.expander("What is Explanatory Modeling?"):
    st.markdown("""
- We observe a phenomenon (HIV transmission patterns).  
- We hypothesize key drivers (treatment, condom use, testing).  
- We build an approximate representation (agent-based model).  
- If the model replicates real-world behavior, our hypotheses gain support.
""")

# --- MODEL CLASSES ---------------------------------------------------------
class HIVStatus(IntEnum):
    SUSCEPTIBLE        = 0
    INFECTED_UNTREATED = 1
    INFECTED_TREATED   = 2
    AIDS_UNTREATED     = 3
    AIDS_TREATED       = 4
    DEAD               = 5

class PersonAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.hiv_status = HIVStatus.SUSCEPTIBLE
        self.infection_time = None
        self.progression_time = None
        self.knows_status = False
        self.on_treatment = False

    def move(self):
        neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        self.model.grid.move_agent(self, self.random.choice(neighbors))

    def update_disease(self):
        if self.hiv_status in (HIVStatus.SUSCEPTIBLE, HIVStatus.DEAD): return
        # Testing & treatment
        if not self.knows_status and self.random.random() < self.model.testing_rate:
            self.knows_status = True
            if self.random.random() < self.model.treatment_coverage:
                self.on_treatment = True
                if self.hiv_status == HIVStatus.INFECTED_UNTREATED:
                    self.hiv_status = HIVStatus.INFECTED_TREATED
                elif self.hiv_status == HIVStatus.AIDS_UNTREATED:
                    self.hiv_status = HIVStatus.AIDS_TREATED
        # Progression
        if self.hiv_status in (HIVStatus.INFECTED_UNTREATED, HIVStatus.INFECTED_TREATED):
            elapsed = self.model.schedule.time - self.infection_time
            eff = elapsed/2 if self.on_treatment else elapsed
            if eff >= self.progression_time:
                self.hiv_status = HIVStatus.AIDS_TREATED if self.on_treatment else HIVStatus.AIDS_UNTREATED
        # Death
        drate = {
            HIVStatus.AIDS_UNTREATED: self.model.death_rate_untreated,
            HIVStatus.AIDS_TREATED:   self.model.death_rate_treated,
            HIVStatus.INFECTED_UNTREATED: self.model.death_rate_untreated/10
        }.get(self.hiv_status, 0)
        if self.random.random() < drate:
            self.hiv_status = HIVStatus.DEAD
            self.model.deaths += 1

    def interact(self):
        if self.hiv_status not in (
            HIVStatus.INFECTED_UNTREATED, HIVStatus.INFECTED_TREATED,
            HIVStatus.AIDS_UNTREATED, HIVStatus.AIDS_TREATED): return
        mates = self.model.grid.get_cell_list_contents([self.pos])
        for other in mates:
            if other.hiv_status != HIVStatus.SUSCEPTIBLE: continue
            prob = self.model.base_transmission_rate
            if self.hiv_status == HIVStatus.AIDS_UNTREATED: prob *= 2.5
            elif self.hiv_status == HIVStatus.AIDS_TREATED: prob *= 0.5
            if self.on_treatment: prob *= (1 - self.model.treatment_effectiveness)
            if self.random.random() < prob:
                other.hiv_status = HIVStatus.INFECTED_UNTREATED
                other.infection_time = self.model.schedule.time
                other.progression_time = self.model.get_progression_time()
                self.model.infections += 1

    def step(self):
        self.update_disease()
        self.move()
        self.interact()

class HIVModel(Model):
    def __init__(self, N, width, height, init_inf,
                 base_trans, condom_use, treat_cov, treat_eff,
                 prog_time, prog_sd, dr_unc, dr_tr, test_rate):
        super().__init__()
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        # Parameters
        self.base_transmission_rate = base_trans
        self.condom_use_rate = condom_use
        self.treatment_coverage = treat_cov
        self.treatment_effectiveness = treat_eff
        self.death_rate_untreated = dr_unc/365
        self.death_rate_treated = dr_tr/365
        self.testing_rate = test_rate/365
        self.prog_time = prog_time
        self.prog_sd = prog_sd
        self.infections = 0
        self.deaths = 0
        # Initialize agents
        for i in range(N):
            agent = PersonAgent(i, self)
            self.schedule.add(agent)
            x, y = self.random.randrange(width), self.random.randrange(height)
            self.grid.place_agent(agent, (x, y))
        # Seed infections
        initially = self.random.sample(self.schedule.agents, init_inf)
        for a in initially:
            a.hiv_status = HIVStatus.INFECTED_UNTREATED
            a.infection_time = 0
            a.progression_time = self.get_progression_time()
        # Data collector
        self.datacollector = DataCollector(
            model_reporters={
                'Step': lambda m: m.schedule.time,
                'Sus': lambda m: sum(a.hiv_status==HIVStatus.SUSCEPTIBLE for a in m.schedule.agents),
                'Inf_U': lambda m: sum(a.hiv_status==HIVStatus.INFECTED_UNTREATED for a in m.schedule.agents),
                'Inf_T': lambda m: sum(a.hiv_status==HIVStatus.INFECTED_TREATED for a in m.schedule.agents),
                'AIDS_U': lambda m: sum(a.hiv_status==HIVStatus.AIDS_UNTREATED for a in m.schedule.agents),
                'AIDS_T': lambda m: sum(a.hiv_status==HIVStatus.AIDS_TREATED for a in m.schedule.agents),
                'Dead': lambda m: sum(a.hiv_status==HIVStatus.DEAD for a in m.schedule.agents),
                'NewInf': lambda m: m.infections,
                'Deaths': lambda m: m.deaths
            }
        )
    def get_progression_time(self):
        return int(np.random.normal(self.prog_time, self.prog_sd))
    def step(self):
        self.infections = 0
        self.deaths = 0
        self.schedule.step()
        self.datacollector.collect(self)

# --- STREAMLIT SIDEBAR -----------------------------------------------------
st.sidebar.header('Simulation Parameters')
cols = st.sidebar.columns(2)
params = {
    'N': cols[0].slider('Population', 100, 2000, 500, help='Total agents'),
    'init_inf': cols[0].slider('Initial Infected', 1, 50, 10, help='Seed infections'),
    'base_trans': cols[0].slider('Transmission Prob.', 0.01, 1.0, 0.1, 0.01, help='Per-contact risk'),
    'condom_use': cols[1].slider('Condom Use Rate', 0.0, 1.0, 0.5, 0.05, help='Reduces transmission'),
    'treat_cov': cols[1].slider('Treatment Coverage', 0.0, 1.0, 0.2, 0.05, help='Proportion tested and treated'),
    'treat_eff': cols[1].slider('Treatment Efficacy', 0.0, 1.0, 0.96, 0.01, help='Reduces progression'),
    'prog_time': cols[0].number_input('Prog. Time (d)', 365, 3650, 3650, help='Untreated → AIDS'),
    'prog_sd': cols[0].number_input('Prog. SD (d)', 100, 2000, 1095, help='Variation in progression'),
    'dr_unc': cols[1].slider('Death Rate Untreated', 0.0, 1.0, 0.1, 0.01, help='Annual untreated'),
    'dr_tr': cols[1].slider('Death Rate Treated', 0.0, 1.0, 0.02, 0.005, help='Annual treated'),
    'test_rate': cols[1].slider('Testing Rate', 0.0, 1.0, 0.3, 0.01, help='Annual probability'),
    'days': st.sidebar.slider('Days', 10, 1000, 100, help='Simulation length')
}
if 'stop' not in st.session_state:
    st.session_state.stop = False

col_run, col_stop = st.sidebar.columns(2)
col_run.button('Start') and st.session_state.__setitem__('stop', False)
col_stop.button('Stop') and st.session_state.__setitem__('stop', True)

# --- MAIN CONTENT AREA with Animation -------------------------------------
ts_placeholder = st.empty()
gr_placeholder = st.empty()

# Animation loop
if st.sidebar.button('Run'):
    days = params.pop('days')
    model = HIVModel(**params, width=20, height=20)
    steps, data = [], {k: [] for k in ['Sus','Inf_U','Inf_T','AIDS_U','AIDS_T','Dead']}
    for _ in range(days):
        if st.session_state.stop:
            break
        model.step()
        rec = model.datacollector.get_model_vars_dataframe().iloc[-1]
        steps.append(rec['Step'])
        for k in data: data[k].append(rec[k])
        # Smaller plots
        fig1, ax1 = plt.subplots(figsize=(3,2))
        for k in data: ax1.plot(steps, data[k], label=k)
        ax1.legend(fontsize='x-small')
        ts_placeholder.pyplot(fig1)
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(3,2))
        w, h = model.grid.width, model.grid.height
        arr = np.full((w,h), np.nan)
        for cell, coords in model.grid.coord_iter():
            x, y = coords
            if cell: arr[x,y] = cell[-1].hiv_status
        ax2.imshow(arr, vmin=0, vmax=5, cmap='tab20')
        ax2.axis('off')
        gr_placeholder.pyplot(fig2)
        plt.close(fig2)
        time.sleep(0.05)

# Sidebar branding
st.sidebar.markdown('---')
st.sidebar.markdown('**HIV ABM Modeling**')
st.sidebar.caption('Powered by Mesa & Streamlit')

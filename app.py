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
# N: Total number of agents
# init_inf: Initial infected
# base_trans: Base transmission probability
# condom_use: Condom use rate (reduces transmission)
# treat_cov: Treatment coverage (proportion treated once tested)
# treat_eff: Treatment efficacy (slows progression)
# prog_time: Mean days from infection→AIDS
# prog_sd: SD for progression time
# dr_unc: Annual death rate untreated AIDS
# dr_tr: Annual death rate treated AIDS
# test_rate: Annual testing probability
# days: Simulation length in days

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
        neigh = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        self.model.grid.move_agent(self, self.random.choice(neigh))

    def update_disease(self):
        if self.hiv_status in (HIVStatus.SUSCEPTIBLE, HIVStatus.DEAD):
            return
        # testing & treatment
        if not self.knows_status and self.random.random() < self.model.testing_rate:
            self.knows_status = True
            if self.random.random() < self.model.treatment_coverage:
                self.on_treatment = True
                if self.hiv_status == HIVStatus.INFECTED_UNTREATED:
                    self.hiv_status = HIVStatus.INFECTED_TREATED
                elif self.hiv_status == HIVStatus.AIDS_UNTREATED:
                    self.hiv_status = HIVStatus.AIDS_TREATED
        # progression
        if self.hiv_status in (HIVStatus.INFECTED_UNTREATED, HIVStatus.INFECTED_TREATED):
            elapsed = self.model.schedule.time - self.infection_time
            effective = elapsed / 2 if self.on_treatment else elapsed
            if effective >= self.progression_time:
                self.hiv_status = (HIVStatus.AIDS_TREATED if self.on_treatment 
                                   else HIVStatus.AIDS_UNTREATED)
        # death
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
            HIVStatus.AIDS_UNTREATED, HIVStatus.AIDS_TREATED):
            return
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        for other in cellmates:
            if other.hiv_status != HIVStatus.SUSCEPTIBLE:
                continue
            prob = self.model.base_transmission_rate
            if self.hiv_status == HIVStatus.AIDS_UNTREATED:
                prob *= 2.5
            elif self.hiv_status == HIVStatus.AIDS_TREATED:
                prob *= 0.5
            if self.on_treatment:
                prob *= (1 - self.model.treatment_effectiveness)
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

        # parameters
        self.base_transmission_rate = base_trans
        self.condom_use_rate = condom_use
        self.treatment_coverage = treat_cov
        self.treatment_effectiveness = treat_eff
        self.death_rate_untreated = dr_unc / 365
        self.death_rate_treated   = dr_tr / 365
        self.testing_rate         = test_rate / 365
        self.prog_time            = prog_time
        self.prog_sd              = prog_sd
        self.infections = 0
        self.deaths     = 0

        # create agents
        for i in range(N):
            a = PersonAgent(i, self)
            self.schedule.add(a)
            x, y = self.random.randrange(width), self.random.randrange(height)
            self.grid.place_agent(a, (x, y))

        # seed initial infections
        infected = self.random.sample(self.schedule.agents, init_inf)
        for a in infected:
            a.hiv_status       = HIVStatus.INFECTED_UNTREATED
            a.infection_time   = 0
            a.progression_time = self.get_progression_time()

        # data collector
        self.datacollector = DataCollector(
            model_reporters={
                'Step':   lambda m: m.schedule.time,
                'Sus':    lambda m: sum(a.hiv_status==HIVStatus.SUSCEPTIBLE for a in m.schedule.agents),
                'Inf_U':  lambda m: sum(a.hiv_status==HIVStatus.INFECTED_UNTREATED for a in m.schedule.agents),
                'Inf_T':  lambda m: sum(a.hiv_status==HIVStatus.INFECTED_TREATED for a in m.schedule.agents),
                'AIDS_U': lambda m: sum(a.hiv_status==HIVStatus.AIDS_UNTREATED for a in m.schedule.agents),
                'AIDS_T': lambda m: sum(a.hiv_status==HIVStatus.AIDS_TREATED for a in m.schedule.agents),
                'Dead':   lambda m: sum(a.hiv_status==HIVStatus.DEAD for a in m.schedule.agents),
                'NewInf': lambda m: m.infections,
                'Deaths': lambda m: m.deaths
            }
        )

    def get_progression_time(self):
        return int(np.random.normal(self.prog_time, self.prog_sd))

    def step(self):
        self.infections = 0
        self.deaths     = 0
        self.schedule.step()
        self.datacollector.collect(self)

# --- SIDEBAR ---------------------------------------------------------------
st.sidebar.header("Simulation Parameters")
c1, c2 = st.sidebar.columns(2)
params = {
    'N':          c1.slider("Population", 100, 2000, 500),
    'init_inf':   c1.slider("Initial Infected", 1, 50, 10),
    'base_trans': c1.slider("Transmission Prob.", 0.01, 1.0, 0.1, 0.01),
    'condom_use': c2.slider("Condom Use Rate", 0.0, 1.0, 0.5, 0.05),
    'treat_cov':  c2.slider("Treatment Coverage", 0.0, 1.0, 0.2, 0.05),
    'treat_eff':  c2.slider("Treatment Efficacy", 0.0, 1.0, 0.96, 0.01),
    'prog_time':  c1.number_input("Prog. Time (d)", 365, 3650, 3650),
    'prog_sd':    c1.number_input("Prog. SD (d)", 100, 2000, 1095),
    'dr_unc':     c2.slider("Death Rate Untreated", 0.0, 1.0, 0.1, 0.01),
    'dr_tr':      c2.slider("Death Rate Treated", 0.0, 1.0, 0.02, 0.005),
    'test_rate':  c2.slider("Testing Rate", 0.0, 1.0, 0.3, 0.01),
    'days':       st.sidebar.slider("Days", 10, 1000, 100)
}

if 'stop' not in st.session_state:
    st.session_state.stop = False

rb, sb = st.sidebar.columns(2)
rb.button("Start") and st.session_state.__setitem__("stop", False)
sb.button("Stop")  and st.session_state.__setitem__("stop", True)

# --- MAIN CONTENT AREA -----------------------------------------------------
ts_ph = st.empty()
gr_ph = st.empty()

# prepare empty time-series DataFrame & chart
df_ts = pd.DataFrame(columns=['Sus','Inf_U','Inf_T','AIDS_U','AIDS_T','Dead'])
chart = ts_ph.line_chart(df_ts)

if st.sidebar.button("Run"):
    days = params.pop('days')
    model = HIVModel(**params, width=20, height=20)

    for _ in range(days):
        if st.session_state.stop:
            break

        model.step()
        rec = model.datacollector.get_model_vars_dataframe().iloc[-1]

        # update time-series
        row = {k: rec[k] for k in ['Sus','Inf_U','Inf_T','AIDS_U','AIDS_T','Dead']}
        df_ts = pd.concat([df_ts, pd.DataFrame([row])], ignore_index=True)
        chart.add_rows(pd.DataFrame([row]))

        # update grid: colored categorical via Matplotlib
        fig2, ax2 = plt.subplots(figsize=(2, 2))
        w, h = model.grid.width, model.grid.height
        arr = np.full((w,h), -1)
        for item in model.grid.coord_iter():
            if len(item) == 3:
                cell, x, y = item
            else:
                cell, coords = item
                x, y = coords
            if cell:
                arr[x, y] = cell[-1].hiv_status

        cmap = plt.get_cmap('tab20', 6)
        ax2.imshow(arr, vmin=0, vmax=5, cmap=cmap)
        ax2.axis('off')
        gr_ph.pyplot(fig2)
        plt.close(fig2)

        time.sleep(0.05)

st.sidebar.markdown("---")
st.sidebar.caption("Powered by Mesa & Streamlit")

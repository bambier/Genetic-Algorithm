#! /usr/bin/env python3.12
# Genetic Algorithm for Optimizing a System
from typing import List, Tuple, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.integrate import odeint

# Seed for reproducibility
np.random.seed(0)

# Data Types
Genome = npt.NDArray[np.float32]
Population = List[Genome]

# Data structure for initial conditions


class Conditions(TypedDict):
    X_v: float  # Cell concentration
    Glc: float  # Glucose concentration
    Gln: float  # Glutamine concentration
    Lac: float  # Lactate concentration
    Amm: float  # Ammonia concentration
    MAb: float  # Monoclonal antibody concentration
    Glc_in: float  # Inlet glucose concentration
    Gln_in: float  # Inlet glutamine concentration
    V: float  # Volume of the system


# Constants
MU_MAX = 1.09  # Maximum specific growth rate (d^-1)
K_D_MAX = 0.69  # Maximum dilution rate (d^-1)
Y_XV__GLC = 1.09e8  # Yield coefficient for glucose (cells/mmol)
Y_XV__GLN = 3.8e8  # Yield coefficient for glutamine (cells/mmol)
M_GLC = 0.17e-8  # Maintenance coefficient for glucose (mmol / cell / d^-1)
K_M_GLC = 19.0  # Half-saturation constant for glucose (mM)
K_GLC = 1.0  # Threshold concentration for glucose (mM)
K_GLN = 1.0  # Threshold concentration for glutamine (mM)
ALPHA_0 = 2.57e-8  # Production rate constant for monoclonal antibody (mmol / cell / d^-1)
K_MU = 0.02  # Constant for specific growth rate (d^-1)
BETA = 0.35e-8  # Additional production rate constant for monoclonal antibody (mmol / cell / d^-1)
K_D_LAC = 0.01  # Dilution rate constant for lactate (mM^-1)
K_D_AMM = 0.06  # Dilution rate constant for ammonia (mM^-1)
K_D_GLN = 0.02  # Dilution rate constant for glutamine (mM)
Y_LAC__GLC = 1.8  # Yield coefficient for lactate from glucose (mmol / mmol)
Y_AMM__GLN = 0.85  # Yield coefficient for ammonia from glutamine (mmol / mmol)
SELECTION_RATE = 0.08  # Selection rate for genetic algorithm
CROSSOVER_RATE = 0.6  # Crossover rate for genetic algorithm
MUTATION_RATE = 0.05  # Mutation rate for genetic algorithm
FEED_LIMIT = 0.5  # Upper limit for feed rates
VOLUME_LIMIT = 2e3  # Upper limit for volume
DAYS = 10  # Number of days for simulation
INDIVIDUALS = 1000  # Number of individuals in the population

# Initial conditions for the simulation
init_conditions: Conditions = {
    "X_v": 2.0e8,
    "Glc": 25,
    "Gln": 4,
    "Lac": 0,
    "Amm": 0,
    "MAb": 0,
    "Glc_in": 25,
    "Gln_in": 25,
    "V": 0.79,
}


# Calculate kinetic expressions based on current concentrations
def kinetic_expressions(Glc: float, Gln: float, Lac: float, Amm: float) -> Tuple[float, float, float, float, float, float, float]:
    # Specific growth rate
    mu = MU_MAX * (Glc / (K_GLC + Glc)) * (Gln / (K_GLN + Gln))
    # Dilution rate
    k_d = K_D_MAX * ((MU_MAX - (K_D_LAC * Lac)) * 0.1) * ((MU_MAX - (K_D_AMM * Amm)) * 0.1) * (K_D_GLN / (K_D_GLN + Gln))
    # Productivities
    q_gln = mu / Y_XV__GLN
    q_glc = mu / Y_XV__GLC + (M_GLC * (Glc / (K_M_GLC + Glc)))
    q_lac = Y_LAC__GLC * q_glc
    q_amm = Y_AMM__GLN * q_gln
    alpha_prime = ALPHA_0 / (K_MU + mu)
    q_MAb = (alpha_prime * mu) + BETA
    return tuple([mu, k_d, q_gln, q_glc, q_lac, q_amm, q_MAb])


# ODE function to describe system dynamics
def multi_feed_expressions(y: List[float], t, F1, F2) -> Tuple[float, float, float, float, float, float, float]:
    X_v, Glc,  Gln, Lac, Amm, MAb, Glc_in, Gln_in, V = y
    mu, k_d, q_gln, q_glc, q_lac, q_amm, q_MAb = kinetic_expressions(Glc=Glc, Gln=Gln, Lac=Lac, Amm=Amm)

    # abstraction
    FFV = (F1 + F2) / V

    # ODEs for each variable
    dX_v__dt = ((mu - k_d) * X_v) - (FFV * X_v)
    dGlc__dt = ((Glc_in - Glc) * FFV) - (q_glc * X_v)
    dGln__dt = ((Gln_in - Gln) * FFV) - (q_gln * X_v)
    dLac__dt = q_lac * X_v - (FFV * Lac)
    dAmm__dt = q_amm * X_v - (FFV * Amm)
    dMAb__dt = q_MAb * X_v - (FFV * MAb)
    dV__dt = F1 + F2

    return np.array([dX_v__dt, dGlc__dt, dGln__dt, dLac__dt, dAmm__dt, dMAb__dt, dV__dt, Glc_in, Gln_in], dtype=np.float32)


# Fitness function to evaluate performance
def fitness(MAb: float, V: float, F1: np.float32, F2: np.float32) -> float:
    F = F1 + F2
    # Check if feed and volume are within acceptable limits
    if 0. <= F <= FEED_LIMIT:
        if 0. < V <= VOLUME_LIMIT:
            return MAb * V
    return 0.


# Perform roulette wheel selection to choose individuals for the next generation
def roulette_wheel_selection(fitness_scores: List[float], population: Population, y: np.ndarray) -> Population:
    total_fitness = sum(fitness_scores)
    if total_fitness == 0:
        # If total fitness is zero, assign equal probabilities
        probabilities = None
    else:
        probabilities = [f / total_fitness for f in fitness_scores]

    # Choose indices based on probabilities
    chosen_indices = np.random.choice(len(population), size=len(population), p=probabilities)

    # Retrieve the chosen populations and their corresponding y values
    chosen_populations = [population[i] for i in chosen_indices]
    choosen_ys = [y[i][5] for i in chosen_indices] # Indice 5 is [MAb]

    # Sort populations based on their corresponding fitness values (MAb concentration)
    sorted_populations = [pop for _, pop in sorted(zip(choosen_ys, chosen_populations), key=lambda x: x[0], reverse=True)]

    return sorted_populations


# Perform crossover operation between two parent genomes
def crossover(parent1: Genome, parent2: Genome) -> Tuple[Genome, Genome]:
    if np.random.rand() < CROSSOVER_RATE:
        child1 = np.array([parent1[0], parent2[1]])
        child2 = np.array([parent2[1], parent1[0]])
        return child1, child2
    else:
        return parent1, parent2


# Perform mutation on a genome
def mutation(genome: Genome) -> Genome:
    for i in range(len(genome)):
        if np.random.rand() < MUTATION_RATE:
            genome[i] = np.random.uniform(0, FEED_LIMIT)
    return genome


# Simulate the system evolution with given initial conditions
def evolution(initial_conditions: Conditions, F1: np.float32, F2: np.float32, t: np.ndarray) -> Tuple[float, float, float, float, float, float, float]:
    y0 = list(initial_conditions.values())
    # Integrate ODEs over time points
    y: np.ndarray = odeint(multi_feed_expressions, y0, t, args=(F1, F2))
    return y[-1]  # Return the final state after integration


# Main function to run the genetic algorithm
def main():
    populations = np.random.uniform(0, FEED_LIMIT, (INDIVIDUALS, 2))
    time_points = np.linspace(0, 1, 10)  # Time points for integration
    MAb_consontrations = []
    popy = []

    for day in range(DAYS):
        print(f"Day: {day + 1} of {DAYS}")
        fitness_scores = []
        new_population = []
        ys = []

        for i, individual in enumerate(populations):
            F1 = individual[0]
            F2 = individual[1]
            y = evolution(init_conditions, F1=F1, F2=F2, t=time_points)
            fitness_score = fitness(y[-4], y[-3], F1, F2)

            fitness_scores.append(fitness_score)
            new_population.append(individual)
            ys.append(y)

        ys = np.array(ys)
        selected_population = roulette_wheel_selection(fitness_scores, new_population, ys)

        # Apply crossover and mutation to create a new population
        new_population = []
        for i in range(0, len(selected_population), 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i + 1]
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutation(child1))
            new_population.append(mutation(child2))

        populations = np.array(new_population[:INDIVIDUALS])

        p = populations[0]

        X_v, Glc, Gln, Lac, Amm, MAb, V, Glc_in, Gln_in = evolution(init_conditions, p[0], p[1], time_points)
        init_conditions.update({
            "X_v": X_v,
            "Glc": Glc,
            "Gln": Gln,
            "Lac": Lac,
            "Amm": Amm,
            "MAb": MAb,
            "Glc_in": Glc_in,
            "Gln_in": Gln_in,
            "V": V,
        })
        MAb_consontrations.append(MAb)
        popy.append(p)

    ploting(MAb_consontrations, np.array(popy))





# Plotting function for results visualization
def ploting(MAb_consontrations, populations: np.ndarray):
    fig, ax = plt.subplots()
    F1 = populations[:, 0]
    F2 = populations[:, 1]
    ax.plot(range(DAYS), MAb_consontrations, color="red", label="MAb Concentration")
    ax.plot(range(DAYS), F1, color="blue", label="Feed Rate F1")
    ax.plot(range(DAYS), F2, color="green", label="Feed Rate F2")

    ax.legend()
    ax.set_xlabel("Days")
    ax.set_ylabel("Concentration")
    file_name = input("Enter file named to be saved >> ") or "days"
    plt.savefig(f"{file_name}.png")
    print(f'Plot was saved in "{file_name}.png" file')


# Entry point of the script
if __name__ == "__main__":
    main()



import random
import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Guard against re-registration if module is reloaded
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)

def _evaluate(individual, X, y, penalty=0.001):
    selected = [i for i, b in enumerate(individual) if b == 1]
    
    # Soft minimum — require at least 2 features
    if len(selected) < 2:
        return (9999,)
    
    X_sub = X.iloc[:, selected]
    model = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1)
    scores = cross_val_score(model, X_sub, y, cv=3,
                             scoring='neg_root_mean_squared_error')
    rmse = -scores.mean()
    # Penalty is tiny — RMSE should dominate, not feature count
    return (rmse + penalty * len(selected),)

def run_genetic_algorithm(X, y, n_gen=20, pop_size=50,
                           cx_prob=0.7, mut_prob=0.3):
    n = X.shape[1]
    tb = base.Toolbox()
    tb.register("bit",        random.randint, 0, 1)
    tb.register("individual", tools.initRepeat, creator.Individual, tb.bit, n=n)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("evaluate",   _evaluate, X=X, y=y)
    tb.register("mate",       tools.cxUniform, indpb=0.5)  # Changed from cxTwoPoint
    tb.register("mutate",     tools.mutFlipBit, indpb=0.1) # Increased from 0.05
    tb.register("select",     tools.selTournament, tournsize=3)

    # Initialize population with at least half bits set to 1
    # This prevents starting with mostly empty individuals
    def init_individual():
        ind = creator.Individual(
            [1 if random.random() > 0.3 else 0 for _ in range(n)]
        )
        return ind

    toolbox_pop = [init_individual() for _ in range(pop_size)]

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    print(f"\n--- Genetic Algorithm ---")
    print(f"Features={n} | Population={pop_size} | Generations={n_gen}")

    pop, log = algorithms.eaSimple(
        toolbox_pop, tb, cxpb=cx_prob, mutpb=mut_prob,
        ngen=n_gen, stats=stats, halloffame=hof, verbose=True
    )

    best = hof[0]
    idx = [i for i, b in enumerate(best) if b == 1]
    selected = X.columns[idx].tolist()
    print(f"\nGA complete → {len(selected)} features | "
          f"Fitness: {best.fitness.values[0]:.4f}")
    return selected, log

def apply_ga_selection(X, y):
    selected, log = run_genetic_algorithm(X, y)
    return X[selected], selected, log
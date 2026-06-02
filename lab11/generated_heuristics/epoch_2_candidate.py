import numpy as np
from typing import Callable
def new_metaheuristic(
    function: Callable[[np.ndarray], float],
    bounds: list[tuple[float, float]],
    budget: int
) -> tuple[float, np.ndarray]:
    D = len(bounds)
    pop_size = 40  # Increased population size
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])

    pop = np.random.uniform(lower, upper, (pop_size, D))
    fitness = np.array([function(ind) for ind in pop])

    best_idx = np.argmin(fitness)
    best_pos = pop[best_idx].copy()
    best_val = fitness[best_idx]

    budget_used = pop_size
    F = 0.8  # Increased mutation weight for more exploration
    CR = 0.7 # Slightly decreased crossover probability

    while budget_used < budget:
        for i in range(pop_size):
            if budget_used >= budget: break

            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]

            mutant = np.clip(a + F * (b - c), lower, upper)
            cross_points = np.random.rand(D) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, D)] = True

            trial = np.where(cross_points, mutant, pop[i])
            f_trial = function(trial)
            budget_used += 1

            if f_trial < fitness[i]:
                pop[i] = trial
                fitness[i] = f_trial
                if f_trial < best_val:
                    best_val = f_trial
                    best_pos = trial.copy()

    return best_val, best_pos
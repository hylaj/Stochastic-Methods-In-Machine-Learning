import numpy as np
from typing import Callable

def new_metaheuristic(
    function: Callable[[np.ndarray], float],
    bounds: list[tuple[float, float]],
    budget: int
) -> tuple[float, np.ndarray]:
    D = len(bounds)
    pop_size = 30  # Changed from 20 to 30
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])

    # Ensure pop_size is at least 4 for DE/rand/1/bin to work (need 3 distinct individuals + current)
    # If budget is too small for initial population, adjust pop_size or handle it
    if budget < pop_size:
        # If budget is extremely small, we might not even be able to initialize the population.
        # For simplicity and to meet the budget constraint, we can set pop_size = budget
        # and proceed, though DE usually requires pop_size >= 4.
        # If budget < 4, a simple random search or returning an initial random point might be more robust.
        # Given the problem context, we assume budget allows for a reasonable pop_size.
        # Let's ensure pop_size is at least 4, or budget if budget is smaller.
        pop_size = max(min(budget, pop_size), 4) if budget > 0 else 1
        if budget == 0:
            # Handle edge case where budget is 0
            if D == 0: return float('inf'), np.array([])
            initial_pos = np.random.uniform(lower, upper, D)
            return function(initial_pos), initial_pos


    pop = np.random.uniform(lower, upper, (pop_size, D))
    
    # Handle the case where budget < pop_size after initialization
    # Only evaluate up to the budget
    fitness = np.full(pop_size, np.inf)
    budget_used = 0
    for i in range(pop_size):
        if budget_used < budget:
            fitness[i] = function(pop[i])
            budget_used += 1
        else:
            break # Stop evaluating if budget is exhausted

    best_idx = np.argmin(fitness[:budget_used]) if budget_used > 0 else -1
    if best_idx != -1:
        best_pos = pop[best_idx].copy()
        best_val = fitness[best_idx]
    else:
        # This case should ideally not be reached if budget >= 1 and D > 0
        # If budget is 0, we already handled it
        if D == 0: return float('inf'), np.array([]) # No dimensions to optimize
        best_pos = np.random.uniform(lower, upper, D) # Fallback to a random point
        best_val = float('inf')


    F = 0.8   # Changed from 0.5 to 0.8 (Mutation weight)
    CR = 0.5  # Changed from 0.9 to 0.5 (Crossover probability)

    while budget_used < budget:
        for i in range(pop_size):
            if budget_used >= budget: break

            # Ensure there are enough distinct individuals for mutation
            if pop_size < 4: # Need at least 3 for mutation + current
                # If pop_size is too small, fallback to a simpler strategy or break
                # For this problem, we assumed pop_size would be >= 4 or handled.
                # If it shrinks unexpectedly, this needs re-evaluation.
                # Given pop_size = 30 initially, this check mostly applies if budget was very low.
                break 

            idxs = [idx for idx in range(pop_size) if idx != i]
            # Ensure idxs has at least 3 elements
            if len(idxs) < 3: 
                break # Cannot perform DE/rand/1/bin, break loop

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
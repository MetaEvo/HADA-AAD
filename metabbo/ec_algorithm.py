"""Differential Evolution optimizer for cocoex problems.

This module provides a `DEOptimizer` class that is controlled by a DQN agent
for hyperparameter optimization. The algorithm accepts an optional `controller`
object implementing the ParamController interface to allow dynamic parameter
adjustment (F, CR).
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SEED


def evaluate_problem(problem, x):
    """Evaluate problem, handling both MetaBox (problem.func) and cocoex (problem(x)) styles."""
    if not isinstance(x, np.ndarray):
        x = np.asarray(x, dtype=float)
    
    if hasattr(problem, 'func'):
        # MetaBox style: use problem.func(x), which REQUIRES 2D input [NP, dim]
        if x.ndim == 1:
            x = x.reshape(1, -1)
        result = problem.func(x)
        return float(np.mean(np.array(result).flatten()))
    else:
        # cocoex style: problem(x)
        return float(problem(x))


class DEOptimizer:
    """Differential Evolution optimizer with DQN controller support.
    
    The controller outputs F and CR parameters dynamically during optimization.
    """
    def __init__(self, dim: int, lower_bounds, upper_bounds, max_evals: int,
                 params: Optional[Dict[str, Any]] = None, controller=None, recorder=None,
                 swarm_size: int = None):
        self.dim = int(dim)
        if lower_bounds is not None:
            self.lower_bounds = np.asarray(lower_bounds, dtype=float)
            if self.lower_bounds.ndim == 0:
                self.lower_bounds = np.full(self.dim, float(self.lower_bounds))
        else:
            self.lower_bounds = -5.0 * np.ones(self.dim)
        
        if upper_bounds is not None:
            self.upper_bounds = np.asarray(upper_bounds, dtype=float)
            if self.upper_bounds.ndim == 0:
                self.upper_bounds = np.full(self.dim, float(self.upper_bounds))
        else:
            self.upper_bounds = 5.0 * np.ones(self.dim)
        
        self.max_evals = int(max_evals)
        self.params = dict(params) if params is not None else {}
        self.controller = controller
        self.recorder = recorder
        self.stats = {}
        self.pop_size = swarm_size if swarm_size is not None else max(10, 5 * self.dim)
        self.F = float(self.params.get('F', 0.5))
        self.CR = float(self.params.get('CR', 0.5))

    def optimize(self, problem, x0=None, train=True, seed=None) -> Tuple[list, Dict[str, Any]]:
        if self.controller is not None:
            self.controller.reset({'dim': self.dim, 'lower': self.lower_bounds, 'upper': self.upper_bounds, 'max_evals': self.max_evals})

        pop_size = self.pop_size
        lb = self.lower_bounds
        ub = self.upper_bounds
        
        effective_seed = seed if seed is not None else SEED
        rng = np.random.RandomState(effective_seed)

        # Initialize population - uniform random initialization
        pop = rng.uniform(lb, ub, (pop_size, self.dim))
        
        # Evaluate initial population
        evals = 0
        fitness = np.full(pop_size, np.inf)
        
        for i in range(pop_size):
            if evals >= self.max_evals:
                break
            fitness[i] = evaluate_problem(problem, pop[i])
            evals += 1

        # Track best
        best_idx = int(np.argmin(fitness))
        gbest_f = float(fitness[best_idx])
        gbest = pop[best_idx].copy()
        
        initial_gbest_f = float(gbest_f)
        initial_population_positions = pop.copy()
        initial_population_fitness = fitness.copy()

        # Main loop
        iteration = 0
        done = False
        
        while not done:
            iteration += 1
            params = dict(self.params)
            if self.controller is not None:
                obs = {'iteration': iteration, 'evals': evals, 'gbest_f': gbest_f, 
                       'initial_gbest_f': initial_gbest_f, 'max_evals': self.max_evals}
                step_params = self.controller.step(obs)
                if isinstance(step_params, dict):
                    params.update(step_params)
            
            F = float(params.get('F', self.F))
            CR = float(params.get('CR', self.CR))
            
            prev_gbest_f = gbest_f
            
            # Generate trial vectors
            new_pop = np.copy(pop)
            for i in range(pop_size):
                # Tournament selection (tournsize=3, does NOT exclude current individual)
                candidates = [idx for idx in range(pop_size) if idx != i]
                if len(candidates) < 3:
                    indices = list(range(pop_size))
                else:
                    indices = rng.choice(candidates, size=3, replace=False)
                
                a, b, c = pop[indices[0]], pop[indices[1]], pop[indices[2]]
                
                # Mutation and crossover
                j_rand = rng.randint(0, self.dim)
                for j in range(self.dim):
                    if rng.random() < CR or j == j_rand:
                        new_pop[i, j] = a[j] + F * (b[j] - c[j])
                
                # Boundary handling: clip
                new_pop[i] = np.clip(new_pop[i], lb, ub)
            
            # Evaluate trial vectors
            for i in range(pop_size):
                if evals >= self.max_evals:
                    done = True
                    break
                trial_f = evaluate_problem(problem, new_pop[i])
                evals += 1
                
                # Greedy selection
                if trial_f < fitness[i]:
                    pop[i] = new_pop[i].copy()
                    fitness[i] = trial_f
                    if trial_f < gbest_f:
                        gbest_f = trial_f
                        gbest = new_pop[i].copy()
            
            # Compute reward: 1 if improved, 0 otherwise
            improved = 1.0 if gbest_f < prev_gbest_f else 0.0
            
            # Store transition in controller's replay buffer (only during training)
            if self.controller is not None and train:
                next_obs = {'iteration': iteration, 'evals': evals, 'gbest_f': gbest_f, 
                           'initial_gbest_f': initial_gbest_f, 'max_evals': self.max_evals}
                done = (evals >= self.max_evals)
                if hasattr(self.controller, 'store_reward'):
                    self.controller.store_reward(improved, next_obs, done)
            
            done = (evals >= self.max_evals)

        self.stats = {
            'gbest': gbest.tolist(),
            'gbest_f': float(gbest_f),
            'evals': int(evals),
            'iterations': int(iteration),
            'initial_gbest_f': initial_gbest_f,
            'initial_population_positions': initial_population_positions.tolist(),
            'initial_population_fitness': initial_population_fitness.tolist(),
        }
        return gbest.tolist(), self.stats
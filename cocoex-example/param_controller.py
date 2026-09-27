"""Parameter controller interface and simple implementations.

Defines `ParamController` abstract interface and example controllers that can
be injected into `PSOOptimizer` to adjust parameters dynamically. This is
the hook that a DQN-based controller should implement.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any

# Import global configuration
import sys
import os
try:
    sys.path.insert(0, '/hyperagents')
    from config import set_global_seed, SEED
    # Set global seed at module load time
    set_global_seed()
except Exception:
    # Fallback if config module is not available
    SEED = 42
    def set_global_seed(*args, **kwargs):
        pass


class ParamController(ABC):
    @abstractmethod
    def reset(self, problem_meta: Dict[str, Any]) -> None:
        """Called at the start of optimization for a new problem/run."""

    @abstractmethod
    def step(self, observation: Dict[str, Any]) -> Dict[str, float]:
        """Given an observation, return a dict of PSO parameters to use."""


class StaticController(ParamController):
    def __init__(self, params: Dict[str, float]):
        self.params = dict(params)

    def reset(self, problem_meta: Dict[str, Any]) -> None:
        return None

    def step(self, observation: Dict[str, Any]) -> Dict[str, float]:
        return dict(self.params)


class DummyHeuristicController(ParamController):
    """A tiny heuristic that decays inertia weight and adjusts
    cognitive/social coefficients based on progress."""
    def __init__(self, w_start=0.9, w_end=0.4, decay_iters=100):
        self.w_start = w_start
        self.w_end = w_end
        self.decay_iters = decay_iters

    def reset(self, problem_meta: Dict[str, Any]) -> None:
        self.it = 0

    def step(self, observation: Dict[str, Any]) -> Dict[str, float]:
        self.it = observation.get('iteration', self.it + 1)
        t = min(1.0, float(self.it) / max(1, self.decay_iters))
        w = self.w_start * (1 - t) + self.w_end * t
        # Encourage exploration early, exploitation later
        c1 = 1.8 * (1.0 - 0.5 * t)
        c2 = 1.4 + 0.4 * t
        # Also support DE-style parameters for multi-objective DE optimizers
        F = 0.9 - 0.5 * t  # Large mutation early, refined late
        CR = 0.9 - 0.4 * t # More crossover early for diversity
        return {'w': w, 'c1': c1, 'c2': c2, 'F': F, 'CR': CR}


class AdaptiveDEController(ParamController):
    """Adaptive controller for DE/JADE algorithms with epsilon-constrained support.

    - Early iterations: high F (exploration), high CR (diversity),
      use 'rand/1' strategy to explore broadly.
    - Late iterations: lower F (exploitation), moderate CR (retain
      good building blocks), switch to 'jade' for fast convergence.
    - Uses stagnation detection to temporarily boost F/CR.
    - Provides a decaying epsilon relaxation for epsilon-constrained handling,
      allowing the optimizer to explore mildly infeasible regions early and
      enforcing feasibility as the budget is consumed.
    - If constraint violations are still large late in the run, epsilon is kept
      nonzero to avoid discarding useful search directions prematurely.
    """
    def __init__(self, f_start=0.9, f_end=0.3, cr_start=0.9, cr_end=0.4,
                 strategy_change_iter=0.3, stagnation_boost=0.2,
                 epsilon_start=1e-2, epsilon_end=1e-8):
        self.f_start = f_start
        self.f_end = f_end
        self.cr_start = cr_start
        self.cr_end = cr_end
        self.strategy_change_iter = strategy_change_iter
        self.stagnation_boost = stagnation_boost
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end

    def reset(self, problem_meta: Dict[str, Any]) -> None:
        self.it = 0
        self.last_gbest = None
        self.last_gbest_cv = None
        self.stagnation_count = 0

    def step(self, observation: Dict[str, Any]) -> Dict[str, float]:
        self.it = observation.get('iteration', self.it + 1)
        max_iter = max(1, int(observation.get('max_evals', 10000) / 100))
        t = min(1.0, float(self.it) / max_iter)

        # Smooth progress-based decay from exploration to exploitation
        F = self.f_start * (1 - t) + self.f_end * t
        CR = self.cr_start * (1 - t) + self.cr_end * t

        # Detect stagnation: no improvement in best fitness for several iterations
        current_gbest = observation.get('gbest_f', None)
        if current_gbest is not None and self.last_gbest is not None:
            if current_gbest < self.last_gbest - 1e-12:
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1
        self.last_gbest = current_gbest

        # If stagnated for a while, add extra diversity
        if self.stagnation_count > 5:
            F = min(1.0, F + self.stagnation_boost)
            CR = min(1.0, CR + self.stagnation_boost)
            self.stagnation_count = 0

        # Switch strategy: rand/1 early for exploration, jade later for convergence
        if t < self.strategy_change_iter:
            strategy = 'rand/1'
        else:
            strategy = 'jade'

        # Decaying epsilon relaxation for epsilon-constrained DE.
        # Use log-scale interpolation for smooth transition.
        log_eps_start = float(np_log(self.epsilon_start))
        log_eps_end = float(np_log(self.epsilon_end))
        log_eps = log_eps_start + t * (log_eps_end - log_eps_start)
        epsilon = float(np_exp(log_eps))

        # If the current best is still infeasible late in the run, relax epsilon
        # slightly so the algorithm is not forced into a tiny feasible region.
        current_gbest_cv = observation.get('gbest_cv', None)
        if current_gbest_cv is not None and t > 0.5 and current_gbest_cv > epsilon:
            epsilon = max(epsilon, min(current_gbest_cv * 0.5, self.epsilon_start))

        return {'F': float(F), 'CR': float(CR), 'strategy': strategy,
                'epsilon': float(epsilon)}
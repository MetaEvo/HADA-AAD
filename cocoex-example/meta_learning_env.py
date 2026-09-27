"""Environment wrapper to train DQN to control DE parameters.

Implements a tiny OpenAI Gym-like interface: reset(problem, max_evals, init_x),
step(action) -> (obs, reward, done, info).

Reward convention: 1.0 if a strictly better solution is found during the
step, 0.0 otherwise.
"""
from typing import Any, Dict, Tuple, Optional

import os
import importlib.util

# Import global configuration
import sys
import os
try:
    sys.path.insert(0, '/hyperagents')
    from config import set_global_seed, SEED
    set_global_seed()
except Exception:
    SEED = 42
    def set_global_seed(*args, **kwargs):
        pass

this_dir = os.path.dirname(__file__)
def _load_local(name, filename):
    path = os.path.join(this_dir, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# load local ec_algorithm module
_ec_mod = _load_local('cocoex_example.ec_algorithm', 'ec_algorithm.py')
DEOptimizer = _ec_mod.DEOptimizer


class DQNEnv:
    def __init__(self, step_budget: int = 20, pop_size: int = None):
        self.problem = None
        self.max_evals = 0
        self.step_budget = int(step_budget)
        self.pop_size = pop_size
        self.total_evals = 0
        self.current_best_x = None
        self.current_best_f = float('inf')
        self.done = False

    def reset(self, problem, max_evals: int, init_x: Optional[list] = None) -> Dict[str, Any]:
        """Attach a cocoex `problem` and initialize internal state.

        Returns initial observation.
        """
        self.problem = problem
        self.max_evals = int(max_evals)
        self.total_evals = 0
        self.done = False
        if init_x is None:
            try:
                init_x = problem.initial_solution
            except Exception:
                init_x = [0] * problem.dimension
        self.current_best_x = list(init_x)
        f = float(problem(self.current_best_x))
        self.current_best_f = f
        self.init_best_f = f
        self.total_evals = problem.evaluations
        return self._get_obs()

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Apply action (dict or array-like) as DE params, run DE for a
        short `step_budget` evaluations starting from current best, and return
        observation, reward, done, info.
        """
        if self.done:
            return self._get_obs(), 0.0, True, {}

        # normalize action into params dict
        params = {}
        if isinstance(action, dict):
            params.update(action)
        else:
            # assume array-like [F, CR]
            params['F'] = float(action[0])
            params['CR'] = float(action[1])

        # run a short DE search
        de = DEOptimizer(self.problem.dimension, getattr(self.problem, 'lower_bounds', None), getattr(self.problem, 'upper_bounds', None), self.step_budget, params=params, swarm_size=self.pop_size)
        xopt, stats = de.optimize(self.problem, x0=self.current_best_x)

        new_best = stats.get('gbest_f', float('inf'))
        new_x = stats.get('gbest', None)

        # reward: 1 if improved, 0 otherwise
        reward = 0.0
        if new_best < self.current_best_f:
            reward = 1.0
            self.current_best_f = new_best
            if new_x is not None:
                self.current_best_x = list(new_x)

        # update eval counters
        self.total_evals = stats.get('evals', self.total_evals)
        done = (self.total_evals >= self.max_evals) or getattr(self.problem, 'final_target_hit', False)
        self.done = done

        obs = self._get_obs()
        info = {'de_stats': stats}
        return obs, float(reward), done, info

    def _get_obs(self) -> Dict[str, Any]:
        return {'iteration': 0, 'evals': int(self.total_evals), 'gbest_f': float(self.current_best_f), 'initial_gbest_f': float(self.init_best_f), 'max_evals': int(self.max_evals)}

    def render(self):
        print('evals', self.total_evals, 'best_f', self.current_best_f)
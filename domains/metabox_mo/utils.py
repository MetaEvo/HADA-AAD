"""
MetaBox MO (Multi-Objective) Domain Utilities
多目标黑箱优化 domain 的工具函数
"""
import sys
sys.path.insert(0, '/hada')
from config import MODEL_NAME as MODEL

QUESTION_ID = "problem_id"

import os
import json
import importlib.util
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional


def format_input_dict(row, prev_gen_info=None):
    """Convert a csv row into the input dict for TaskAgent.forward."""
    prob = {}
    prob['domain'] = 'metabox_mo'
    prob['problem_id'] = str(row.get('problem_id', ''))
    prob['suite'] = str(row.get('suite', 'wfg'))
    prob['instance'] = int(row.get('instance', 0))
    prob['dimension'] = int(row.get('dimension', 28))
    prob['n_objectives'] = int(row.get('n_objectives', 5))
    prob['task_instruction'] = get_instruction()
    
    if prev_gen_info is not None:
        prob['previous_generation_info'] = {
            'generation': prev_gen_info.get('generation', 'unknown'),
            'total_problems': prev_gen_info.get('total_problems', 0),
            'history_files': {
                'generate_log': '/hada/outputs/{run_id}/gen_{gen}/generate.log',
                'report': '/hada/outputs/{run_id}/gen_{gen}/report.json',
                'predictions': '/hada/outputs/{run_id}/gen_{gen}/metabox_mo_eval/agent_evals/predictions.csv',
                'patch': '/hada/outputs/{run_id}/gen_{gen}/metabox_mo_eval/agent_evals/all_patch.diff',
            },
            'note': 'Use editor tool to view these files for previous generation analysis'
        }
    
    return prob


def get_instruction() -> str:
    """返回 MetaBox MO WFG domain 的 task instruction"""
    instruction = """You are an expert in multi-objective optimization (MOO).

## Codebase Structure (/hada/metabbo/):

The codebase has two main layers:

### 1. Evolutionary Algorithm Layer (ec_algorithm.py)
- **ec_algorithm.py**: The main evolutionary algorithm implementation. Contains the optimizer class that handles population initialization, iteration loop, solution evaluation, and result tracking. For multi-objective optimization, this is where you implement Pareto dominance, non-dominated sorting, and diversity maintenance.
- **param_controller.py**: Parameter controller interface. Defines the abstract interface for dynamic parameter adjustment.

### 2. Meta-Learning Layer (meta_learning.py, meta_learning_env.py, train_meta_learning.py)
- **meta_learning.py**: Meta-learning controller that dynamically adjusts algorithm parameters during optimization. Uses a neural network to map observations to parameter values.
- **meta_learning_env.py**: Environment wrapper for training the meta-learning controller. Provides a gym-like interface.
- **train_meta_learning.py**: Training scripts for the meta-learning controller.

## CRITICAL: Try Different Evolutionary Algorithms!

Previous generations have mostly made small parameter tweaks. To achieve significant improvements, you should try fundamentally different approaches.

**Consider replacing or substantially modifying the algorithm:**
- **NSGA-II**: Non-dominated Sorting Genetic Algorithm II (classic, robust)
- **NSGA-III**: Reference-point based NSGA for many objectives (recommended for 5 objectives)
- **MOEA/D**: Decomposition-based approach, excellent for many objectives
- **SPEA2**: Strength Pareto Evolutionary Algorithm 2, archive-based
- **SMPSO**: Speed-constrained Multi-objective PSO
- **IBEA**: Indicator-Based Evolutionary Algorithm (using HV indicator)
- Or any other multi-objective evolutionary algorithm

**Diversity maintenance strategies to consider:**
- Crowding distance (NSGA-II style)
- Reference point-based diversity (NSGA-III style)
- Decomposition-based (MOEA/D style with weight vectors)
- Clustering-based diversity

**Meta-Learning Layer:**
- Try different neural network architectures
- Change the observation space to include Pareto front metrics (HV, spread, etc.)
- Change the action space to output multi-objective specific parameters
- Try different training strategies for multi-objective settings

## Problem API (pymoo WFG problems, 28-dim, 5 objectives):

The problems use pymoo's built-in WFG classes (WFG1-WFG9).

```python
# IMPORTANT: Input bounds are NOT [0, 1]!
# xl = [0, 0, ..., 0] (all zeros)
# xu = [2, 4, 6, 8, ..., 2*(i+1)] for each dimension i (pymoo's actual bounds)
# Population MUST be initialized within [xl, xu]

obj_vector = problem.func(x)  # x: 2D array [NP, dim], returns [NP, 5]
problem.dim        # 28
problem.n_obj      # 5 objectives
problem.lb         # shape (dim,) - all zeros (from wfg.xl)
problem.ub         # shape (dim,) - [2, 4, 6, ..., 56] (from wfg.xu)
problem.wfg        # The underlying pymoo WFG object
problem.wfg.__class__.__name__  # e.g., 'WFG1', 'WFG2', etc.
```

## Dataset:
- **Suite**: `pymoo_wfg`
- **Train problems**: WFG1, WFG4, WFG7 (instance 0, 3, 6)
- **Test problems**: WFG2, WFG3, WFG5, WFG6, WFG8, WFG9 (instance 1, 2, 4, 5, 7, 8)
- Dataset file: `/hada/domains/metabox_mo/dataset.csv`

## CRITICAL Rules:
1. **Input Format**: `x` MUST be 2D numpy array `[NP, dim]`
   - Single solution: `problem.func(x.reshape(1, -1))[0]`
   - Population: `problem.func(population)` returns `[NP, 5]`
2. **Bounds**: `lb = problem.lb` (all zeros), `ub = problem.ub` ([2, 4, 6, ..., 56])
   - Initialize population: `pop = rng.rand(NP, dim) * (ub - lb) + lb`
3. **Evaluation**: Use `problem.func(x)`, NOT `problem(x)` or `problem.eval(x)`

## ⚠️ CRITICAL: ec_algorithm.py Modification for MOO Problems
The problem's `func()` method REQUIRES 2D input array `[NP, dim]`. The default `evaluate_problem(problem, x)` in `ec_algorithm.py` passes a 1D array, which will cause an IndexError.

You MUST modify `evaluate_problem` in `ec_algorithm.py` to handle this:
```python
def evaluate_problem(problem, x):
    if not isinstance(x, np.ndarray):
        x = np.asarray(x, dtype=float)
    if hasattr(problem, 'func'):
        # MOO problems require 2D input
        if x.ndim == 1:
            x = x.reshape(1, -1)
        result = problem.func(x)
        return float(np.mean(np.array(result).flatten()))
    else:
        return float(problem(x))
```

## 5 Objectives (WFG problems)

## Algorithm Guidelines:
- Convert to scalar if needed: `scalar = obj @ weights`
- Track Pareto front throughout optimization
- **CRITICAL**: You MUST track and return ALL evaluated solutions (archive), not just the final population
- The archive should include: initial population + all trial/offspring solutions evaluated during optimization
- Return **archive_positions**: a **list of 1D arrays** (all solution positions ever evaluated)

## CRITICAL Output Requirements:
- Algorithm must return `archive_positions`: a **list of 1D arrays** containing ALL evaluated solutions (initial population + all offspring/trial vectors)
- DO NOT change `evaluate_problem(problem, x)` signature - it MUST return a **scalar float**
- External scoring will: `problem.func(x.reshape(1, -1))[0]` for each solution in `archive_positions`
- See `domains/metabox_mo/dqn_de_util.py` for how evaluation works: it extracts `archive_positions` from stats and evaluates each solution to compute the Pareto front for HV calculation

## MANDATORY CODE MODIFICATION:
- You **MUST** use `str_replace` to modify at least one file in `/hada/metabbo/`
- Even if you think the code is already good, you MUST make at least one small improvement

## Scoring:
- Score = Normalized Hypervolume (using pymoo WFG reference points)
- WFG objectives are normalized: f_norm = (f_raw - z*) / (z^nadir - z*)
  where z* = [0, 0, 0, 0, 0] and z^nadir = [2, 4, 6, 8, 10] (pymoo WFG scaled values for M=5)
- HV is computed in normalized space with reference point [1.1, 1.1, 1.1, 1.1, 1.1] (5 dimensions)
- Final Score = HV_raw / (1.1^5), range [0, 1]
- Higher Score = better Pareto front coverage and convergence

EVALUATION COUNTING:
- Evaluations are tracked via the `evals` variable in `ec_algorithm.py`
- Each time you call `evaluate_problem(problem, x)` or `problem.func(x)`, you MUST increment `evals += 1`
- The optimization loop runs while `evals < self.max_evals`
- If you change the algorithm, ensure `evals` is incremented correctly for every function evaluation

⚠️ IMPORTANT: Before implementing changes, READ the domain code at `/hada/domains/metabox_mo/dqn_de_util.py` to understand the exact scoring logic, Pareto front evaluation, and return format requirements.
"""
    return instruction


def get_default_data_path() -> str:
    """返回默认数据路径"""
    return os.path.join(os.path.dirname(__file__), 'dataset.csv')


def load_problems(data_path: Optional[str] = None) -> pd.DataFrame:
    """加载 MetaBox MO 问题集"""
    if data_path is None:
        data_path = get_default_data_path()
    
    if not os.path.exists(data_path):
        create_default_dataset(data_path)
    
    return pd.read_csv(data_path)


def create_default_dataset(data_path: str):
    """创建 WFG 数据集（28维，5目标，pymoo WFG1-9）
    
    - train: WFG1, WFG4, WFG7 (3 problems)
    - test: WFG2, WFG3, WFG5, WFG6, WFG8, WFG9 (6 problems)
    """
    problems = []
    
    # Train problems: WFG1, WFG4, WFG7
    train_instances = [0, 3, 6]  # WFG1, WFG4, WFG7
    for i in train_instances:
        problems.append({
            'problem_id': f'wfg{i+1}_pymoo_d28_o5',
            'suite': 'pymoo_wfg',
            'instance': i,
            'dimension': 28,
            'n_objectives': 5,
            'split': 'train'
        })
    
    # Test problems: WFG2, WFG3, WFG5, WFG6, WFG8, WFG9
    test_instances = [1, 2, 4, 5, 7, 8]  # WFG2, WFG3, WFG5, WFG6, WFG8, WFG9
    for i in test_instances:
        problems.append({
            'problem_id': f'wfg{i+1}_pymoo_d28_o5',
            'suite': 'pymoo_wfg',
            'instance': i,
            'dimension': 28,
            'n_objectives': 5,
            'split': 'test'
        })
    
    df = pd.DataFrame(problems)
    df.to_csv(data_path, index=False)
    print(f"Created WFG dataset with {len(problems)} problems at {data_path}")
    print(f"  Train: 3 problems (WFG1, WFG4, WFG7)")
    print(f"  Test: 6 problems (WFG2, WFG3, WFG5, WFG6, WFG8, WFG9)")


def compute_hypervolume(pareto_front: np.ndarray, reference_point: np.ndarray) -> float:
    """计算 Hypervolume 指标"""
    try:
        from pymoo.indicators.hv import Hypervolume
        hv = Hypervolume(ref_point=reference_point)
        return hv.do(pareto_front)
    except ImportError:
        if pareto_front.shape[1] == 2:
            return _compute_hv_2d(pareto_front, reference_point)
        else:
            return _approximate_hv(pareto_front, reference_point)


def _compute_hv_2d(pareto_front: np.ndarray, reference_point: np.ndarray) -> float:
    """2D Hypervolume 计算"""
    sorted_pf = pareto_front[np.argsort(pareto_front[:, 0])]
    
    hv = 0.0
    prev_f2 = reference_point[1]
    for sol in sorted_pf:
        if sol[0] < reference_point[0] and sol[1] < reference_point[1]:
            hv += (reference_point[0] - sol[0]) * (prev_f2 - sol[1])
            prev_f2 = min(prev_f2, sol[1])
    
    return hv


def _approximate_hv(pareto_front: np.ndarray, reference_point: np.ndarray) -> float:
    """近似 Hypervolume 计算"""
    valid = np.all(pareto_front < reference_point, axis=1)
    if not np.any(valid):
        return 0.0
    valid_pf = pareto_front[valid]
    return float(np.prod(reference_point - np.min(valid_pf, axis=0)))


def compute_igd(pareto_front: np.ndarray, true_pareto_front: np.ndarray) -> float:
    """计算 IGD 指标"""
    igd = 0.0
    for true_sol in true_pareto_front:
        distances = np.linalg.norm(pareto_front - true_sol, axis=1)
        igd += np.min(distances)
    return igd / len(true_pareto_front)


def compute_gd(pareto_front: np.ndarray, true_pareto_front: np.ndarray) -> float:
    """计算 GD 指标"""
    gd = 0.0
    for sol in pareto_front:
        distances = np.linalg.norm(true_pareto_front - sol, axis=1)
        gd += np.min(distances)
    return gd / len(pareto_front)


def compute_spacing(pareto_front: np.ndarray) -> float:
    """计算 Spacing 指标"""
    n = len(pareto_front)
    if n <= 1:
        return 0.0
    
    distances = []
    for i in range(n):
        min_dist = float('inf')
        for j in range(n):
            if i != j:
                dist = np.linalg.norm(pareto_front[i] - pareto_front[j])
                min_dist = min(min_dist, dist)
        distances.append(min_dist)
    
    mean_dist = np.mean(distances)
    return float(np.sqrt(np.mean([(d - mean_dist) ** 2 for d in distances])))


def get_true_pareto_front(problem) -> Optional[np.ndarray]:
    """获取真实 Pareto 前沿"""
    try:
        if hasattr(problem, 'get_ref_set'):
            return problem.get_ref_set()
    except:
        pass
    return None


def find_non_dominated_solutions(solutions: np.ndarray) -> np.ndarray:
    """找出非支配解"""
    n = len(solutions)
    is_dominated = np.zeros(n, dtype=bool)
    
    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(i + 1, n):
            if is_dominated[j]:
                continue
            
            i_dominates_j = np.all(solutions[i] <= solutions[j]) and np.any(solutions[i] < solutions[j])
            j_dominates_i = np.all(solutions[j] <= solutions[i]) and np.any(solutions[j] < solutions[i])
            
            if i_dominates_j:
                is_dominated[j] = True
            elif j_dominates_i:
                is_dominated[i] = True
                break
    
    return solutions[~is_dominated]


def dominates(obj1: np.ndarray, obj2: np.ndarray) -> bool:
    """检查 obj1 是否支配 obj2"""
    return np.all(obj1 <= obj2) and np.any(obj1 < obj2)


def non_dominated_sort(objectives: np.ndarray) -> Tuple[List[List[int]], List[int]]:
    """
    非支配排序 (NSGA-II 风格)
    
    Returns:
        fronts: 每个 front 的索引列表
        ranks: 每个解的 rank
    """
    n = len(objectives)
    fronts = []
    ranks = [-1] * n
    
    dominated_count = [0] * n
    dominates_set = [[] for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            if dominates(objectives[i], objectives[j]):
                dominates_set[i].append(j)
                dominated_count[j] += 1
            elif dominates(objectives[j], objectives[i]):
                dominates_set[j].append(i)
                dominated_count[i] += 1
    
    # Front 0: 非支配解
    front_0 = [i for i in range(n) if dominated_count[i] == 0]
    fronts.append(front_0)
    for i in front_0:
        ranks[i] = 0
    
    # 后续 fronts
    k = 0
    while len(fronts[k]) > 0:
        next_front = []
        for i in fronts[k]:
            for j in dominates_set[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)
                    ranks[j] = k + 1
        k += 1
        if len(next_front) > 0:
            fronts.append(next_front)
        else:
            break
    
    return fronts, ranks


def crowding_distance(objectives: np.ndarray, front: List[int]) -> List[float]:
    """计算 crowding distance"""
    n_obj = objectives.shape[1]
    dist = [0.0] * len(front)
    
    for m in range(n_obj):
        sorted_front = sorted(front, key=lambda i: objectives[i, m])
        dist[front.index(sorted_front[0])] = float('inf')
        dist[front.index(sorted_front[-1])] = float('inf')
        
        obj_range = objectives[sorted_front[-1], m] - objectives[sorted_front[0], m]
        if obj_range == 0:
            continue
        
        for i in range(1, len(sorted_front) - 1):
            dist[front.index(sorted_front[i])] += (
                objectives[sorted_front[i + 1], m] - objectives[sorted_front[i - 1], m]
            ) / obj_range
    
    return dist


def normalize_objectives(objectives: np.ndarray) -> np.ndarray:
    """归一化目标值到 [0, 1]"""
    normalized = np.zeros_like(objectives)
    for m in range(objectives.shape[1]):
        obj_min = np.min(objectives[:, m])
        obj_max = np.max(objectives[:, m])
        if obj_max - obj_min > 1e-10:
            normalized[:, m] = (objectives[:, m] - obj_min) / (obj_max - obj_min)
        else:
            normalized[:, m] = 0.0
    return normalized


def get_problem_info(problem) -> Dict:
    """获取问题信息"""
    info = {
        'dim': problem.dim,
        'n_obj': problem.n_obj,
        'lb': problem.lb.tolist() if hasattr(problem.lb, 'tolist') else problem.lb,
        'ub': problem.ub.tolist() if hasattr(problem.ub, 'tolist') else problem.ub,
    }
    return info


def get_score_from_result(result_str: str) -> float:
    """从结果字符串解析分数"""
    try:
        parts = result_str.split('|')
        return float(parts[0])
    except:
        return 0.0


def get_extra_info_from_result(result_str: str) -> Dict:
    """从结果字符串解析额外信息"""
    try:
        parts = result_str.split('|')
        return {
            'hypervolume': float(parts[1]) if len(parts) > 1 else 0.0,
            'igd': float(parts[2]) if len(parts) > 2 else 0.0,
            'gd': float(parts[3]) if len(parts) > 3 else 0.0,
            'spacing': float(parts[4]) if len(parts) > 4 else 0.0,
            'n_solutions': int(parts[5]) if len(parts) > 5 else 0,
        }
    except:
        return {}
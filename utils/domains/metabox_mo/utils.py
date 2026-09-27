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
    prob['suite'] = str(row.get('suite', 'uav'))
    prob['instance'] = int(row.get('instance', 1))
    prob['dimension'] = int(row.get('dimension', 30))
    prob['n_objectives'] = int(row.get('n_objectives', 2))
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
    """返回 MetaBox MO UAV domain 的 task instruction"""
    instruction = """You are an expert in multi-objective UAV path planning using PSO.

## Available Files in /hada/metabbo/:

- `pso.py` - The main PSO algorithm implementation. Contains particle swarm optimization logic including velocity/position updates, fitness evaluation, and the optimization loop. It integrates with a parameter controller that can dynamically adjust PSO parameters (w, c1, c2) during execution.
- `dqn_controller.py` - A Deep Q-Network based controller that outputs PSO parameters at each iteration. It takes the current optimization state as observation and learns to output parameter values through reinforcement learning training.
- `param_controller.py` - An interface/abstraction layer for parameter control. The DQN controller implements this interface.
- `train_dqn_simple.py` / `train_dqn.py` - Scripts for training the DQN controller through interaction with the optimization process.

## Domain Goal:

The MetaBox MO domain consists of multi-objective UAV path planning problems with 5 objectives (path length, threat exposure, altitude, smoothness, terrain). The goal is to find a diverse set of high-quality solutions that approximate the Pareto front. Performance is measured by Normalized Hypervolume (HV) and IGD metrics.

## UAV Problem API:
```python
obj_vector = problem.func(x)  # x: 2D array [NP, dim], returns [NP, 5]
problem.dim        # 30 (3 * dv, dv=10 waypoints)
problem.n_obj      # 5 objectives
problem.lb         # shape (dim,)
problem.ub         # shape (dim,)
```

## CRITICAL Rules:
1. **Input Format**: `x` MUST be 2D numpy array `[NP, dim]`
   - Single solution: `problem.func(x.reshape(1, -1))[0]`
   - Population: `problem.func(population)` returns `[NP, 5]`
2. **No `initial_solution`**: UAV problem doesn't have this attribute. Use `(lb + ub) / 2.0` as center.
3. **Evaluation**: Use `problem.func(x)`, NOT `problem(x)` or `problem.eval(x)`

## CRITICAL Output Requirements:
- PSO must return `final_pbest`: a **list of 1D arrays** (particle positions)
- DO NOT change `evaluate_problem(problem, x)` signature - it MUST return a **scalar float**
- External scoring will: `problem.func(x.reshape(1, -1))[0]` for each solution in `final_pbest`

## MANDATORY CODE MODIFICATION:
- You **MUST** use `str_replace` to modify at least one file in `/hada/metabbo/`
- Even if you think the code is already good, you MUST make at least one small improvement

## Scoring:
- Score = Normalized Hypervolume (computed on final Pareto front)
- HV is normalized by reference point volume: Score = HV_raw / prod(reference_point)
- Reference point is loaded from reference.csv (suite-specific)
- Higher Score = better Pareto front coverage and convergence

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
    """创建 UAV difficult 模式数据集（56 个问题）
    
    difficult 模式划分（seed=3849）：
    - train: 奇数 ID (1, 3, 5, ..., 55) - 28 个
    - test: 偶数 ID (0, 2, 4, ..., 54) - 28 个
    """
    problems = []
    
    for problem_id in range(56):
        # difficult 模式：奇数 ID 为 train，偶数 ID 为 test
        split = 'train' if problem_id % 2 == 1 else 'test'
        
        problems.append({
            'problem_id': f'uav_i{problem_id + 1}_d30_o5',  # instance 从 1 开始
            'suite': 'uav',
            'function': problem_id,
            'instance': problem_id + 1,  # instance 从 1 开始
            'dimension': 30,  # 3 * dv, dv=10
            'n_objectives': 5,
            'split': split
        })
    
    df = pd.DataFrame(problems)
    df.to_csv(data_path, index=False)
    print(f"Created UAV difficult dataset with {len(problems)} problems at {data_path}")
    print(f"  Train: {len(df[df['split'] == 'train'])} problems (odd IDs: 1, 3, 5, ..., 55)")
    print(f"  Test: {len(df[df['split'] == 'test'])} problems (even IDs: 0, 2, 4, ..., 54)")


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
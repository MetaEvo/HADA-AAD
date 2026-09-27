"""
MetaBox MT (Multi-Task) Domain Utilities
多任务黑箱优化 domain 的工具函数
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
    prob['domain'] = 'metabox_mt'
    prob['problem_id'] = str(row.get('problem_id', ''))
    prob['suite'] = str(row.get('suite', 'cec2017mto'))
    prob['instance'] = int(row.get('instance', 1))
    prob['dimension'] = int(row.get('dimension', 50))
    prob['n_tasks'] = int(row.get('n_tasks', 50))
    prob['task_instruction'] = get_instruction()
    
    # 添加前代信息（如果有的话）
    if prev_gen_info is not None:
        prob['previous_generation_info'] = {
            'generation': prev_gen_info.get('generation', 'unknown'),
            'total_problems': prev_gen_info.get('total_problems', 0),
            'history_files': {
                'generate_log': '/hada/outputs/{run_id}/gen_{gen}/generate.log',
                'report': '/hada/outputs/{run_id}/gen_{gen}/report.json',
                'predictions': '/hada/outputs/{run_id}/gen_{gen}/metabox_mt_eval/agent_evals/predictions.csv',
                'patch': '/hada/outputs/{run_id}/gen_{gen}/metabox_mt_eval/agent_evals/all_patch.diff',
            },
            'note': 'Use editor tool to view these files for previous generation analysis'
        }
    
    return prob


def get_instruction() -> str:
    """返回 MetaBox MT domain 的 task instruction"""
    instruction = """You are an expert in multi-task black-box optimization using PSO.

## Available Files in /hada/metabbo/:

- `pso.py` - The main PSO algorithm implementation. Contains particle swarm optimization logic including velocity/position updates, fitness evaluation, and the optimization loop. It integrates with a parameter controller that can dynamically adjust PSO parameters (w, c1, c2) during execution.
- `dqn_controller.py` - A Deep Q-Network based controller that outputs PSO parameters at each iteration. It takes the current optimization state as observation and learns to output parameter values through reinforcement learning training.
- `param_controller.py` - An interface/abstraction layer for parameter control. The DQN controller implements this interface.
- `train_dqn_simple.py` / `train_dqn.py` - Scripts for training the DQN controller through interaction with the optimization process.

## Domain Goal:

The MetaBox MT domain consists of multi-task optimization problems where a single algorithm must optimize multiple related tasks simultaneously. The goal is to leverage knowledge transfer between tasks to improve overall performance across all tasks. Performance is measured by Relative Improvement score on each task plus Knowledge Transfer Effectiveness (KTE).

## Architecture:

The PSO receives the ENTIRE problem object (containing 50 tasks) at once:
```python
gbest, stats = pso_opt.optimize(problem, train=True)
```

Your PSO code should handle ALL 50 tasks internally and return:
```python
stats = {
    'task_gbest_f': [gbest_f_for_task_0, ..., gbest_f_for_task_49],
    'task_initial_bests': [initial_best_0, ..., initial_best_49],
    ...
}
```

## WCCI2020 Problem API:

```python
# Each problem has 50 tasks (instances)
tasks = problem.tasks  # List of 50 task objects

for task in tasks:
    fitness = task.eval(x)  # x in [0,1], returns scalar
    task.dim      # 50
    task.lb       # Lower bound (scalar)
    task.ub       # Upper bound (scalar)
    task.optimum  # Theoretical optimal value
```

## Key Points:

1. **Evaluation**: Use `task.eval(x)` where `x` is 1D array in [0,1]. `eval()` auto-decodes to [lb, ub].
2. **Bounds**: `task.lb` and `task.ub` are scalars. Convert: `lb = np.full(dim, float(task.lb))`
3. **Optimal value**: ALWAYS use `task.optimum`. Most functions ≈ 0.0, Schwefel ≈ 19176.99.
4. **Return format**: MUST return `stats['task_gbest_f']` and `stats['task_initial_bests']` as lists of 50 values.

## CRITICAL Rules:
- Use absolute imports only in pso.py
- PSO should handle multi-task optimization internally
- PSO searches in [0,1] space
- DQN controller is passed via `controller` parameter
- PRIORITIZE modifying pso.py to support multi-task optimization

## Scoring:
- Score = (initial_best - best) / (initial_best - opt)
  - initial_best: PSO初始化后种群的最优值
  - best: PSO运行结束后的最优值
  - opt: 官方最优值 (task.optimum)
- KTE: Knowledge Transfer Effectiveness
- Final Score = 0.7 * avg(task_score) + 0.3 * KTE

## ⚠️ CRITICAL: Common Errors to AVOID ⚠️

1. **NO `problem.initial_solution`**: This attribute does NOT exist. Initialize particles in `[0,1]` space directly.
2. **NO `problem(x)` or `problem.func(x)`**: The problem object is NOT callable. Use `task.eval(x)` for each task.
3. **NO `evaluate_problem()` helper**: It's for single-task cocoex, not multi-task. Write your own evaluation logic.
4. **Evaluation pattern**: Iterate `for task in problem.tasks: fitness = task.eval(position)` where `position` is in `[0,1]` space.
5. **Return format**: MUST return `stats['task_gbest_f']` and `stats['task_initial_bests']` as lists of 50 values (one per task).

⚠️ IMPORTANT: Before implementing changes, READ the domain code at `/hada/domains/metabox_mt/dqn_de_util.py` to understand the exact scoring logic, KTE calculation, and return format requirements.
"""
    return instruction


def get_default_data_path() -> str:
    """返回默认数据路径"""
    return os.path.join(os.path.dirname(__file__), 'dataset.csv')


def load_problems(data_path: Optional[str] = None) -> pd.DataFrame:
    """加载 MetaBox MT 问题集"""
    if data_path is None:
        data_path = get_default_data_path()
    
    if not os.path.exists(data_path):
        # 创建默认数据集
        create_default_dataset(data_path)
    
    return pd.read_csv(data_path)


def create_default_dataset(data_path: str):
    """创建默认的 MetaBox MT 数据集"""
    # MetaBox MT 问题集: CEC2017-MTO, WCCI2020, Augmented-WCCI2020
    problems = []
    
    # CEC2017-MTO (9 instances, 2 tasks each, 25D-50D)
    for instance in range(1, 10):
        for n_tasks in [2]:
            for dim in [25, 50]:
                problems.append({
                    'problem_id': f'cec2017mto_i{instance}_d{dim}_t{n_tasks}',
                    'suite': 'cec2017mto',
                    'instance': instance,
                    'dimension': dim,
                    'n_tasks': n_tasks,
                    'split': 'train' if instance <= 5 else 'test'
                })
    
    # WCCI2020 (10 instances, 2 tasks each, 50D)
    for instance in range(1, 11):
        problems.append({
            'problem_id': f'wcci2020_i{instance}_d50_t2',
            'suite': 'wcci2020',
            'instance': instance,
            'dimension': 50,
            'n_tasks': 2,
            'split': 'train' if instance <= 5 else 'test'
        })
    
    # Augmented-WCCI2020 (20 instances, 2-3 tasks, 50D)
    for instance in range(1, 21):
        for n_tasks in [2, 3]:
            problems.append({
                'problem_id': f'augwcci2020_i{instance}_d50_t{n_tasks}',
                'suite': 'augwcci2020',
                'instance': instance,
                'dimension': 50,
                'n_tasks': n_tasks,
                'split': 'train' if instance <= 10 else 'test'
            })
    
    df = pd.DataFrame(problems)
    df.to_csv(data_path, index=False)
    print(f"Created default dataset with {len(problems)} problems at {data_path}")


def compute_multi_task_score(scores: np.ndarray, method: str = 'average') -> float:
    """
    计算多任务优化分数
    
    Args:
        scores: 各任务的分数，shape (n_tasks,)
        method: 聚合方法 ('average', 'rank', 'normalized')
    
    Returns:
        多任务综合分数
    """
    if method == 'average':
        return np.mean(scores)
    elif method == 'rank':
        # 使用排名的倒数
        ranks = np.argsort(np.argsort(-scores)) + 1  # 1-based rank
        return np.mean(1.0 / ranks)
    elif method == 'normalized':
        # 归一化到 [0, 1]
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score - min_score < 1e-10:
            return 1.0
        normalized = (scores - min_score) / (max_score - min_score)
        return np.mean(normalized)
    else:
        return np.mean(scores)


def compute_knowledge_transfer_efficiency(
    multi_task_score: float, 
    single_task_scores: List[float]
) -> float:
    """
    计算知识迁移效率 (KTE)
    
    Args:
        multi_task_score: 多任务优化的分数
        single_task_scores: 各任务单独优化的分数列表
    
    Returns:
        知识迁移效率
    """
    avg_single = np.mean(single_task_scores)
    if avg_single < 1e-10:
        return 1.0
    return multi_task_score / avg_single


def compute_task_similarity(task1_features: np.ndarray, task2_features: np.ndarray) -> float:
    """
    计算任务相似度
    
    Args:
        task1_features: 任务1的特征
        task2_features: 任务2的特征
    
    Returns:
        相似度分数 [0, 1]
    """
    # 使用余弦相似度
    dot_product = np.dot(task1_features, task2_features)
    norm1 = np.linalg.norm(task1_features)
    norm2 = np.linalg.norm(task2_features)
    
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    return (similarity + 1) / 2  # 映射到 [0, 1]


def adaptive_parameter_sharing(
    task_performances: List[float], 
    similarity_matrix: np.ndarray
) -> np.ndarray:
    """
    自适应参数共享权重
    
    Args:
        task_performances: 各任务的表现
        similarity_matrix: 任务相似度矩阵
    
    Returns:
        参数共享权重矩阵
    """
    n_tasks = len(task_performances)
    weights = np.zeros((n_tasks, n_tasks))
    
    for i in range(n_tasks):
        for j in range(n_tasks):
            if i == j:
                weights[i, j] = 0.5  # 自身任务权重
            else:
                # 基于相似度和表现的权重
                perf_ratio = task_performances[j] / (task_performances[i] + 1e-10)
                weights[i, j] = 0.5 * similarity_matrix[i, j] * perf_ratio
    
    # 归一化
    row_sums = weights.sum(axis=1, keepdims=True)
    weights = weights / (row_sums + 1e-10)
    
    return weights


def select_representative_tasks(
    task_features: np.ndarray, 
    n_selected: int
) -> List[int]:
    """
    选择代表性任务
    
    Args:
        task_features: 任务特征矩阵
        n_selected: 要选择多少个任务
    
    Returns:
        选中任务的索引列表
    """
    n_tasks = len(task_features)
    if n_selected >= n_tasks:
        return list(range(n_tasks))
    
    # 使用 K-means 风格的贪心选择
    selected = [0]  # 从第一个任务开始
    
    for _ in range(n_selected - 1):
        max_min_dist = -1
        best_idx = -1
        
        for i in range(n_tasks):
            if i in selected:
                continue
            
            # 计算到已选任务的最小距离
            min_dist = min(
                np.linalg.norm(task_features[i] - task_features[j])
                for j in selected
            )
            
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_idx = i
        
        if best_idx >= 0:
            selected.append(best_idx)
    
    return selected


def compute_improvement_ratio(
    current_score: float, 
    baseline_score: float
) -> float:
    """
    计算改进比率
    
    Args:
        current_score: 当前分数
        baseline_score: 基线分数
    
    Returns:
        改进比率
    """
    if baseline_score < 1e-10:
        return 1.0 if current_score > 0 else 0.0
    return (current_score - baseline_score) / baseline_score
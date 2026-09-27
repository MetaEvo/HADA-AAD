"""
MetaBox MT Domain - DQN-PSO Utility
多任务黑箱优化的 DQN-PSO 工具函数
"""
import os
import sys
import json
import math
import importlib.util
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import traceback

# 添加路径并导入全局配置
sys.path.insert(0, '/hada')
from config import set_global_seed, SEED, MODEL_NAME
set_global_seed()

# 导入 utils
from domains.metabox_mt.utils import (
    compute_multi_task_score, compute_knowledge_transfer_efficiency,
    compute_task_similarity, adaptive_parameter_sharing
)


def run_pso_and_score(inputs: Dict, log_fn=None, mode='train') -> str:
    """
    运行 PSO 并计算多任务优化分数

    Args:
        inputs: 包含 problem_id 等信息的字典
        log_fn: 日志函数
        mode: 'train' 或 'test'

    Returns:
        prediction (string) in format "score|avg_task_score|knowledge_transfer|n_tasks"
    """
    problem_id = inputs.get('problem_id', 'unknown')

    if log_fn:
        log_fn(f"[{mode.upper()}] {problem_id} Running multi-task PSO optimization...")

    try:
        # 动态加载 PSO 和 DQN controller (参考 bbob_constrained)
        this_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'metabbo')
        if this_dir not in sys.path:
            sys.path.insert(0, this_dir)

        # load local pso module
        spec = importlib.util.spec_from_file_location('pso_mod', os.path.join(this_dir, 'pso.py'))
        pso_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pso_mod)

        # load controller implementations
        spec2 = importlib.util.spec_from_file_location('dqn_mod', os.path.join(this_dir, 'dqn_controller.py'))
        dqn_mod = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(dqn_mod)

        spec3 = importlib.util.spec_from_file_location('pc_mod', os.path.join(this_dir, 'param_controller.py'))
        pc_mod = importlib.util.module_from_spec(spec3)
        spec3.loader.exec_module(pc_mod)

        # 创建 MetaBox MT 问题
        problem = create_metabox_mt_problem(inputs)

        if problem is None:
            return "0.0|0.0|0.0|0"

        # 运行 DQN-PSO
        result = run_dqn_pso_mt(problem, pso_mod, dqn_mod, log_fn, mode)

        # 返回格式: score|avg_task_score|knowledge_transfer|n_tasks|task_scores|task_gbest_f|task_similarity
        score = result.get('score', 0.0)
        avg_task_score = result.get('avg_task_score', 0.0)
        kte = result.get('knowledge_transfer', 0.0)
        n_tasks = result.get('n_tasks', 0)
        task_scores = result.get('task_scores', [])
        task_gbest_f = result.get('task_gbest_f', [])
        task_similarity = result.get('task_similarity', 0.5)

        # 将列表转换为 JSON 字符串
        task_scores_str = json.dumps(task_scores)
        task_gbest_f_str = json.dumps(task_gbest_f)

        return f"{score}|{avg_task_score}|{kte}|{n_tasks}|{task_scores_str}|{task_gbest_f_str}|{task_similarity}"

    except Exception as e:
        error_msg = f"Error in run_pso_and_score: {str(e)}\n{traceback.format_exc()}"
        if log_fn:
            log_fn(error_msg)
        return "0.0|0.0|0.0|0"


def create_metabox_mt_problem(inputs: Dict) -> Optional[Any]:
    """
    创建 MetaBox MT 问题实例 (基于 MetaBox-v2 官方规范)
    
    对于 cec2017mto 50维问题，使用索引直接访问：
    - instance 1-6: index 0-5 (dataset[0])
    - instance 7-9: index 0-2 (dataset[1])
    """
    try:
        # 1. 解析传入的测试信息
        problem_id = inputs.get('problem_id', '')
        suite = str(inputs.get('suite', 'wcci2020')).lower()
        instance = inputs.get('instance', 1)
        dimension = inputs.get('dimension', 50)
        n_tasks = inputs.get('n_tasks', 50)

        # 2. 加载 WCCI2020 数据集
        from metaevobox.environment.problem.MTO.WCCI2020.wcci2020_dataset import WCCI2020_Dataset
        
        train_set, test_set = WCCI2020_Dataset.get_datasets(
            version='numpy',
            difficulty='difficult'  # train: P7-P10, test: P1-P6
        )
        
        # 3. 根据 instance 获取对应的问题
        # instance 1-10 对应 Problem 0-9
        # train: instance 7-10 -> Problem 0-3 (in train_set)
        # test: instance 1-6 -> Problem 0-5 (in test_set)
        if instance >= 7:
            # 训练集问题
            problem_idx = instance - 7  # 7->0, 8->1, 9->2, 10->3
            if problem_idx < len(train_set.data):
                return train_set.data[problem_idx]
            else:
                print(f"Warning: Instance {instance} out of range for train set")
                return train_set.data[0]
        else:
            # 测试集问题
            problem_idx = instance - 1  # 1->0, 2->1, ..., 6->5
            if problem_idx < len(test_set.data):
                return test_set.data[problem_idx]
            else:
                print(f"Warning: Instance {instance} out of range for test set")
                return test_set.data[0]

    except Exception as e:
        print(f"Error creating MetaBox MT problem: {e}")
        import traceback
        print(traceback.format_exc())
        return None


def run_dqn_pso_mt(problem: Any, pso_mod: Any, dqn_mod: Any,
                   log_fn=None, mode: str = 'train') -> Dict:
    """
    运行 DQN-PSO 进行多任务优化

    架构：将整个问题（包含50个tasks）传递给PSO，让PSO内部处理多任务并行优化
    PSO必须返回 stats['task_gbest_f'] 和 stats['task_initial_bests']
    """
    try:
        mode_str = "TRAIN" if mode == 'train' else "TEST"

        tasks = getattr(problem, 'tasks', [])
        n_tasks = len(tasks)
        
        if n_tasks == 0:
            return {
                'score': 0.0, 'avg_task_score': 0.0, 'knowledge_transfer': 0.0,
                'n_tasks': 0, 'error': 'No tasks found'
            }
        
        first_task = tasks[0]
        dim = getattr(first_task, 'dim', None) or getattr(first_task, 'n_var', 50)
        max_evals = 100000  # 所有任务共享的总评估预算

        lower_bounds = getattr(first_task, 'lb', None)
        upper_bounds = getattr(first_task, 'ub', None)

        if lower_bounds is None:
            lower_bounds = np.zeros(dim)
        elif not hasattr(lower_bounds, '__len__'):
            lower_bounds = np.full(dim, lower_bounds)
        
        if upper_bounds is None:
            upper_bounds = np.ones(dim)
        elif not hasattr(upper_bounds, '__len__'):
            upper_bounds = np.full(dim, upper_bounds)

        model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'metabbo', 'dqn_model.pt')
        controller = dqn_mod.DiscreteDQNController(model_path=model_path)
        
        pso_opt = pso_mod.PSOOptimizer(
            dim, lower_bounds, upper_bounds, max_evals,
            params={'w': 0.729, 'c1': 1.49445, 'c2': 1.49445},
            controller=controller, swarm_size=150
        )

        # 运行 PSO，传递整个 problem 对象
        gbest, stats = pso_opt.optimize(problem, train=(mode == 'train'))
        
        # 训练模式下保存模型
        if mode == 'train' and hasattr(controller, 'net'):
            import torch
            torch.save(controller.net.state_dict(), model_path)
            if log_fn:
                log_fn(f"[TRAIN] Model saved to {model_path}")

        # 获取每个任务的结果
        task_gbest_f = stats.get('task_gbest_f', [])
        task_initial_bests = stats.get('task_initial_bests', [])

        if not task_gbest_f or len(task_gbest_f) != n_tasks:
            if log_fn:
                log_fn(f"[{mode_str}] PSO did not return per-task results, score=0")
            return {
                'score': 0.0, 'avg_task_score': 0.0, 'knowledge_transfer': 0.0,
                'n_tasks': n_tasks, 'error': 'PSO did not return per-task results'
            }

        # 计算每个任务的 score
        task_scores = []
        for task_id in range(n_tasks):
            gbest_f = task_gbest_f[task_id]
            initial_best = task_initial_bests[task_id]
            task = tasks[task_id]
            
            opt_val = 0.0
            if hasattr(task, 'optimum'):
                opt_val = float(task.optimum)
            elif hasattr(task, 'func') and hasattr(task, 'opt'):
                try:
                    opt_pos = task.opt
                    if hasattr(opt_pos, 'shape') and len(opt_pos.shape) > 1:
                        opt_pos = opt_pos[0]
                    f_val = task.func(opt_pos)
                    opt_val = float(f_val[0]) if hasattr(f_val, '__len__') else float(f_val)
                except:
                    opt_val = 0.0

            if abs(initial_best - opt_val) < 1e-12:
                task_score = 1.0 if gbest_f <= opt_val + 1e-12 else 0.0
            else:
                task_score = (initial_best - gbest_f) / (initial_best - opt_val)
                task_score = max(0.0, min(1.0, task_score))
            
            task_scores.append(task_score)

            if log_fn:
                log_fn(f"[{mode_str}] Task {task_id}: initial={initial_best:.6f}, gbest_f={gbest_f:.6f}, opt={opt_val:.6f}, score={task_score:.4f}")

        avg_score = np.mean(task_scores) if task_scores else 0.0

        # 计算 KTE
        kte = 0.0
        valid_scores = [s for s in task_scores if s != 0.0]
        if len(valid_scores) > 1:
            cv = np.std(valid_scores) / (np.mean(valid_scores) + 1e-10)
            kte = (1.0 / (1.0 + cv)) * avg_score
        elif len(valid_scores) == 1:
            kte = 0.0

        # 计算任务间相似度
        avg_similarity = 1.0
        if n_tasks > 1:
            sims = []
            for i in range(n_tasks):
                for j in range(i + 1, n_tasks):
                    if task_gbest_f[i] != float('inf') and task_gbest_f[j] != float('inf'):
                        sims.append(1.0 / (1.0 + abs(task_gbest_f[i] - task_gbest_f[j])))
            avg_similarity = np.mean(sims) if sims else 0.5

        final_score = 0.7 * avg_score + 0.3 * kte
        
        if log_fn:
            log_fn(f"[{mode_str}] Final: avg_score={avg_score:.4f}, KTE={kte:.4f}, score={final_score:.4f}")

        return {
            'score': final_score, 'avg_task_score': avg_score, 'knowledge_transfer': kte,
            'n_tasks': n_tasks, 'task_scores': task_scores, 'task_gbest_f': task_gbest_f,
            'task_initial_bests': task_initial_bests, 'task_similarity': avg_similarity
        }

    except Exception as e:
        if log_fn:
            log_fn(f"[{mode.upper()}] DQN-PSO-MT failed: {e}")
            log_fn(traceback.format_exc())
        return {
            'score': 0.0, 'avg_task_score': 0.0, 'knowledge_transfer': 0.0,
            'n_tasks': 0, 'error': str(e)
        }


def score_predictions(predictions: pd.DataFrame, mode: str = 'train') -> Dict:
    """
    计算多组预测的分数

    Args:
        predictions: DataFrame 包含预测结果
        mode: 'train' 或 'test'

    Returns:
        分数统计字典
    """
    scores = []
    task_scores_list = []
    kte_list = []
    n_tasks_list = []

    for idx, row in predictions.iterrows():
        try:
            pred_str = row.get('prediction', '0.0|0.0|0.0|0')
            parts = pred_str.split('|')
            score = float(parts[0])
            avg_task_score = float(parts[1]) if len(parts) > 1 else 0.0
            kte = float(parts[2]) if len(parts) > 2 else 0.0
            n_tasks = int(parts[3]) if len(parts) > 3 else 0

            scores.append(score)
            task_scores_list.append(avg_task_score)
            kte_list.append(kte)
            n_tasks_list.append(n_tasks)
        except Exception as e:
            print(f"Error scoring prediction {idx}: {e}")
            scores.append(0.0)
            task_scores_list.append(0.0)
            kte_list.append(0.0)
            n_tasks_list.append(0)

    return {
        'mean_score': np.mean(scores) if scores else 0.0,
        'std_score': np.std(scores) if scores else 0.0,
        'mean_task_score': np.mean(task_scores_list) if task_scores_list else 0.0,
        'mean_kte': np.mean(kte_list) if kte_list else 0.0,
        'mean_n_tasks': np.mean(n_tasks_list) if n_tasks_list else 0.0,
        'scores': scores,
        'task_scores': task_scores_list,
        'kte': kte_list,
    }
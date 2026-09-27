import os
import importlib.util
import csv
import math
import numpy as np

# Import global configuration
import sys
sys.path.insert(0, '/hada')
from config import set_global_seed, SEED, MODEL_NAME

# Set global seed at module load time
set_global_seed()


def load_opt_vals():
    """加载 dataset.csv 文件，返回一个字典 {(function, instance, dimension): opt_val}"""
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset.csv')
    opt_vals = {}
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (int(row['function']), int(row['instance']), int(row['dimension']))
                opt_vals[key] = float(row['opt_val'])
    return opt_vals


def compute_ri_score(initial_gbest_f, gbest_f, opt_val):
    """计算 score = (initial_best - best) / (initial_best - opt)。
    
    其中：
    - initial_best = initial_gbest_f（PSO初始化后种群的最优值）
    - best = gbest_f（PSO运行结束后的最优值）
    - opt = opt_val（理论最优值）
    
    返回截断到 [0, 1] 的 score。
    """    
    if abs(initial_gbest_f - opt_val) < 1e-12:
        return 1.0 if gbest_f <= opt_val + 1e-12 else 0.0
    
    score = (initial_gbest_f - gbest_f) / (initial_gbest_f - opt_val)
    return max(0.0, min(1.0, score))


def run_de_and_score(inputs, log_fn=print, mode='test', controller=None, seed=None):
    """运行 DQN 控制的 DE 并计算 RI 评分。
    
    Args:
        inputs: dict from domains.bbob_unconstrained.utils.format_input_dict
        Expected keys: 'problem_id','dimension','instance'
        log_fn: logging function
        mode: 'train' for training DQN (updates model), 'test' for evaluation only
        controller: optional external controller (shared across problems)
        seed: random seed for DE initialization
    
    Returns:
        prediction (string) in format "score|gbest_f|initial_gbest_f|opt_val"
    """
    # lazy import cocoex and DE module from metabbo
    import cocoex

    this_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'metabbo')
    # 将 metabbo 目录添加到 sys.path，以便模块内的相对导入可以正常工作
    if this_dir not in __import__('sys').path:
        __import__('sys').path.insert(0, this_dir)
    # load local ec_algorithm module
    spec = importlib.util.spec_from_file_location('ec_mod', os.path.join(this_dir, 'ec_algorithm.py'))
    ec_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ec_mod)
    
    # Create or use provided controller
    if controller is None:
        # load meta learning implementations
        spec2 = importlib.util.spec_from_file_location('meta_mod', os.path.join(this_dir, 'meta_learning.py'))
        meta_mod = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(meta_mod)
        model_path = os.path.join(this_dir, 'dqn_model.pt')
        controller = meta_mod.DiscreteDQNController(model_path=model_path, training=(mode == 'train'), seed=seed)
    
    # prepare suite filter (dimension + instance)
    dim = int(inputs.get('dimension', 10))
    inst = int(inputs.get('instance', 1))
    max_evals = max(1000, dim * 500)

    # 从 problem_id 字符串中提取 function 编号（例如 f1_i1 -> 1）
    problem_id_str = inputs.get('problem_id', 'f1_i1')
    problem_id = int(str(problem_id_str).replace('f', '').split('_')[0])

    suite = cocoex.Suite('bbob', '', f'dimensions: {dim} instance_indices:{inst} function_indices: {problem_id}')
    # pick the matching problem from the suite
    problem = next(iter(suite))

    # run DE algorithm using DEOptimizer to get stats
    try:
        de_opt = ec_mod.DEOptimizer(problem.dimension, getattr(problem, 'lower_bounds', None), getattr(problem, 'upper_bounds', None), max_evals, params={'F': 0.5, 'CR': 0.5}, controller=controller, recorder=None, swarm_size=40)
        # 训练模式会更新 DQN 模型，测试模式只使用模型
        gbest, stats = de_opt.optimize(problem, train=(mode == 'train'), seed=seed)
        gbest_f = stats.get('gbest_f', None)
        initial_gbest_f = stats.get('initial_gbest_f', gbest_f)
        
        # 训练模式下保存模型（只在 controller 是内部创建时保存）
        if mode == 'train' and controller is not None and hasattr(controller, 'net'):
            import torch
            model_path = os.path.join(this_dir, 'dqn_model.pt')
            torch.save(controller.net.state_dict(), model_path)
            log_fn(f"[TRAIN] Model saved to {model_path}")
    except Exception as e:
        return f'error: DE run failed: {e}'

    # 计算 RI 分数
    # 加载理论最优值
    opt_vals = load_opt_vals()
    opt_val_key = (problem_id, inst, dim)
    opt_val = opt_vals.get(opt_val_key, None)
    
    # 使用 RI 评分方式
    score = compute_ri_score(initial_gbest_f, gbest_f, opt_val)
    
    mode_str = "TRAIN" if mode == 'train' else "TEST"
    log_fn(f"[{mode_str}] f{problem_id} Score: {score}")
    log_fn(f"[{mode_str}] f{problem_id} gbest_f: {gbest_f}")
    log_fn(f"[{mode_str}] f{problem_id} initial_gbest_f: {initial_gbest_f}")
    log_fn(f"[{mode_str}] f{problem_id} opt_val: {opt_val}")
    
    # 训练模式下收集 loss 和 return
    if mode == 'train':
        training_loss = controller.get_total_loss() if hasattr(controller, 'get_total_loss') else None
        sweep_return = controller.get_total_reward() if hasattr(controller, 'get_total_reward') else None
        if training_loss is not None:
            log_fn(f"[TRAIN] Total DQN loss: {training_loss:.6f}")
        if sweep_return is not None:
            log_fn(f"[TRAIN] Total DQN return: {sweep_return:.4f}")
        return {
            'score': score,
            'gbest_f': gbest_f,
            'initial_gbest_f': initial_gbest_f,
            'opt_val': opt_val,
            'training_loss': training_loss,
            'sweep_return': sweep_return,
            'controller': controller
        }
    
    # 返回格式："score|gbest_f|initial_gbest_f|opt_val" 以便 report.py 解析
    prediction = f"{score}|{gbest_f}|{initial_gbest_f}|{opt_val}"

    # 也返回 dict 格式供测试脚本使用
    return {
        'score': score,
        'gbest_f': gbest_f,
        'initial_gbest_f': initial_gbest_f,
        'opt_val': opt_val,
        'training_loss': None,
        'sweep_return': controller.get_total_reward() if hasattr(controller, 'get_total_reward') else None,
        'controller': controller
    }
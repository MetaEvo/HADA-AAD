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


def compute_ri_score(initial_gbest_f, gbest_f, opt_val, is_feasible=True):
    """计算 score = (initial_best - best) / (initial_best - opt)。
    
    对于约束优化问题：
    - 如果最终解不可行（is_feasible=False），分数为 0
    - 如果最终解可行，计算 score = (initial_best - best) / (initial_best - opt)
    """
    if not is_feasible:
        return 0.0
    
    if abs(initial_gbest_f - opt_val) < 1e-12:
        return 1.0 if gbest_f <= opt_val + 1e-12 else 0.0
    
    score = (initial_gbest_f - gbest_f) / (initial_gbest_f - opt_val)
    return max(0.0, min(1.0, score))


def run_pso_and_score(inputs, log_fn=print, mode='test'):
    """运行 DQN 控制的 PSO 并计算 RI 评分。
    
    Args:
        inputs: dict from domains.bbob_constrained.utils.format_input_dict
        Expected keys: 'problem_id','dimension','instance','constraint_num'
        log_fn: logging function
        mode: 'train' for training DQN (updates model), 'test' for evaluation only
    
    Returns:
        prediction (string) in format "score|gbest_f|initial_gbest_f|opt_val"
    """
    # lazy import cocoex and pso module from metabbo
    import cocoex

    this_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'metabbo')
    # 将 metabbo 目录添加到 sys.path，以便模块内的相对导入可以正常工作
    if this_dir not in __import__('sys').path:
        __import__('sys').path.insert(0, this_dir)
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

    # prepare suite filter (dimension + instance)
    dim = int(inputs.get('dimension', 10))
    inst = int(inputs.get('instance', 1))
    max_evals = 10000

    # 从 problem_id 字符串中提取 function 编号（例如 f1_i1 -> 1）
    problem_id_str = inputs.get('problem_id', 'f1_i1')
    problem_id = int(str(problem_id_str).replace('f', '').split('_')[0])

    suite = cocoex.Suite('bbob-constrained', '', f'dimensions: {dim} instance_indices:{inst} function_indices: {problem_id}')
    # pick the matching problem from the suite
    problem = next(iter(suite))

    # build controller: prefer discrete DQN if available and model exists
    model_path = os.path.join(this_dir, 'dqn_model.pt')
    controller = dqn_mod.DiscreteDQNController(model_path=model_path)
    # run PSO using PSOOptimizer to get stats
    try:
        pso_opt = pso_mod.PSOOptimizer(problem.dimension, getattr(problem, 'lower_bounds', None), getattr(problem, 'upper_bounds', None), max_evals, params={'w': 0.729, 'c1': 1.49445, 'c2': 1.49445}, controller=controller, recorder=None, swarm_size=50)
        # 训练模式会更新 DQN 模型，测试模式只使用模型
        gbest, stats = pso_opt.optimize(problem, train=(mode == 'train'))
        gbest_f = stats.get('gbest_f', None)
        official_initial_f = stats.get('initial_gbest_f', gbest_f)  # f(x0) 官方初始解
        initial_population_best = stats.get('initial_population_best', gbest_f)
        initial_population_positions = stats.get('initial_population_positions', [])
        initial_population_fitness = stats.get('initial_population_fitness', [])
        
        # 训练模式下保存模型
        if mode == 'train' and hasattr(controller, 'net'):
            import torch
            torch.save(controller.net.state_dict(), model_path)
            log_fn(f"[TRAIN] Model saved to {model_path}")
    except Exception as e:
        return f'error: pso run failed: {e}'

    # 评估初始种群最优解的可行性
    # cocoex bbob-constrained API:
    # - problem(x) 只返回目标函数值 (float)
    # - problem.constraint(x) 返回约束值数组 (numpy.ndarray)
    # 约束满足条件：所有约束值 <= 0
    mode_str = "TRAIN" if mode == 'train' else "TEST"
    
    # 检查初始种群最优解是否可行
    initial_population_feasible = False
    if initial_population_positions and initial_population_fitness:
        # 找到初始种群中适应度最好的粒子的索引
        best_idx = int(np.argmin(initial_population_fitness))
        best_initial_x = np.array(initial_population_positions[best_idx], dtype=float)
        
        try:
            constraints = problem.constraint(best_initial_x)
            if hasattr(constraints, '__len__'):
                initial_population_feasible = all(float(g) <= 1e-10 for g in constraints)
            else:
                initial_population_feasible = float(constraints) <= 1e-10
        except Exception as e:
            log_fn(f"[{mode_str}] f{problem_id} WARNING: checking initial population feasibility failed: {e}")
    
    # 如果初始种群最优解不可行，使用官方初始解
    if initial_population_feasible:
        initial_best = initial_population_best
        log_fn(f"[{mode_str}] f{problem_id} Initial population best is feasible: {initial_best:.6f}")
    else:
        initial_best = official_initial_f
        log_fn(f"[{mode_str}] f{problem_id} Initial population infeasible, using official initial solution: {initial_best:.6f}")
    
    # 从最终种群中选择最优可行解
    is_feasible = False
    best_feasible_x = None
    best_feasible_f = float('inf')
    
    final_pbest = stats.get('final_pbest', [])
    final_pbest_f = stats.get('final_pbest_f', [])
    
    if final_pbest and final_pbest_f:
        log_fn(f"[{mode_str}] f{problem_id} Checking {len(final_pbest)} particles for feasible solutions...")
        
        for i, (x_particle, f_particle) in enumerate(zip(final_pbest, final_pbest_f)):
            if x_particle is None:
                continue
            x = np.array(x_particle, dtype=float)
            
            # 获取约束值
            try:
                constraints = problem.constraint(x)
                
                # 检查是否所有约束都满足（每个约束值 <= 0）
                if hasattr(constraints, '__len__'):
                    is_particle_feasible = all(float(g) <= 1e-10 for g in constraints)
                else:
                    is_particle_feasible = float(constraints) <= 1e-10
                
                # 如果是可行解且目标函数值更好，则更新最优可行解
                if is_particle_feasible and f_particle < best_feasible_f:
                    best_feasible_x = x_particle
                    best_feasible_f = f_particle
                    is_feasible = True
                    log_fn(f"[{mode_str}] f{problem_id} Particle {i}: feasible, f={f_particle:.6f}")
                elif not is_particle_feasible:
                    if hasattr(constraints, '__len__'):
                        violated = [float(g) for g in constraints if float(g) > 1e-10]
                        log_fn(f"[{mode_str}] f{problem_id} Particle {i}: infeasible, violated constraints: {violated}")
                    else:
                        log_fn(f"[{mode_str}] f{problem_id} Particle {i}: infeasible, constraint={float(constraints):.6f}")
            except Exception as e:
                log_fn(f"[{mode_str}] f{problem_id} Particle {i}: error checking constraints: {e}")
        
        # 如果找到了可行解，使用它作为最终解
        if best_feasible_x is not None:
            log_fn(f"[{mode_str}] f{problem_id} Found feasible solution! f={best_feasible_f:.6f}")
            gbest = best_feasible_x
            gbest_f = best_feasible_f
        else:
            log_fn(f"[{mode_str}] f{problem_id} WARNING: No feasible solution found in population! Using initial solution.")
            # 如果没有可行解，使用初始解
            is_feasible = True
            gbest_f = initial_best
    else:
        # 如果没有种群信息，回退到原始方法
        if gbest is not None:
            x = np.array(gbest, dtype=float)
            try:
                constraints = problem.constraint(x)
                if hasattr(constraints, '__len__'):
                    is_feasible = all(float(g) <= 1e-10 for g in constraints)
                else:
                    is_feasible = float(constraints) <= 1e-10
                # 如果 gbest 不可行，使用初始解
                if not is_feasible:
                    log_fn(f"[{mode_str}] f{problem_id} gbest is infeasible, using initial solution")
                    gbest_f = initial_best
                    is_feasible = True
            except Exception as e:
                log_fn(f"[{mode_str}] f{problem_id} WARNING: problem.constraint(x) failed: {e}")
                gbest_f = initial_best
                is_feasible = True
    
    log_fn(f"[{mode_str}] f{problem_id} Final solution: feasible={is_feasible}, f={gbest_f}")

    # 计算 score（基于最优可行解）
    # 加载理论最优值
    opt_vals = load_opt_vals()
    opt_val_key = (problem_id, inst, dim)
    opt_val = opt_vals.get(opt_val_key, None)
    
    # 使用新评分方式，只有可行解才有分数
    score = compute_ri_score(initial_best, gbest_f, opt_val, is_feasible)
    
    log_fn(f"[{mode_str}] f{problem_id} Score: {score}")
    log_fn(f"[{mode_str}] f{problem_id} gbest_f: {gbest_f}")
    log_fn(f"[{mode_str}] f{problem_id} initial_best: {initial_best}")
    log_fn(f"[{mode_str}] f{problem_id} opt_val: {opt_val}")
    log_fn(f"[{mode_str}] f{problem_id} is_feasible: {is_feasible}")
    
    # 返回格式："score|gbest_f|initial_best|opt_val" 以便 report.py 解析
    prediction = f"{score}|{gbest_f}|{initial_best}|{opt_val}"

    return prediction
"""
MetaBox MO Domain - DQN-PSO Utility
多目标黑箱优化的 DQN-PSO 工具函数
"""
import os
import sys
import json
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
from domains.metabox_mo.utils import (
    compute_hypervolume, compute_igd, compute_gd, compute_spacing, non_dominated_sort,
    crowding_distance, normalize_objectives, dominates
)


def run_de_and_score(inputs: Dict, log_fn=None, mode='train', controller=None, seed=None) -> str:
    """
    运行 DE 并计算多目标优化分数

    Args:
        inputs: 包含 problem_id 等信息的字典
        log_fn: 日志函数
        mode: 'train' 或 'test'
        controller: optional external controller (shared across problems)
        seed: random seed for DE initialization

    Returns:
        prediction (string) in format "score|hypervolume|igd|n_solutions"
    """
    problem_id = inputs.get('problem_id', 'unknown')

    if log_fn:
        log_fn(f"[{mode.upper()}] {problem_id} Running multi-objective DE optimization...")

    try:
        # 动态加载 DE 和 DQN controller (参考 bbob_constrained)
        this_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'metabbo')
        if this_dir not in sys.path:
            sys.path.insert(0, this_dir)

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

        # 创建 MetaBox MO 问题
        problem = create_metabox_mo_problem(inputs)

        if problem is None:
            return "0.0|0.0|inf|0"

        # 运行 DQN-DE
        result = run_dqn_de_mo(problem, ec_mod, controller, log_fn, mode, seed=seed)

        # 返回格式: score|hv|n_solutions
        score = result.get('score', 0.0)
        hv = result.get('hypervolume', 0.0)
        n_solutions = result.get('n_solutions', 0)

        return {
            'prediction': f"{score}|{hv}|{n_solutions}",
            'score': score,
            'hypervolume': hv,
            'n_solutions': n_solutions,
            'controller': controller
        }

    except Exception as e:
        error_msg = f"Error in run_de_and_score: {str(e)}\n{traceback.format_exc()}"
        if log_fn:
            log_fn(error_msg)
        return "0.0|0.0|inf|0"


class PymooWFGWrapper:
    """
    Wrapper for pymoo standard WFG problems to match MetaBox problem interface
    
    IMPORTANT: 
    - WFG1-WFG9 are separate classes in pymoo
    - M=5 objectives, D=28 dimensions (k=8, l=20)
    - Input bounds: xl=[0]*D, xu=[2, 4, 6, ..., 2*(i+1)] (pymoo's actual bounds)
    """
    def __init__(self, wfg_class, problem_id, n_var=28, n_obj=5, k=8, l=20):
        self.wfg = wfg_class(n_var=n_var, n_obj=n_obj, k=k, l=l)
        self.problem_id = problem_id
        self.dim = n_var
        self.n_var = n_var
        self.n_obj = n_obj
        self.k = k
        self.l = l
        # Use actual bounds from pymoo WFG
        self.lb = self.wfg.xl
        self.ub = self.wfg.xu
    
    def func(self, x):
        """
        Evaluate WFG problem
        
        Args:
            x: 2D array [NP, dim] with values in [xl, xu]
        
        Returns:
            2D array [NP, n_obj] with objective values
        """
        # Ensure 2D array
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        return self.wfg.evaluate(x)


def create_metabox_mo_problem(inputs: Dict) -> Optional[Any]:
    """
    创建 MetaBox MO pymoo标准WFG问题实例
    
    WFG多目标优化问题（WFG1-9，28 维，5 目标）
    - dim = 28
    - n_obj = 5
    - k = 8 (position parameters)
    - l = 20 (distance parameters)
    - difficult 模式划分：
      - train: WFG1, WFG4, WFG7
      - test: WFG2, WFG3, WFG5, WFG6, WFG8, WFG9
    """
    try:
        # 1. 解析传入的测试信息
        problem_id = inputs.get('problem_id', '')
        suite = str(inputs.get('suite', 'wfg')).lower()
        instance = int(inputs.get('instance', 0))
        dimension = inputs.get('dimension', 28)
        n_objectives = inputs.get('n_objectives', 5)

        # 2. 导入pymoo标准WFG问题
        from pymoo.problems.many.wfg import WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9
        
        # WFG参数设置
        M = n_objectives  # 5 objectives
        k = 8             # 8 position parameters
        l = 20            # 20 distance parameters
        D = k + l         # 28 dimensions
        
        wfg_classes = [WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9]
        
        # 3. 根据instance获取对应的问题
        # instance 0-8: WFG1-9
        # train: WFG1, WFG4, WFG7 (instance 0, 3, 6)
        # test: WFG2, WFG5, WFG8 (instance 1, 4, 7)
        if instance < len(wfg_classes):
            wfg_class = wfg_classes[instance]
            return PymooWFGWrapper(
                wfg_class=wfg_class,
                problem_id=instance + 1,
                n_var=D,
                n_obj=M,
                k=k,
                l=l
            )
        
        # 如果没找到，返回WFG1
        return PymooWFGWrapper(
            wfg_class=WFG1,
            problem_id=1,
            n_var=D,
            n_obj=M,
            k=k,
            l=l
        )

    except Exception as e:
        print(f"Error creating MetaBox MO problem: {e}")
        import traceback
        print(traceback.format_exc())
        return None


def run_dqn_de_mo(problem: Any, ec_mod: Any, controller,
                   log_fn=None, mode: str = 'train', seed=None) -> Dict:
    """
    运行 DQN-DE 进行多目标优化

    使用 DE 优化，返回最终种群计算 HV
    """
    try:
        mode_str = "TRAIN" if mode == 'train' else "TEST"

        # 获取问题维度
        dim = getattr(problem, 'dim', None) or getattr(problem, 'n_var', 30)
        max_evals = 2000

        # 获取边界
        lower_bounds = getattr(problem, 'lb', None)
        upper_bounds = getattr(problem, 'ub', None)

        if lower_bounds is None:
            lower_bounds = np.zeros(dim, dtype=float)
        else:
            lower_bounds = np.asarray(lower_bounds, dtype=float)
        
        if upper_bounds is None:
            upper_bounds = np.ones(dim, dtype=float)
        else:
            upper_bounds = np.asarray(upper_bounds, dtype=float)

        model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'metabbo', 'dqn_model.pt')

        # 使用 DEOptimizer
        de_opt = ec_mod.DEOptimizer(
            dim,
            lower_bounds,
            upper_bounds,
            max_evals,
            params={'F': 0.5, 'CR': 0.5},
            controller=controller,
            recorder=None,
            swarm_size=50
        )

        # 运行 DE 优化（只运行一次）
        gbest, stats = de_opt.optimize(problem, train=(mode == 'train'), seed=seed)
        
        # 模型保存由 harness.py 在 sweep 结束后统一处理，不在这里保存

        # 获取所有 archive（Agent 修改 ec_algorithm.py 后应返回 archive_positions）
        archive_positions = stats.get('archive_positions', [])
        
        # 如果没有 archive，回退到最终种群
        if len(archive_positions) == 0:
            archive_positions = stats.get('final_pbest', [])
            archive_fitness = stats.get('final_pbest_f', [])

        if log_fn:
            log_fn(f"[{mode_str}] Archive size: {len(archive_positions)}")

        # 评估所有 archive 的解
        best_solutions = []
        for x in archive_positions:
            if x is not None:
                try:
                    x_arr = np.asarray(x, dtype=float)
                    # CRITICAL: MetaBox problem.func expects 2D array [NP, dim]
                    if x_arr.ndim == 1:
                        x_arr = x_arr.reshape(1, -1)
                    obj_vector = problem.func(x_arr)
                    # Convert back to 1D if single solution
                    obj_vector = np.array(obj_vector)
                    if obj_vector.ndim == 2 and obj_vector.shape[0] == 1:
                        obj_vector = obj_vector[0]
                    obj_vector = obj_vector.flatten()
                    if hasattr(obj_vector, '__len__') and len(obj_vector) >= 2:
                        best_solutions.append({
                            'x': x,
                            'f': obj_vector
                        })
                except Exception as e:
                    if log_fn:
                        log_fn(f"[{mode_str}] Error evaluating solution: {e}")
                    pass

        if log_fn:
            log_fn(f"[{mode_str}] Evaluated {len(best_solutions)} solutions")

        # 提取 Pareto 前沿
        if len(best_solutions) == 0:
            return {
                'score': 0.0,
                'hypervolume': 0.0,
                'igd': float('inf'),
                'n_solutions': 0,
            }

        objectives = np.array([s['f'] for s in best_solutions])

        # 非支配排序获取 Pareto 前沿
        fronts, _ = non_dominated_sort(objectives)

        if len(fronts) == 0 or len(fronts[0]) == 0:
            return {
                'score': 0.0,
                'hypervolume': 0.0,
                'igd': float('inf'),
                'n_solutions': 0,
            }

        pareto_front = objectives[fronts[0]]

        if log_fn:
            log_fn(f"[{mode_str}] Pareto front size: {len(pareto_front)}")

        # WFG 归一化（使用理论 ideal/nadir 参考点）
        # M=5: ideal=[2,4,6,8,10], nadir=[10,20,30,40,50]
        
        M = len(pareto_front[0])  # 目标数
        
        # pymoo WFG M=5: objectives scaled to [0, 2*i] for i-th objective
        z_ideal_raw = np.array([0, 0, 0, 0, 0], dtype=float)[:M]
        z_nadir_raw = np.array([2, 4, 6, 8, 10], dtype=float)[:M]
        
        if log_fn:
            log_fn(f"[{mode_str}] Using pymoo WFG ideal/nadir for M={M}")
        
        # 归一化 Pareto 前沿
        delta = z_nadir_raw - z_ideal_raw
        delta = np.where(delta < 1e-10, 1.0, delta)
        pareto_front_norm = (pareto_front - z_ideal_raw) / delta
        
        # HV 计算（归一化空间）
        from pymoo.indicators.hv import HV
        hv_ref = np.ones(M) * 1.1
        hv_indicator = HV(ref_point=hv_ref)
        raw_hv = hv_indicator(pareto_front_norm)
        
        # 归一化 HV：除以 1.1^M
        ref_volume = 1.1 ** M
        hv = raw_hv / ref_volume

        if log_fn:
            log_fn(f"[{mode_str}] Normalized HV: {hv:.6f} (Raw HV={raw_hv:.4f})")
            log_fn(f"[{mode_str}] WFG ideal (raw): {z_ideal_raw}")
            log_fn(f"[{mode_str}] WFG nadir (raw): {z_nadir_raw}")
            log_fn(f"[{mode_str}] HV ref (normalized): {hv_ref}")

        # 得分 = 归一化 HV（范围 [0, 1]）
        score = hv

        if log_fn:
            log_fn(f"[{mode_str}] Score: {score:.6f} (Normalized HV)")

        result = {
            'score': score,
            'hypervolume': hv,
            'n_solutions': len(pareto_front),
            'pareto_front': pareto_front.tolist(),
        }

        # 训练模式下收集 loss 和 return
        if mode == 'train':
            training_loss = controller.get_total_loss() if hasattr(controller, 'get_total_loss') else None
            sweep_return = controller.get_total_reward() if hasattr(controller, 'get_total_reward') else None
            result['training_loss'] = training_loss
            result['sweep_return'] = sweep_return
            if training_loss is not None and log_fn:
                log_fn(f"[TRAIN] Total DQN loss: {training_loss:.6f}")
            if sweep_return is not None and log_fn:
                log_fn(f"[TRAIN] Total DQN return: {sweep_return:.4f}")

        return result

    except Exception as e:
        if log_fn:
            log_fn(f"[{mode_str}] Error in run_dqn_pso_mo: {e}")
            log_fn(traceback.format_exc())
        return {
            'score': 0.0,
            'hypervolume': 0.0,
            'n_solutions': 0,
            'error': str(e)
        }


def generate_approximate_pareto_front(problem: Any, n_points: int = 1000) -> Optional[np.ndarray]:
    """生成近似的真实 Pareto 前沿"""
    try:
        # 对于 ZDT1，可以解析计算真实 Pareto 前沿
        if hasattr(problem, 'func_id') and problem.func_id == 1:
            f1 = np.linspace(0, 1, n_points)
            f2 = 1 - np.sqrt(f1)
            return np.column_stack([f1, f2])

        # 对于其他问题，使用随机采样
        n_obj = problem.n_obj if hasattr(problem, 'n_obj') else 2
        dim = problem.dim if hasattr(problem, 'dim') else 30

        samples = []
        for _ in range(n_points):
            x = np.random.uniform(0, 1, dim)
            # MO 问题使用 func() 方法而不是直接调用
            obj_vector = problem.func(x)
            obj_vector = np.array(obj_vector).flatten()
            samples.append(obj_vector)

        objectives = np.array(samples)
        fronts, _ = non_dominated_sort(objectives)

        if len(fronts) > 0 and len(fronts[0]) > 0:
            return objectives[fronts[0]]

        return None

    except Exception as e:
        print(f"Error generating Pareto front: {e}")
        return None


def score_predictions(predictions: pd.DataFrame, mode: str = 'train') -> Dict:
    """
    计算多组预测的分数

    Args:
        predictions: DataFrame 包含预测结果
        mode: 'train' 或 'test'

    Returns:
        包含平均分数和详细信息的字典
    """
    if predictions.empty:
        return {
            'score': 0.0,
            'mean_hypervolume': 0.0,
            'mean_igd': float('inf'),
            'total': 0
        }

    # 解析预测结果
    scores = []
    hypervolumes = []
    igds = []

    for _, row in predictions.iterrows():
        try:
            # prediction 格式: score|hv|n_solutions
            parts = str(row['prediction']).split('|')
            if len(parts) >= 3:
                scores.append(float(parts[0]))
                hypervolumes.append(float(parts[1]))
        except:
            continue

    if len(scores) == 0:
        return {
            'score': 0.0,
            'mean_hypervolume': 0.0,
            'mean_igd': float('inf'),
            'total': len(predictions)
        }

    return {
        'score': np.mean(scores),
        'mean_hypervolume': np.mean(hypervolumes),
        'mean_igd': np.mean(igds),
        'total': len(predictions),
        'mean_score': np.mean(scores)
    }
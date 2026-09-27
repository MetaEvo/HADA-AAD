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


def _load_unified_reference_point() -> Optional[np.ndarray]:
    """从 reference.csv 加载统一参考点"""
    try:
        ref_path = os.path.join(os.path.dirname(__file__), 'reference.csv')
        if not os.path.exists(ref_path):
            return None
        
        df = pd.read_csv(ref_path)
        if 'problem_id' not in df.columns or 'n_objectives' not in df.columns:
            return None
        
        # 查找统一参考点行
        ref_row = df[df['problem_id'] == 'unified_reference']
        if ref_row.empty:
            return None
        
        n_obj = int(ref_row['n_objectives'].iloc[0])
        ref_point = np.array([ref_row[f'obj_{i+1}'].iloc[0] for i in range(n_obj)])
        return ref_point
    except Exception:
        return None


def run_pso_and_score(inputs: Dict, log_fn=None, mode='train') -> str:
    """
    运行 PSO 并计算多目标优化分数

    Args:
        inputs: 包含 problem_id 等信息的字典
        log_fn: 日志函数
        mode: 'train' 或 'test'

    Returns:
        prediction (string) in format "score|hypervolume|igd|n_solutions"
    """
    problem_id = inputs.get('problem_id', 'unknown')

    if log_fn:
        log_fn(f"[{mode.upper()}] {problem_id} Running multi-objective PSO optimization...")

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

        # 创建 MetaBox MO 问题
        problem = create_metabox_mo_problem(inputs)

        if problem is None:
            return "0.0|0.0|inf|0"

        # 运行 DQN-PSO
        result = run_dqn_pso_mo(problem, pso_mod, dqn_mod, log_fn, mode)

        # 返回格式: score|hypervolume|igd|gd|spacing|n_solutions
        score = result.get('score', 0.0)
        hv = result.get('hypervolume', 0.0)
        igd = result.get('igd', float('inf'))
        gd = result.get('gd', float('inf'))
        spacing = result.get('spacing', 0.0)
        n_solutions = result.get('n_solutions', 0)

        return f"{score}|{hv}|{igd}|{gd}|{spacing}|{n_solutions}"

    except Exception as e:
        error_msg = f"Error in run_pso_and_score: {str(e)}\n{traceback.format_exc()}"
        if log_fn:
            log_fn(error_msg)
        return "0.0|0.0|inf|0"


def create_metabox_mo_problem(inputs: Dict) -> Optional[Any]:
    """
    创建 MetaBox MO UAV 问题实例
    
    UAV 路径规划问题（多目标优化，5 个目标）
    - dim = 3 * dv（dv=10 时 dim=30）
    - difficult 模式划分（seed=42）：
      - train: 偶数 ID (0, 2, 4, ..., 54) - 28 个
      - test: 奇数 ID (1, 3, 5, ..., 55) - 28 个
    """
    try:
        # 1. 解析传入的测试信息
        problem_id = inputs.get('problem_id', '')
        suite = str(inputs.get('suite', 'uav')).lower()
        instance = int(inputs.get('instance', 0))
        dimension = inputs.get('dimension', 30)

        # 2. 加载 UAV 数据集
        from metaevobox.environment.problem.MOO.UAV.uav_dataset import UAV_Dataset
        
        model_path = '/hada/domains/metabox_mo/Model56.pkl'
        
        train_set, test_set = UAV_Dataset.get_datasets(
            version='numpy',
            difficulty='difficult',
            dv=10,  # 航点数量，dim = 3 * 10 = 30
            j_pen=1e4,
            seed=42,
            num=56,
            mode='standard',  # 使用 standard 模式，加载 model56.pkl 地形数据
            path=model_path
        )
        
        # 3. 根据 instance 获取对应的问题
        # instance 对应 UAV 的 problem_id（从 0 开始，0-55）
        # difficult 模式划分（seed=42）：
        # train: 偶数 ID (0, 2, 4, ..., 54) - 28 个
        # test: 奇数 ID (1, 3, 5, ..., 55) - 28 个
        
        all_problems = train_set.data + test_set.data
        
        # instance 从 0 开始，对应 ID = instance
        target_id = instance
        
        for problem in all_problems:
            if hasattr(problem, 'problem_id') and problem.problem_id == target_id:
                return problem
            elif hasattr(problem, 'id') and problem.id == target_id:
                return problem
        
        # 如果没找到，返回第一个
        if len(all_problems) > 0:
            return all_problems[0]
        
        return None

    except Exception as e:
        print(f"Error creating MetaBox MO problem: {e}")
        import traceback
        print(traceback.format_exc())
        return None


def run_dqn_pso_mo(problem: Any, pso_mod: Any, dqn_mod: Any,
                   log_fn=None, mode: str = 'train') -> Dict:
    """
    运行 DQN-PSO 进行多目标优化

    使用 PSO 优化，返回最终种群计算 HV
    """
    try:
        mode_str = "TRAIN" if mode == 'train' else "TEST"

        # 获取问题维度
        dim = getattr(problem, 'dim', None) or getattr(problem, 'n_var', 30)
        max_evals = 2500

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

        # build controller: prefer discrete DQN if available and model exists
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'metabbo', 'dqn_model.pt')
        controller = dqn_mod.DiscreteDQNController(model_path=model_path)

        # 使用 PSOOptimizer
        pso_opt = pso_mod.PSOOptimizer(
            dim,
            lower_bounds,
            upper_bounds,
            max_evals,
            params={'w': 0.729, 'c1': 1.49445, 'c2': 1.49445},
            controller=controller,
            recorder=None,
            swarm_size=50
        )

        # 运行 PSO 优化（只运行一次）
        gbest, stats = pso_opt.optimize(problem, train=(mode == 'train'))
        
        # 训练模式下保存模型
        if mode == 'train' and hasattr(controller, 'net'):
            import torch
            torch.save(controller.net.state_dict(), model_path)
            if log_fn:
                log_fn(f"[TRAIN] Model saved to {model_path}")

        # 获取最终种群
        final_pbest = stats.get('final_pbest', [])
        final_pbest_f = stats.get('final_pbest_f', [])

        if log_fn:
            log_fn(f"[{mode_str}] Final population size: {len(final_pbest)}")

        # 评估最终种群的所有解
        best_solutions = []
        for x in final_pbest:
            if x is not None:
                try:
                    x_arr = np.asarray(x, dtype=float)
                    obj_vector = problem.func(x_arr)
                    obj_vector = np.array(obj_vector).flatten()
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

        # 计算参考点：使用统一参考点（从 reference.csv 读取）
        reference_point = _load_unified_reference_point()
        if reference_point is None:
            # 如果没有参考点文件，使用当前种群最大值的 1.1 倍作为回退
            reference_point = np.max(objectives, axis=0) * 1.1
            if log_fn:
                log_fn(f"[{mode_str}] WARNING: No reference.csv found, using dynamic reference point")

        # 计算归一化 Hypervolume：HV / (参考点围成的超立方体体积)
        from pymoo.indicators.hv import HV
        hv_indicator = HV(ref_point=reference_point)
        raw_hv = hv_indicator(pareto_front)
        ref_volume = np.prod(reference_point)
        hv = raw_hv / ref_volume

        if log_fn:
            log_fn(f"[{mode_str}] Normalized HV: {hv:.6f} (Raw HV={raw_hv:.4f}, Ref Volume={ref_volume:.4e})")
            log_fn(f"[{mode_str}] Reference point: {reference_point}")

        # 得分 = 归一化 HV（范围 [0, 1]）
        score = hv

        if log_fn:
            log_fn(f"[{mode_str}] Score: {score:.6f} (Normalized HV)")

        return {
            'score': score,
            'hypervolume': hv,
            'igd': 0.0,
            'gd': 0.0,
            'spacing': 0.0,
            'n_solutions': len(pareto_front),
            'pareto_front': pareto_front.tolist(),
        }

    except Exception as e:
        if log_fn:
            log_fn(f"[{mode_str}] Error in run_dqn_pso_mo: {e}")
            log_fn(traceback.format_exc())
        return {
            'score': 0.0,
            'hypervolume': 0.0,
            'igd': float('inf'),
            'gd': float('inf'),
            'spacing': 0.0,
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
            # prediction 格式: score|hypervolume|igd|n_solutions
            parts = str(row['prediction']).split('|')
            if len(parts) >= 4:
                scores.append(float(parts[0]))
                hypervolumes.append(float(parts[1]))
                igds.append(float(parts[2]))
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